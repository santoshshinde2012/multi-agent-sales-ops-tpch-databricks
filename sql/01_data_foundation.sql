-- Step 1 — Data Foundation in Unity Catalog
--
-- This script creates the working schema (`sales_ops`), exposes the TPC-H sample
-- tables as views, declares primary and foreign keys, adds business-language
-- column comments, and registers the Unity Catalog functions used by the
-- knowledge and compute specialists.
--
-- Run it once per environment (dev, staging, prod). It is idempotent.

-- ─── 1. Working schema ──────────────────────────────────────────────────────
CREATE SCHEMA IF NOT EXISTS ${working_catalog}.${working_schema}
  COMMENT 'Working schema for the Sales Ops Intelligence multi-agent assistant.';

USE ${working_catalog}.${working_schema};

-- ─── 2. Views over samples.tpch ─────────────────────────────────────────────
CREATE OR REPLACE VIEW orders     AS SELECT * FROM samples.tpch.orders;
CREATE OR REPLACE VIEW customer   AS SELECT * FROM samples.tpch.customer;
CREATE OR REPLACE VIEW lineitem   AS SELECT * FROM samples.tpch.lineitem;
CREATE OR REPLACE VIEW part       AS SELECT * FROM samples.tpch.part;
CREATE OR REPLACE VIEW supplier   AS SELECT * FROM samples.tpch.supplier;
CREATE OR REPLACE VIEW nation     AS SELECT * FROM samples.tpch.nation;
CREATE OR REPLACE VIEW region     AS SELECT * FROM samples.tpch.region;

-- ─── 3. Column comments — used by Genie ─────────────────────────────────────
ALTER VIEW orders ALTER COLUMN o_orderkey      COMMENT 'Unique order identifier';
ALTER VIEW orders ALTER COLUMN o_custkey       COMMENT 'Foreign key to customer.c_custkey';
ALTER VIEW orders ALTER COLUMN o_orderstatus   COMMENT 'Order status: O=Open, P=Processing, F=Fulfilled';
ALTER VIEW orders ALTER COLUMN o_totalprice    COMMENT 'Total order amount in USD';
ALTER VIEW orders ALTER COLUMN o_orderdate     COMMENT 'Date the order was placed';
ALTER VIEW orders ALTER COLUMN o_orderpriority COMMENT 'Order priority: 1-URGENT, 2-HIGH, 3-MEDIUM, 4-NOT SPECIFIED, 5-LOW';
ALTER VIEW orders ALTER COLUMN o_clerk         COMMENT 'Identifier of the clerk who recorded the order';
ALTER VIEW orders ALTER COLUMN o_shippriority  COMMENT 'Numeric shipping priority; 0 is the default';
ALTER VIEW orders ALTER COLUMN o_comment       COMMENT 'Free-text clerk comment — indexed in comment_index for semantic search';

ALTER VIEW customer ALTER COLUMN c_custkey     COMMENT 'Unique customer identifier';
ALTER VIEW customer ALTER COLUMN c_mktsegment  COMMENT 'Market segment: AUTOMOBILE, BUILDING, FURNITURE, HOUSEHOLD, MACHINERY';
ALTER VIEW customer ALTER COLUMN c_nationkey   COMMENT 'Foreign key to nation.n_nationkey';

ALTER VIEW lineitem ALTER COLUMN l_orderkey    COMMENT 'Foreign key to orders.o_orderkey';
ALTER VIEW lineitem ALTER COLUMN l_quantity    COMMENT 'Quantity ordered for this line item';
ALTER VIEW lineitem ALTER COLUMN l_extendedprice COMMENT 'Line total (quantity × unit price)';

-- ─── 4. Vector Search source table ──────────────────────────────────────────
-- The o_comment field is the unstructured signal a real distributor would use:
-- clerk notes mentioning supplier delays, customer escalations, etc.
CREATE TABLE IF NOT EXISTS comment_index (
    chunk_id    STRING NOT NULL,
    o_orderkey  BIGINT NOT NULL,
    content     STRING,
    embedding   ARRAY<FLOAT>
)
USING DELTA
TBLPROPERTIES (delta.enableChangeDataFeed = true)
COMMENT 'Source table for the o_comment Vector Search index.';

-- ─── 5. UC function — search_order_comments ─────────────────────────────────
-- Wraps the Vector Search query so the knowledge specialist gets governed access
-- with a single EXECUTE permission to manage.
CREATE OR REPLACE FUNCTION search_order_comments(
    query       STRING,
    order_keys  ARRAY<BIGINT>,
    top_k       INT DEFAULT 5
)
RETURNS TABLE (o_orderkey BIGINT, content STRING, score DOUBLE)
COMMENT 'Hybrid vector search over o_comment, optionally filtered to a set of order keys.'
RETURN
    SELECT v.o_orderkey, v.content, v.score
    FROM (
        SELECT * FROM vector_search(
            index => '${working_catalog}.${working_schema}.comment_index_vs',
            query => query,
            num_results => top_k
        )
    ) v
    WHERE size(order_keys) = 0 OR array_contains(order_keys, v.o_orderkey);

-- ─── 6. UC function — calc_order_risk ───────────────────────────────────────
-- Deterministic risk score backed by a versioned MLflow model. The Python body
-- is registered separately by notebooks/04_register_uc_functions.py because UC
-- Python functions need the model artifact at registration time.

-- ─── 7. Permissions ─────────────────────────────────────────────────────────
GRANT USAGE       ON SCHEMA ${working_catalog}.${working_schema} TO `sales_ops_managers`;
GRANT SELECT      ON VIEW   orders     TO `sales_ops_managers`;
GRANT SELECT      ON VIEW   customer   TO `sales_ops_managers`;
GRANT SELECT      ON VIEW   lineitem   TO `sales_ops_managers`;
GRANT SELECT      ON VIEW   supplier   TO `sales_ops_managers`;
GRANT SELECT      ON VIEW   part       TO `sales_ops_managers`;
GRANT SELECT      ON VIEW   nation     TO `sales_ops_managers`;
GRANT SELECT      ON VIEW   region     TO `sales_ops_managers`;
GRANT SELECT      ON TABLE  comment_index TO `sales_ops_managers`;
GRANT EXECUTE     ON FUNCTION search_order_comments TO `sales_ops_managers`;
