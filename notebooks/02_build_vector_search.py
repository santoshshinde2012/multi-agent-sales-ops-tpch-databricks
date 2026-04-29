# Databricks notebook source
# MAGIC %md
# MAGIC # Step 3 — Build the Vector Search Index
# MAGIC
# MAGIC Chunks the `o_comment` field of `samples.tpch.orders`, embeds with
# MAGIC `databricks-bge-large-en`, and lands the chunks in `comment_index`.
# MAGIC
# MAGIC Then creates a Delta Sync index on top of that table so the knowledge
# MAGIC specialist can query it through `search_order_comments`.

# COMMAND ----------
# MAGIC %pip install databricks-vectorsearch
# MAGIC %restart_python

# COMMAND ----------
dbutils.widgets.text("working_catalog",   "main")
dbutils.widgets.text("working_schema",    "sales_ops")
dbutils.widgets.text("vs_endpoint",       "customer_ops_endpoint")
dbutils.widgets.text("vs_index_suffix",   "comment_index_vs")

working_catalog = dbutils.widgets.get("working_catalog")
working_schema  = dbutils.widgets.get("working_schema")
vs_endpoint     = dbutils.widgets.get("vs_endpoint")
vs_index        = f"{working_catalog}.{working_schema}.{dbutils.widgets.get('vs_index_suffix')}"
src_table       = f"{working_catalog}.{working_schema}.comment_index"

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Chunk and write to the source table
# MAGIC TPC-H comments are short by design; we treat each whole comment as one chunk.

# COMMAND ----------
from pyspark.sql import functions as F

(spark.table("samples.tpch.orders")
      .filter(F.col("o_comment").isNotNull())
      .selectExpr(
          "concat('order_', cast(o_orderkey as string)) AS chunk_id",
          "o_orderkey",
          "o_comment AS content",
      )
      .withColumn("embedding", F.lit(None).cast("array<float>"))
      .write.mode("overwrite").saveAsTable(src_table))

print(f"Wrote {spark.table(src_table).count()} rows to {src_table}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Create the Delta Sync vector index

# COMMAND ----------
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

vsc.create_delta_sync_index(
    endpoint_name=vs_endpoint,
    index_name=vs_index,
    primary_key="chunk_id",
    source_table_name=src_table,
    pipeline_type="TRIGGERED",
    embedding_source_column="content",
    embedding_model_endpoint_name="databricks-bge-large-en",
)
print(f"Vector index {vs_index} created on endpoint {vs_endpoint}")
