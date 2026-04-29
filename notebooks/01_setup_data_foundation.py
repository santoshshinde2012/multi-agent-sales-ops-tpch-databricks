# Databricks notebook source
# MAGIC %md
# MAGIC # Step 1 — Data Foundation in Unity Catalog
# MAGIC
# MAGIC This notebook executes `sql/01_data_foundation.sql` to create the
# MAGIC `sales_ops` working schema, register views over `samples.tpch`, add
# MAGIC business-language column comments, and register the Vector Search source
# MAGIC table.
# MAGIC
# MAGIC **Run this once per environment.** It is idempotent — safe to re-run.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Parameters

# COMMAND ----------
dbutils.widgets.text("working_catalog", "main")
dbutils.widgets.text("working_schema",  "sales_ops")

working_catalog = dbutils.widgets.get("working_catalog")
working_schema  = dbutils.widgets.get("working_schema")
print(f"Setting up {working_catalog}.{working_schema}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Run the SQL script

# COMMAND ----------
from pathlib import Path

sql_path = Path("../sql/01_data_foundation.sql")
sql_text = sql_path.read_text()
sql_text = (sql_text
            .replace("${working_catalog}", working_catalog)
            .replace("${working_schema}",  working_schema))

for statement in [s.strip() for s in sql_text.split(";") if s.strip()]:
    spark.sql(statement)

print(f"Data foundation ready in {working_catalog}.{working_schema}")
