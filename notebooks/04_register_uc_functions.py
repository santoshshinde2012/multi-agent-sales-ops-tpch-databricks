# Databricks notebook source
# MAGIC %md
# MAGIC # Step 4 — Register Unity Catalog Functions
# MAGIC
# MAGIC Registers `calc_order_risk` as a Python UC function that the
# MAGIC compute specialist can call. The function loads the `order_risk` MLflow
# MAGIC model from the Production stage (registered by the previous notebook).

# COMMAND ----------
dbutils.widgets.text("working_catalog", "main")
dbutils.widgets.text("working_schema",  "sales_ops")

ws_cat = dbutils.widgets.get("working_catalog")
ws_sch = dbutils.widgets.get("working_schema")

# COMMAND ----------
spark.sql(f"USE {ws_cat}.{ws_sch}")

spark.sql("""
CREATE OR REPLACE FUNCTION calc_order_risk(o_orderkey BIGINT)
RETURNS STRUCT<score: DOUBLE, drivers: ARRAY<STRING>>
LANGUAGE PYTHON
COMMENT 'Returns a 0-1 risk score and top drivers for an open order.'
AS $$
    import mlflow
    from pyspark.sql import SparkSession

    spark = SparkSession.getActiveSession()
    features_df = (
        spark.table("samples.tpch.orders").alias("o")
             .join(spark.table("samples.tpch.lineitem").alias("l"),
                   "o.o_orderkey == l.l_orderkey")
             .filter(f"o.o_orderkey == {o_orderkey}")
             .groupBy("o.o_orderkey", "o.o_totalprice", "o.o_shippriority")
             .count()
    )
    if features_df.count() == 0:
        return {"score": 0.0, "drivers": ["unknown order"]}

    row = features_df.first()
    model = mlflow.pyfunc.load_model("models:/order_risk/Production")
    score = float(model.predict([[row.o_totalprice, row.o_shippriority, row["count"]]])[0])

    drivers = []
    if row.o_totalprice > 200000:    drivers.append("high total price")
    if row.o_shippriority == 0:      drivers.append("ship priority not escalated")
    if row["count"] > 5:              drivers.append("many line items")
    if not drivers:                  drivers.append("low risk profile")

    return {"score": score, "drivers": drivers}
$$
""")

print(f"Registered {ws_cat}.{ws_sch}.calc_order_risk")
