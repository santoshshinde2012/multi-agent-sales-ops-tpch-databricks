# Databricks notebook source
# MAGIC %md
# MAGIC # Step 4 — Train and Register the `order_risk` MLflow Model
# MAGIC
# MAGIC Trains a tiny logistic-regression model that predicts the probability an
# MAGIC open URGENT order will become "stuck" (proxied here by ship priority,
# MAGIC supplier nation, line-item count, and total price). Logs to MLflow and
# MAGIC promotes to the `Production` stage.
# MAGIC
# MAGIC The model is tiny on purpose — the article's point is that risk scoring
# MAGIC must be **deterministic and auditable**, not state-of-the-art ML.

# COMMAND ----------
import mlflow
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F

mlflow.autolog(disable=True)  # we'll log explicitly

# COMMAND ----------
features = (spark.table("samples.tpch.orders")
            .alias("o")
            .join(spark.table("samples.tpch.lineitem").alias("l"),
                  F.col("o.o_orderkey") == F.col("l.l_orderkey"))
            .groupBy("o.o_orderkey", "o.o_totalprice", "o.o_shippriority")
            .agg(F.count("*").alias("lineitem_count"))
            .withColumn("label",
                        F.when((F.col("o_totalprice") > 200000) &
                               (F.col("o_shippriority") == 0), 1).otherwise(0))
            .selectExpr("o_orderkey", "o_totalprice", "o_shippriority",
                        "lineitem_count", "label"))

# COMMAND ----------
assembler = VectorAssembler(
    inputCols=["o_totalprice", "o_shippriority", "lineitem_count"],
    outputCol="features",
)
train = assembler.transform(features)

# COMMAND ----------
with mlflow.start_run(run_name="order_risk_v1") as run:
    lr = LogisticRegression(featuresCol="features", labelCol="label")
    model = lr.fit(train)
    mlflow.spark.log_model(model, "model", registered_model_name="order_risk")
    mlflow.log_metric("training_rows", train.count())
    print(f"Logged model in run {run.info.run_id}")

# COMMAND ----------
client = mlflow.MlflowClient()
latest = client.get_latest_versions("order_risk", stages=["None"])[-1]
client.transition_model_version_stage("order_risk", latest.version, "Production")
print(f"Promoted order_risk v{latest.version} to Production")
