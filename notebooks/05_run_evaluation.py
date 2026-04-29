# Databricks notebook source
# MAGIC %md
# MAGIC # Step 7 — Evaluation as a Release Gate
# MAGIC
# MAGIC Loads the JSONL eval set, runs `mlflow.evaluate`, and compares each
# MAGIC metric against a per-metric threshold. The cell at the bottom either
# MAGIC promotes the model to Production or fails the notebook (which fails any
# MAGIC enclosing job, blocking deployment).

# COMMAND ----------
dbutils.widgets.text("model_uri", "models:/sales_ops_agent/Staging")
dbutils.widgets.text("eval_set_path", "../resources/eval_set.jsonl")

# COMMAND ----------
from pathlib import Path
from sales_ops_agent.evaluation import EvaluationGate

gate = EvaluationGate(model_uri=dbutils.widgets.get("model_uri"))
result = gate.run(Path(dbutils.widgets.get("eval_set_path")))

print("Metrics :", result.metrics)
print("Passed  :", result.passed)
if not result.passed:
    print("Failures:", result.failures)
    raise RuntimeError("Evaluation gate FAILED — deployment blocked.")
