# Databricks notebook source
# MAGIC %md
# MAGIC # End-to-End Smoke Test
# MAGIC
# MAGIC Run this notebook **after** you've completed the setup steps:
# MAGIC
# MAGIC 1. ✅ Notebook 01 — data foundation
# MAGIC 2. ✅ Notebook 02 — Vector Search index
# MAGIC 3. ✅ Notebook 03 — order_risk MLflow model
# MAGIC 4. ✅ Notebook 04 — UC functions registered
# MAGIC 5. ✅ Genie space created with synonyms from `resources/genie_space_config.yaml`
# MAGIC
# MAGIC This notebook constructs the full agent and asks the canonical question
# MAGIC from the article. If it returns a cited answer, your environment is wired
# MAGIC correctly and you're ready to deploy via `databricks bundle deploy`.

# COMMAND ----------
# MAGIC %pip install -e ../.
# MAGIC %restart_python

# COMMAND ----------
import os
# Configure the agent for this workspace. In a real deployment, on-behalf-of-user
# auth replaces the env vars below — the bundle declares the scopes.
os.environ.setdefault("DATABRICKS_HOST", spark.conf.get("spark.databricks.workspaceUrl") or "")
dbutils.widgets.text("warehouse_id",   "")
dbutils.widgets.text("genie_space_id", "")
os.environ["WAREHOUSE_ID"]     = dbutils.widgets.get("warehouse_id")
os.environ["GENIE_SPACE_ID"]   = dbutils.widgets.get("genie_space_id")

# COMMAND ----------
from sales_ops_agent.bootstrap import build_production_agent
from mlflow.types.responses import ResponsesAgentRequest

agent = build_production_agent()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Ask the canonical question

# COMMAND ----------
question = "Which high-priority open orders are at risk this week and what came up in clerk notes?"
request  = ResponsesAgentRequest(input=[{"role": "user", "content": question}])
response = agent.predict(request)

# Pull the assistant text out of the structured response.
output_item = response.output[0]
assistant_text = (
    output_item["content"][0]["text"]
    if isinstance(output_item, dict)
    else output_item.content[0].text
)

print(assistant_text)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Expected
# MAGIC
# MAGIC The response should include:
# MAGIC * A list of URGENT-priority open orders from the last week (Genie)
# MAGIC * Clerk-comment excerpts for those orders (Knowledge)
# MAGIC * Risk scores for each order (Compute)
# MAGIC * A `Sources` block with citations
# MAGIC
# MAGIC If you see "Could not retrieve…" messages instead, double-check:
# MAGIC * `WAREHOUSE_ID` is correct and you have permission to use it
# MAGIC * `GENIE_SPACE_ID` matches the space you curated
# MAGIC * Notebooks 1–4 ran successfully in this catalog/schema
