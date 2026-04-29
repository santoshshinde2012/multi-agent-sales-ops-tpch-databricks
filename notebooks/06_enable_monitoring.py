# Databricks notebook source
# MAGIC %md
# MAGIC # Step 9 — Enable Continuous Monitoring
# MAGIC
# MAGIC Once the agent is deployed, this notebook turns on Agent Monitoring with
# MAGIC sensible defaults: groundedness, tool-call accuracy, latency, and a 10%
# MAGIC sample rate. Run it once after deployment.

# COMMAND ----------
dbutils.widgets.text("endpoint_name", "sales-ops-agent")

# COMMAND ----------
from sales_ops_agent.monitoring import enable_agent_monitoring

enable_agent_monitoring(
    endpoint_name=dbutils.widgets.get("endpoint_name"),
    sample_rate=0.10,
    alert_on_regression=True,
)

print("Agent Monitoring enabled. Check the Monitoring tab in your endpoint UI.")
