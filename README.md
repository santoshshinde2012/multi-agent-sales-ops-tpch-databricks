# Sales Ops Agent — Multi-Agent System on Databricks

A **reference implementation** of a production-grade multi-agent system on Databricks, built on the
built-in `samples.tpch` dataset so anyone can run it end-to-end without preparing data.

This repository is the companion code for the article
**[Beyond Genie Code: Orchestrating Production Multi-Agent Systems on Databricks](https://medium.com/data-science-collective/beyond-genie-code-orchestrating-production-multi-agent-systems-on-databricks-86ac51e9c55b)**.
Every file here maps to a numbered step in that article.

> **Who is this for?** Anyone with a Databricks workspace who wants to understand — and run — a real
> multi-agent system. No prior agent-framework experience required. If you can run a notebook, you
> can run this.

---

## Quick start (15 minutes)

### 1. Clone and install

```bash
git clone https://github.com/santoshshinde2012/multi-agent-sales-ops-tpch-databricks.git
cd multi-agent-sales-ops-tpch-databricks
pip install -e ".[dev]"
```

The rest of this guide is unchanged from main. This commit only replaces the placeholder `your-org` clone URL with this repository.
