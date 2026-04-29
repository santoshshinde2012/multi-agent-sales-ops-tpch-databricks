# Sales Ops Agent — Multi-Agent System on Databricks

A **reference implementation** of a production-grade multi-agent system on Databricks, built on the
built-in `samples.tpch` dataset so anyone can run it end-to-end without preparing data.

This repository is the companion code for the article
**[Beyond Genie Code: Orchestrating Production Multi-Agent Systems on Databricks](../docs/article.md)**.
Every file here maps to a numbered step in that article.

> **Who is this for?** Anyone with a Databricks workspace who wants to understand — and run — a real
> multi-agent system. No prior agent-framework experience required. If you can run a notebook, you
> can run this.

---

## What you'll build

By the time you finish this README you will have:

- A **chat assistant** that answers Sales Operations questions like *"which high-priority open orders
  are at risk this week?"*
- Four cooperating specialists behind it: a **Genie agent** for SQL, a **Knowledge agent** for
  document search, a **Unity Catalog function** for risk scoring, and an **MCP tool** for ticket
  creation.
- A **supervisor** that routes user questions to the right specialists, written in
  [LangGraph](https://langchain-ai.github.io/langgraph/).
- An **evaluation suite** that gates deployment.
- A **Databricks App** deployment that respects each user's individual permissions.

```
              ┌──────────────────────────────────────────────┐
              │          Databricks App (Chat UI)            │
              └──────────────────────┬───────────────────────┘
                                     ▼
              ┌──────────────────────────────────────────────┐
              │  SalesOpsAgent  (MLflow ResponsesAgent)      │
              └──────────────────────┬───────────────────────┘
                                     ▼
              ┌──────────────────────────────────────────────┐
              │  Supervisor  (LangGraph: router → synth)     │
              └──┬─────────────┬───────────────┬─────────────┘
                 ▼             ▼               ▼
           ┌────────┐   ┌──────────┐    ┌──────────────┐
           │ Genie  │   │ Knowledge │    │  UC Function │
           │ Agent  │   │  Agent   │    │ order_risk   │
           └────────┘   └──────────┘    └──────────────┘
                                     ▲
              ┌──────────────────────┴───────────────────────┐
              │  samples.tpch  +  Vector Search + MLflow     │
              └──────────────────────────────────────────────┘
```

---

## Prerequisites

You need **one** Databricks workspace with the following features enabled. All four are standard on
modern workspaces:

- [Unity Catalog](https://docs.databricks.com/aws/en/data-governance/unity-catalog/)
- [AI/BI Genie](https://docs.databricks.com/aws/en/genie/)
- [Mosaic AI Vector Search](https://docs.databricks.com/aws/en/generative-ai/vector-search.html)
- [Model Serving](https://docs.databricks.com/aws/en/machine-learning/model-serving/) (for the LLM
  endpoint and the deployed agent)

You also need on your laptop:

- Python **3.10 or newer**
- The [Databricks CLI](https://docs.databricks.com/aws/en/dev-tools/cli/install.html) — `pip install
  databricks-cli` and run `databricks configure --token`

That's it. Everything else (the data, the LLM, the schema) lives in your workspace.

---

## Quick start (15 minutes)

### 1. Clone and install

```bash
git clone https://github.com/your-org/multi-agent-sales-ops-tpch-databricks.git
cd multi-agent-sales-ops-tpch-databricks
pip install -e ".[dev]"
```

### 2. Set up the data foundation

This creates the `sales_ops` schema, registers seven views over `samples.tpch`, declares primary and
foreign keys, and adds business-language column comments. **One command.**

```bash
databricks bundle deploy --target dev
databricks bundle run setup_data_foundation --target dev
```

What this does in plain English:

1. Creates a working schema called `sales_ops`.
2. Adds views like `sales_ops.orders` that point at `samples.tpch.orders`.
3. Tags every column with a comment so [Genie](https://docs.databricks.com/aws/en/genie/) knows what
   each column means.
4. Builds a [Vector Search index](https://docs.databricks.com/aws/en/generative-ai/vector-search.html)
   over the `o_comment` field of the orders table.

### 3. Create the Genie space

In your Databricks workspace UI:

1. Open **Genie** → **Create new space**.
2. Add the seven `sales_ops.*` views.
3. Copy-paste the synonyms and example questions from
   [`resources/genie_space_config.yaml`](resources/genie_space_config.yaml).
4. Note the **Genie space ID** from the URL (it looks like `01ef…`).

### 4. Configure the agent

Copy the example config and fill in the five values you collected above:

```bash
cp .env.example .env
# then edit .env to set:
#   DATABRICKS_HOST=...
#   WAREHOUSE_ID=...                         # SQL Warehouse ID (UC functions need this)
#   GENIE_SPACE_ID=...
#   VECTOR_SEARCH_ENDPOINT=customer_ops_endpoint
#   LLM_ENDPOINT=databricks-claude-sonnet-4-5
```

### 5. Run the agent locally

```bash
python -m sales_ops_agent.cli "Which high-priority open orders are at risk this week?"
```

You should see a streaming response that joins data from Genie, citations from Vector Search, and a
risk score from the UC function.

### 6. Deploy to Databricks Apps

```bash
databricks bundle deploy --target prod
```

The bundle handles model registration, on-behalf-of-user auth, resource grants, and app deployment.

---

## Project layout

The codebase mirrors the **nine steps** of the article. If you find a step in the article, the code
for it is in the matching file:

```
multi-agent-sales-ops-tpch-databricks/
├── README.md                        ← you are here
├── pyproject.toml                   ← project metadata + dependencies
├── databricks.yml                   ← Asset Bundle (Step 8 — deployment)
├── .env.example                     ← non-secret configuration template
│
├── sql/
│   └── 01_data_foundation.sql       ← Step 1 — sales_ops schema + comments + keys
│
├── resources/
│   ├── genie_space_config.yaml      ← Step 2 — Genie synonyms & example queries
│   └── eval_set.jsonl               ← Step 7 — release-gate eval cases
│
├── src/sales_ops_agent/
│   ├── config/
│   │   └── settings.py              ← typed configuration (Pydantic)
│   ├── specialists/
│   │   ├── base.py                  ← Specialist Protocol (Open/Closed principle)
│   │   ├── sql_executor.py          ← DatabricksSqlExecutor + Protocol
│   │   ├── genie_agent.py           ← Step 2 — structured-data specialist
│   │   ├── knowledge_agent.py       ← Step 3 — document specialist
│   │   ├── compute_function.py      ← Step 4 — UC function specialist
│   │   └── action_mcp.py            ← MCP action specialist (ticket creation)
│   ├── supervisor/
│   │   ├── router.py                ← intent classifier
│   │   ├── synthesizer.py           ← combines specialist outputs with citations
│   │   └── graph.py                 ← Step 5 — LangGraph wiring
│   ├── agent.py                     ← Step 6 — MLflow ResponsesAgent wrapper
│   ├── bootstrap.py                 ← composition root (where DI is concrete)
│   ├── evaluation/
│   │   └── runner.py                ← Step 7 — mlflow.evaluate harness
│   ├── monitoring/
│   │   └── enable.py                ← Step 9 — Agent Monitoring setup
│   └── cli.py                       ← interactive local runner
│
├── notebooks/
│   ├── 01_setup_data_foundation.py  ← runs sql/01_data_foundation.sql
│   ├── 02_build_vector_search.py    ← Step 3 — chunk + embed o_comment
│   ├── 03_train_order_risk_model.py ← Step 4 — train + register MLflow model
│   ├── 04_register_uc_functions.py  ← registers UC functions
│   ├── 05_run_evaluation.py         ← Step 7 — evaluation gate
│   ├── 06_enable_monitoring.py      ← Step 9 — Agent Monitoring
│   └── 07_smoke_test.py             ← end-to-end verification on your workspace
│
├── scripts/
│   └── seed_eval_set.py             ← generates a starter eval set
│
└── tests/
    ├── fakes.py                     ← shared test doubles
    ├── test_router.py
    ├── test_synthesizer.py
    ├── test_specialists.py          ← Protocol-conformance tests
    ├── test_sql_executor.py         ← SDK shape contract tests
    ├── test_integration.py          ← every specialist end-to-end with fakes
    └── test_agent.py                ← supervisor + ResponsesAgent wiring
```

---

## Design principles

This codebase deliberately follows the [SOLID principles](https://en.wikipedia.org/wiki/SOLID):

| Principle | How it shows up here |
|---|---|
| **S**ingle Responsibility | Each specialist module has one job. `genie_agent.py` only talks to Genie. `knowledge_agent.py` only talks to Vector Search. The supervisor only routes — it never reasons. |
| **O**pen / Closed | New specialists are added by implementing the `Specialist` Protocol — the supervisor doesn't change. Look at [`specialists/base.py`](src/sales_ops_agent/specialists/base.py). |
| **L**iskov Substitution | Every specialist is interchangeable behind the `Specialist` Protocol. The supervisor never knows whether it's calling Genie or Knowledge — it just calls `specialist.handle(state)`. |
| **I**nterface Segregation | Three small interfaces — `Specialist`, `Router`, `Synthesizer` — instead of one fat `Agent` interface. |
| **D**ependency Inversion | The `Supervisor` depends on the `Specialist` Protocol (an abstraction), not on concrete agent classes. Agents are injected at construction time, which makes testing trivial. |

The article also calls out three [anti-patterns](../docs/article.md#anti-patterns-to-avoid) — this
repo deliberately avoids all three:

- ❌ **The God Agent** — we use a supervisor + specialists from the start.
- ❌ **The LLM as a Calculator** — risk scoring lives in a UC function, not a prompt.
- ❌ **Service Principal for All Users** — the bundle declares
  [on-behalf-of-user auth](https://docs.databricks.com/aws/en/generative-ai/agent-framework/multi-agent-apps).

---

## Running the example end-to-end

Once setup is complete, here's the conversation Santosh, the Sales Ops Manager, has with the agent:

> **Santosh:** Which high-priority open orders are at risk this week?
>
> **Agent:** I found 12 URGENT-priority orders that are still open and were placed in the last 7
> days. Three stand out:
>
> | Order | Customer | Total | Risk |
> |---|---|---|---|
> | 11396166 | Customer #179329 | $211,452.30 | 0.87 (high) |
> | 12005471 | Customer #145002 | $189,003.10 | 0.82 (high) |
> | 13710944 | Customer #227285 | $168,520.00 | 0.74 (high) |
>
> Clerk comments on order 11396166 mention *"shipping delays from Indonesian supplier"* (source:
> `samples.tpch.orders.o_comment`). Top risk drivers across all three: high total price, supplier
> nation outside region, and ship priority not yet escalated.
>
> Shall I open follow-up tickets for these three?

Behind the scenes:

1. The supervisor classifies the question as `compose` (needs SQL + documents + compute).
2. The Genie agent runs SQL on `samples.tpch.orders + customer + lineitem`.
3. The Knowledge agent searches the `comment_index` for blocker mentions.
4. The UC function `calc_order_risk` returns a score and drivers per order.
5. The synthesizer combines everything with citations and streams the answer back.

Every step is traced in [MLflow](https://mlflow.org/), every action respects on-behalf-of-user
permissions, and every conversation feeds back into Agent Monitoring.

---

## Why does Genie sometimes give different answers to the same question?

This is a real and common production behavior, not a bug in your code. It's
worth understanding before you debug.

**The root cause:** Genie's NL→SQL step is performed by a foundation model.
Identical prompts can produce slightly different SQL on different calls —
different join order, slightly different filter, a different `LIMIT`, or just
different prose around the same numbers. This is amplified when the question
is ambiguous or the Genie space lacks curated examples for the question
pattern.

**This repo addresses the issue with three layered defenses:**

1. **Response caching** *(code-level)*. The `GenieSpecialist` accepts a
   `ResponseCache`. When configured (it is by default in production), the
   *exact same question* asked within `GENIE_CACHE_TTL_SECONDS` returns the
   *same cached answer*. Asking 3 times in a row gives 3 identical responses
   instead of 3 slightly different ones. Default TTL: 5 minutes.
   See [`response_cache.py`](src/sales_ops_agent/specialists/response_cache.py).

2. **Curated Genie space** *(setup-level)*. The synonyms and example
   queries in [`resources/genie_space_config.yaml`](resources/genie_space_config.yaml)
   pin Genie to specific SQL patterns for common questions. Genie's accuracy
   improves dramatically once a few canonical examples are in place — this
   is the article's [Step 2](docs/article.md) recommendation.

3. **Stateless invocation** *(architectural-level)*. Each call to Genie
   passes only the current user message — never prior conversation history.
   This prevents earlier questions from silently biasing later answers.

**Tuning the cache:** set `GENIE_CACHE_TTL_SECONDS` in `.env`. Set to `0` to
disable caching entirely (useful in environments where data changes minute
by minute). Set higher (say 1800) if your underlying tables update on a slow
cadence.

**Swapping the cache backend:** `ResponseCache` is a Protocol. The default
`InMemoryResponseCache` is process-local; for multi-replica deployments,
implement a Redis or Lakebase-backed cache that satisfies the same protocol
and inject it into `GenieSpecialist`. The specialist code does not change.

---

## Testing

```bash
pytest tests/
```

The suite has **42 tests** spanning:

- The **router**, **synthesizer**, and **supervisor wiring**.
- Every concrete **specialist** (Genie, Knowledge, Compute, Action) using fakes
  for the `SqlExecutor` / `GenieInvoker` / `TicketCreator` protocols.
- The **DatabricksSqlExecutor** itself, with fakes shaped exactly like the real
  Databricks SDK response objects.
- The **Genie response cache** — including the canonical "same question 3 times
  returns the same answer" guarantee.

There are no real Databricks calls in the test suite, so it runs in seconds
without a workspace. To verify your real workspace end-to-end, run the
**smoke test notebook** at `notebooks/07_smoke_test.py` after setup.

---

## Going further

- Read the [companion article](docs/article.md) for the full architectural rationale, the design
  patterns, and the anti-patterns to avoid.
- Adapt the specialists for your own domain. Start by editing
  [`specialists/genie_agent.py`](src/sales_ops_agent/specialists/genie_agent.py) — point it at your
  own Genie space and the rest follows.
- Add a new specialist by implementing the `Specialist` Protocol and registering it in
  [`supervisor/graph.py`](src/sales_ops_agent/supervisor/graph.py). The supervisor needs no changes.

---

## License

MIT — see [LICENSE](LICENSE).

## Further reading

- [Mosaic AI Agent Framework](https://docs.databricks.com/aws/en/generative-ai/agent-framework/)
- [Build a Multi-Agent System on Databricks Apps](https://docs.databricks.com/aws/en/generative-ai/agent-framework/multi-agent-apps)
- [Use Genie in Multi-Agent Systems](https://docs.databricks.com/aws/en/generative-ai/agent-framework/multi-agent-genie)
- [State of AI Agents 2026](https://www.databricks.com/resources/ebook/state-of-ai-agents-report)
