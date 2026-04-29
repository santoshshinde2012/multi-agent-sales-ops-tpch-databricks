# Architecture

This document is a short architectural reference for contributors. Read the
[full companion article](article.md) for the rationale; this file covers the
**code layout** and **how the pieces fit together**.

## High-level diagram

```
                      ┌──────────────────────────────────────┐
                      │        Databricks App (Chat UI)      │
                      └──────────────────┬───────────────────┘
                                         ▼
                      ┌──────────────────────────────────────┐
                      │  SalesOpsAgent (ResponsesAgent)      │  ← src/sales_ops_agent/agent.py
                      └──────────────────┬───────────────────┘
                                         ▼
                      ┌──────────────────────────────────────┐
                      │  Supervisor (LangGraph)              │  ← src/sales_ops_agent/supervisor/
                      │  ┌────────┐  ┌────────────┐          │
                      │  │ Router │→ │ Synthesizer│          │
                      │  └────────┘  └────────────┘          │
                      └─┬─────────┬───────────┬──────────────┘
                        ▼         ▼           ▼
                  ┌─────────┐ ┌────────┐ ┌──────────┐
                  │  Genie  │ │ Know.  │ │ Compute  │   ← src/sales_ops_agent/specialists/
                  └─────────┘ └────────┘ └──────────┘
                        │         │           │
                        ▼         ▼           ▼
                  samples.tpch  Vector       UC Function
                                Search       (MLflow model)
```

## SOLID principles in this codebase

### Single Responsibility
Every module has one job:
- `genie_agent.py` only talks to Genie.
- `knowledge_agent.py` only talks to Vector Search.
- `compute_function.py` only calls the UC function.
- `router.py` only classifies intent.
- `synthesizer.py` only produces final text.

### Open/Closed
Adding a new specialist (e.g. a `CalendarSpecialist`) means:
1. Implement the `Specialist` Protocol.
2. Pass it to `SupervisorBuilder.with_specialist(...)`.

The supervisor's wiring code in `graph.py` does not change.

### Liskov Substitution
The supervisor holds a list of `Specialist` Protocol instances. Production
wires real specialists; tests wire `FakeSpecialist` from `tests/fakes.py`.
The supervisor cannot tell the difference — that's the whole point.

### Interface Segregation
Three small protocols — `Specialist`, `IntentClassifier`, `Synthesizer` —
instead of one fat `Agent` interface. Each has exactly one method:
- `Specialist.handle(state)`
- `IntentClassifier.classify(message)`
- `Synthesizer.synthesize(state)`

### Dependency Inversion
The composition root (`bootstrap.py`) is the only file that knows how to
construct concrete dependencies. Every other module receives its
collaborators by Protocol type. Swap `databricks-claude-sonnet-4-5` for
another LLM by editing one line in `bootstrap.py`.

## Data flow

1. `SalesOpsAgent.predict` extracts the user message from the request.
2. The compiled supervisor runs `router → specialist(s) → synthesizer`.
3. The synthesizer reads `state.results` and produces the final cited text.
4. MLflow tracing wraps the whole thing.

## Where each article step lives

| Article step | Code |
|---|---|
| 1. Data foundation | `sql/01_data_foundation.sql`, `notebooks/01_setup_data_foundation.py` |
| 2. Genie space | `resources/genie_space_config.yaml`, `specialists/genie_agent.py` |
| 3. Knowledge agent | `notebooks/02_build_vector_search.py`, `specialists/knowledge_agent.py` |
| 4. UC function | `notebooks/03_train_order_risk_model.py`, `notebooks/04_register_uc_functions.py`, `specialists/compute_function.py` |
| 5. Supervisor | `supervisor/graph.py`, `supervisor/router.py`, `supervisor/synthesizer.py` |
| 6. ResponsesAgent | `agent.py` |
| 7. Evaluation | `evaluation/runner.py`, `notebooks/05_run_evaluation.py`, `resources/eval_set.jsonl` |
| 8. Deployment | `databricks.yml` |
| 9. Monitoring | `monitoring/enable.py`, `notebooks/06_enable_monitoring.py` |
