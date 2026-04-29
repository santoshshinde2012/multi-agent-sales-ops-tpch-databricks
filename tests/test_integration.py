"""Integration tests for every concrete specialist.

These tests exercise the **real** specialist code paths — including SQL
statement construction, response parsing, and error handling — using fakes
that match the protocols. No Databricks workspace is required.

This is the test layer that proves the production code paths work end-to-end.
"""

from __future__ import annotations

from typing import Any

from sales_ops_agent.specialists import (
    ActionMCPSpecialist,
    ComputeFunctionSpecialist,
    GenieSpecialist,
    KnowledgeSpecialist,
    SpecialistResult,
    SupervisorState,
)


# ────────────────────────────────────────────────────────────────────────────
# Fake SqlExecutor
# ────────────────────────────────────────────────────────────────────────────
class FakeSqlExecutor:
    """Captures statements for assertion and returns canned rows."""

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self._rows = rows or []
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def execute(
        self,
        statement: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self.calls.append((statement, parameters))
        return self._rows


# ────────────────────────────────────────────────────────────────────────────
# Genie specialist
# ────────────────────────────────────────────────────────────────────────────
def test_genie_specialist_parses_text_rows_and_sql_summary() -> None:
    fake_response = {
        "messages": [
            {"role": "user", "content": "How many URGENT open orders this week?"},
            {"role": "assistant", "content": "There are 3 URGENT open orders."},
        ],
        "context": {
            "sql": "SELECT COUNT(*) FROM main.sales_ops.orders\nWHERE o_orderpriority = '1-URGENT'",
        },
        "rows": [{"o_orderkey": 11396166}, {"o_orderkey": 12005471}],
    }
    specialist = GenieSpecialist(invoker=lambda _msg: fake_response)
    state = SupervisorState(user_message="How many URGENT open orders this week?")

    result = specialist.handle(state)

    assert result.specialist == "genie"
    assert "3 URGENT open orders" in result.content
    assert result.data["rows"] == fake_response["rows"]
    assert result.citations[0].source == "samples.tpch (via Genie)"
    assert "SELECT COUNT(*)" in result.citations[0].detail


def test_genie_specialist_returns_friendly_message_on_invoker_error() -> None:
    def boom(_msg: str) -> dict[str, Any]:
        raise RuntimeError("network down")

    specialist = GenieSpecialist(invoker=boom)
    state = SupervisorState(user_message="anything")

    result = specialist.handle(state)

    assert "Could not retrieve structured data" in result.content
    assert "network down" in result.content


def test_genie_specialist_handles_empty_response() -> None:
    specialist = GenieSpecialist(invoker=lambda _msg: {"messages": []})
    state = SupervisorState(user_message="anything")

    result = specialist.handle(state)

    assert result.specialist == "genie"
    assert "no answer" in result.content.lower()


# ────────────────────────────────────────────────────────────────────────────
# Knowledge specialist
# ────────────────────────────────────────────────────────────────────────────
def _state_with_genie_rows(*order_keys: int) -> SupervisorState:
    state = SupervisorState(user_message="any blockers in recent comments?")
    state.results.append(
        SpecialistResult(
            specialist="genie",
            content="(genie ran)",
            data={"rows": [{"o_orderkey": k} for k in order_keys]},
        )
    )
    return state


def test_knowledge_specialist_calls_uc_function_with_array_literal_and_top_k() -> None:
    sql = FakeSqlExecutor(
        rows=[
            {"o_orderkey": 11396166, "content": "shipping delays", "score": 0.83},
            {"o_orderkey": 12005471, "content": "supplier issue", "score": 0.71},
        ]
    )
    specialist = KnowledgeSpecialist(
        sql_executor=sql,
        function_fqn="main.sales_ops.search_order_comments",
        top_k=5,
    )

    result = specialist.handle(_state_with_genie_rows(11396166, 12005471))

    statement, params = sql.calls[0]
    assert "search_order_comments(:query, array(11396166, 12005471), :top_k)" in statement
    assert params == {"query": "any blockers in recent comments?", "top_k": 5}
    assert "shipping delays" in result.content
    assert any(c.source == "comment_index:11396166" for c in result.citations)


def test_knowledge_specialist_returns_friendly_message_when_no_chunks_found() -> None:
    sql = FakeSqlExecutor(rows=[])
    specialist = KnowledgeSpecialist(sql_executor=sql)

    result = specialist.handle(_state_with_genie_rows(99))

    assert "No relevant clerk comments found" in result.content


def test_knowledge_specialist_uses_empty_array_when_no_genie_rows() -> None:
    sql = FakeSqlExecutor(rows=[])
    specialist = KnowledgeSpecialist(sql_executor=sql)
    state = SupervisorState(user_message="general blocker search")

    specialist.handle(state)

    statement, _ = sql.calls[0]
    assert "array()" in statement


def test_knowledge_specialist_recovers_from_sql_failure() -> None:
    class BoomSql:
        calls: list = []

        def execute(self, statement: str, parameters: dict | None = None) -> list[dict]:
            raise RuntimeError("warehouse unavailable")

    specialist = KnowledgeSpecialist(sql_executor=BoomSql())

    result = specialist.handle(_state_with_genie_rows(1))

    assert "Could not retrieve clerk comments" in result.content
    assert "warehouse unavailable" in result.content


# ────────────────────────────────────────────────────────────────────────────
# Compute specialist
# ────────────────────────────────────────────────────────────────────────────
def test_compute_specialist_scores_each_order_from_genie_results() -> None:
    sql = FakeSqlExecutor(
        rows=[{"r": {"score": 0.87, "drivers": ["high price", "ship priority not escalated"]}}]
    )
    specialist = ComputeFunctionSpecialist(
        sql_executor=sql,
        function_fqn="main.sales_ops.calc_order_risk",
    )

    result = specialist.handle(_state_with_genie_rows(11396166, 12005471))

    # Two genie rows → two calls to calc_order_risk
    assert len(sql.calls) == 2
    assert "calc_order_risk(:order_key)" in sql.calls[0][0]
    assert sql.calls[0][1] == {"order_key": 11396166}
    assert sql.calls[1][1] == {"order_key": 12005471}
    # Output contains both risk lines
    assert "Order 11396166: risk 0.87" in result.content
    assert "high price" in result.content
    assert result.data["scored"][0]["score"] == 0.87


def test_compute_specialist_handles_flattened_struct_response() -> None:
    """Some warehouses unpack STRUCT into separate columns."""
    sql = FakeSqlExecutor(rows=[{"score": 0.42, "drivers": ["low risk profile"]}])
    specialist = ComputeFunctionSpecialist(sql_executor=sql)
    state = _state_with_genie_rows(1)

    result = specialist.handle(state)

    assert result.data["scored"][0]["score"] == 0.42
    assert result.data["scored"][0]["top_driver"] == "low risk profile"


def test_compute_specialist_says_so_when_no_orders_to_score() -> None:
    sql = FakeSqlExecutor(rows=[])
    specialist = ComputeFunctionSpecialist(sql_executor=sql)
    state = SupervisorState(user_message="score risks")

    result = specialist.handle(state)

    assert "No orders to score" in result.content
    assert sql.calls == []


# ────────────────────────────────────────────────────────────────────────────
# Action specialist
# ────────────────────────────────────────────────────────────────────────────
def _state_with_compute_results() -> SupervisorState:
    state = SupervisorState(user_message="open tickets for the top 3")
    state.results.append(
        SpecialistResult(
            specialist="compute",
            content="(compute ran)",
            data={
                "scored": [
                    {"o_orderkey": 1, "score": 0.91, "drivers": ["high price"], "top_driver": "high price"},
                    {"o_orderkey": 2, "score": 0.85, "drivers": ["supplier"], "top_driver": "supplier"},
                ]
            },
        )
    )
    return state


def test_action_handle_proposes_but_never_creates_tickets() -> None:
    created: list[dict] = []

    def fake_creator(payload: dict) -> dict:
        created.append(payload)
        return {"ticket_id": "TKT-1"}

    specialist = ActionMCPSpecialist(ticket_creator=fake_creator)

    result = specialist.handle(_state_with_compute_results())

    assert "Reply 'confirm' to proceed" in result.content
    assert created == []  # no side effect on handle()


def test_action_confirm_creates_tickets_and_summarizes() -> None:
    created: list[dict] = []

    def fake_creator(payload: dict) -> dict:
        created.append(payload)
        return {"ticket_id": f"TKT-{len(created)}"}

    specialist = ActionMCPSpecialist(ticket_creator=fake_creator)
    scored = _state_with_compute_results().results[0].data["scored"]

    result = specialist.confirm(scored)

    assert len(created) == 2
    assert "Risk follow-up for order 1" in created[0]["title"]
    assert "Opened 2 follow-up tickets" in result.content
    assert "TKT-1" in result.content


def test_action_confirm_without_creator_returns_friendly_message() -> None:
    specialist = ActionMCPSpecialist(ticket_creator=None)

    result = specialist.confirm([{"o_orderkey": 1, "score": 0.9, "top_driver": "x"}])

    assert "not configured" in result.content


def test_action_handle_says_nothing_to_escalate_when_no_compute_results() -> None:
    specialist = ActionMCPSpecialist()
    state = SupervisorState(user_message="open tickets")

    result = specialist.handle(state)

    assert "nothing to escalate" in result.content.lower()
