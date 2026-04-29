"""Tests for the DatabricksSqlExecutor.

These exercise the real SDK type contract: we construct fake SDK response
objects with the exact attribute names the real API uses, then verify our
executor parses them correctly. If the SDK shape changes, these tests fail
loudly — which is exactly what we want.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from sales_ops_agent.specialists.sql_executor import DatabricksSqlExecutor


# ─── SDK-shaped fakes ────────────────────────────────────────────────────────
@dataclass
class FakeColumn:
    name: str


@dataclass
class FakeSchema:
    columns: list[FakeColumn]


@dataclass
class FakeManifest:
    schema: FakeSchema | None


@dataclass
class FakeResult:
    data_array: list[list[Any]] | None


@dataclass
class FakeStatementResponse:
    manifest: FakeManifest | None
    result: FakeResult | None


class FakeStatementExecution:
    """Stands in for ws.statement_execution; records calls."""

    def __init__(self, response: FakeStatementResponse) -> None:
        self._response = response
        self.last_call: dict[str, Any] | None = None

    def execute_statement(
        self,
        *,
        statement: str,
        warehouse_id: str,
        parameters: list | None = None,
        wait_timeout: str | None = None,
    ) -> FakeStatementResponse:
        self.last_call = {
            "statement": statement,
            "warehouse_id": warehouse_id,
            "parameters": parameters,
            "wait_timeout": wait_timeout,
        }
        return self._response


class FakeWorkspaceClient:
    def __init__(self, response: FakeStatementResponse) -> None:
        self.statement_execution = FakeStatementExecution(response)


# ─── Tests ───────────────────────────────────────────────────────────────────
def test_executor_returns_rows_as_dicts() -> None:
    response = FakeStatementResponse(
        manifest=FakeManifest(
            schema=FakeSchema(columns=[FakeColumn(name="o_orderkey"), FakeColumn(name="score")])
        ),
        result=FakeResult(data_array=[[11396166, 0.87], [12005471, 0.71]]),
    )
    ws = FakeWorkspaceClient(response)
    executor = DatabricksSqlExecutor(workspace_client=ws, warehouse_id="abc123")  # type: ignore[arg-type]

    rows = executor.execute("SELECT * FROM t WHERE k = :k", {"k": 1})

    assert rows == [
        {"o_orderkey": 11396166, "score": 0.87},
        {"o_orderkey": 12005471, "score": 0.71},
    ]


def test_executor_passes_parameters_with_correct_shape() -> None:
    from databricks.sdk.service.sql import StatementParameterListItem

    response = FakeStatementResponse(
        manifest=FakeManifest(schema=FakeSchema(columns=[FakeColumn(name="x")])),
        result=FakeResult(data_array=[]),
    )
    ws = FakeWorkspaceClient(response)
    executor = DatabricksSqlExecutor(workspace_client=ws, warehouse_id="abc123")  # type: ignore[arg-type]

    executor.execute("SELECT :a, :b", {"a": "hello", "b": 42})

    call = ws.statement_execution.last_call
    assert call is not None
    assert call["warehouse_id"] == "abc123"
    assert call["wait_timeout"] == "30s"
    params = call["parameters"]
    assert params is not None and len(params) == 2
    assert all(isinstance(p, StatementParameterListItem) for p in params)
    assert {p.name: p.value for p in params} == {"a": "hello", "b": "42"}


def test_executor_returns_empty_list_for_empty_result() -> None:
    response = FakeStatementResponse(
        manifest=FakeManifest(schema=FakeSchema(columns=[FakeColumn(name="x")])),
        result=FakeResult(data_array=None),
    )
    ws = FakeWorkspaceClient(response)
    executor = DatabricksSqlExecutor(workspace_client=ws, warehouse_id="abc123")  # type: ignore[arg-type]

    assert executor.execute("SELECT 1") == []


def test_executor_returns_empty_list_when_manifest_missing() -> None:
    response = FakeStatementResponse(manifest=None, result=None)
    ws = FakeWorkspaceClient(response)
    executor = DatabricksSqlExecutor(workspace_client=ws, warehouse_id="abc123")  # type: ignore[arg-type]

    assert executor.execute("SELECT 1") == []


def test_executor_handles_no_parameters() -> None:
    response = FakeStatementResponse(
        manifest=FakeManifest(schema=FakeSchema(columns=[FakeColumn(name="x")])),
        result=FakeResult(data_array=[[1]]),
    )
    ws = FakeWorkspaceClient(response)
    executor = DatabricksSqlExecutor(workspace_client=ws, warehouse_id="abc123")  # type: ignore[arg-type]

    rows = executor.execute("SELECT 1")

    assert rows == [{"x": 1}]
    assert ws.statement_execution.last_call["parameters"] is None


@pytest.mark.parametrize("missing_field", ["manifest", "result"])
def test_executor_is_resilient_to_missing_fields(missing_field: str) -> None:
    response = FakeStatementResponse(
        manifest=None if missing_field == "manifest" else FakeManifest(
            schema=FakeSchema(columns=[FakeColumn(name="x")])
        ),
        result=None if missing_field == "result" else FakeResult(data_array=[[1]]),
    )
    ws = FakeWorkspaceClient(response)
    executor = DatabricksSqlExecutor(workspace_client=ws, warehouse_id="abc123")  # type: ignore[arg-type]

    assert executor.execute("SELECT 1") == []
