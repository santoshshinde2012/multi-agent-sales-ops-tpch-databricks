"""Tiny SQL helper that wraps the Databricks Statement Execution API correctly.

Why a dedicated helper?

* The real Databricks SDK has a non-trivial result-reading shape (manifest +
  result.data_array, not a simple list of rows). Encoding that knowledge in one
  place keeps every specialist focused on its own job (Single Responsibility).
* Both the knowledge and compute specialists need it; without this helper they'd
  duplicate the same boilerplate.
* It exposes a small, easy-to-fake interface so unit tests can substitute a stub.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient

logger = logging.getLogger(__name__)


class SqlExecutor(Protocol):
    """Anything that can execute a parameterized SQL statement and return rows of dicts.

    Tests provide a fake matching this protocol; production wires in
    ``DatabricksSqlExecutor``. The supervisor and specialists never depend on
    the concrete class.
    """

    def execute(
        self,
        statement: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        ...


class DatabricksSqlExecutor:
    """Real implementation backed by the Databricks Statement Execution API."""

    def __init__(self, workspace_client: WorkspaceClient, warehouse_id: str) -> None:
        self._ws = workspace_client
        self._warehouse_id = warehouse_id

    def execute(
        self,
        statement: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a SQL statement on the configured warehouse and return rows-of-dicts."""
        from databricks.sdk.service.sql import StatementParameterListItem

        param_list: list[StatementParameterListItem] | None = None
        if parameters:
            param_list = [
                StatementParameterListItem(name=k, value=str(v) if v is not None else None)
                for k, v in parameters.items()
            ]

        logger.debug("execute_statement: %s (params=%s)", statement[:200], parameters)
        response = self._ws.statement_execution.execute_statement(
            statement=statement,
            warehouse_id=self._warehouse_id,
            parameters=param_list,
            wait_timeout="30s",   # synchronous return for queries that finish in <30s
        )
        return _rows_to_dicts(response)


def _rows_to_dicts(response: Any) -> list[dict[str, Any]]:
    """Convert a StatementResponse into a list of column-keyed dicts.

    Handles both the success case and the empty-result case. Defensive against
    None on every nested attribute because the SDK uses optional fields.
    """
    manifest = getattr(response, "manifest", None)
    result = getattr(response, "result", None)
    if manifest is None or result is None:
        return []

    schema = getattr(manifest, "schema", None)
    columns = getattr(schema, "columns", None) if schema is not None else None
    if not columns:
        return []
    column_names = [c.name for c in columns]

    data_array = getattr(result, "data_array", None) or []
    return [dict(zip(column_names, row, strict=False)) for row in data_array]
