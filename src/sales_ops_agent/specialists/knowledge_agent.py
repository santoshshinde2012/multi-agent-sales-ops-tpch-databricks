"""Knowledge specialist — semantic search over the ``o_comment`` field.

Implementation of **Step 3** in the article. The retriever is wrapped in a
Unity Catalog function (``sales_ops.search_order_comments``) so it inherits
the same ``EXECUTE`` permission model as everything else in the lakehouse.

Single Responsibility: this module only does document retrieval. It does not
score risk, write SQL by hand, or synthesize answers.

Dependency Inversion: it depends on the ``SqlExecutor`` protocol, never on
the Databricks SDK directly. The concrete executor is injected at construction
time, which makes unit tests trivial.
"""

from __future__ import annotations

import logging

from .base import Citation, SpecialistResult, SupervisorState
from .sql_executor import SqlExecutor

logger = logging.getLogger(__name__)


class KnowledgeSpecialist:
    """Retrieves clerk-comment excerpts relevant to the user's question."""

    name = "knowledge"

    def __init__(
        self,
        sql_executor: SqlExecutor,
        function_fqn: str = "main.sales_ops.search_order_comments",
        top_k: int = 5,
    ) -> None:
        self._sql = sql_executor
        self._function_fqn = function_fqn
        self._top_k = top_k

    def handle(self, state: SupervisorState) -> SpecialistResult:
        """Run a hybrid vector search and return cited excerpts."""
        order_keys = self._extract_order_keys(state)

        try:
            chunks = self._sql.execute(
                statement=(
                    f"SELECT * FROM {self._function_fqn}(:query, "
                    f"{_array_literal(order_keys)}, :top_k)"
                ),
                parameters={
                    "query": state.user_message,
                    "top_k": self._top_k,
                },
            )
        except Exception as exc:
            logger.exception("Knowledge retrieval failed")
            return SpecialistResult(
                specialist=self.name,
                content=f"Could not retrieve clerk comments: {exc}",
            )

        if not chunks:
            return SpecialistResult(
                specialist=self.name,
                content="No relevant clerk comments found for the matching orders.",
            )

        bullet_lines = [
            f"- Order {c['o_orderkey']}: \"{str(c.get('content', ''))[:140].strip()}\""
            for c in chunks
        ]
        return SpecialistResult(
            specialist=self.name,
            content="Recent clerk comments worth noting:\n" + "\n".join(bullet_lines),
            citations=[
                Citation(
                    source=f"comment_index:{c['o_orderkey']}",
                    detail=f"score {float(c.get('score', 0.0)):.2f}",
                )
                for c in chunks
            ],
            data={"chunks": chunks},
        )

    @staticmethod
    def _extract_order_keys(state: SupervisorState) -> list[int]:
        """Pull o_orderkey values from any prior Genie result in the state."""
        for prior in state.results:
            if prior.specialist == "genie":
                rows = prior.data.get("rows", [])
                return [int(r["o_orderkey"]) for r in rows if "o_orderkey" in r]
        return []


def _array_literal(order_keys: list[int]) -> str:
    """Render a Python list of ints as a SQL ARRAY literal.

    Inlined into the statement (rather than passed via :param) because the
    Statement Execution API's named-parameter binding does not support array
    types in every SDK version. Casting through int() prevents SQL injection.

    Examples
    --------
    >>> _array_literal([])
    'array()'
    >>> _array_literal([1, 2, 3])
    'array(1, 2, 3)'
    """
    if not order_keys:
        return "array()"
    return "array(" + ", ".join(str(int(k)) for k in order_keys) + ")"
