"""Compute specialist — deterministic order risk scoring.

Implementation of **Step 4** in the article. Risk scoring lives in a Unity
Catalog function backed by a versioned MLflow model — never inside an LLM
prompt — so the output is reproducible, auditable, and reviewable by the
business.

Avoids the "LLM as a Calculator" anti-pattern called out in the article.
"""

from __future__ import annotations

import logging

from .base import Citation, SpecialistResult, SupervisorState
from .sql_executor import SqlExecutor

logger = logging.getLogger(__name__)


class ComputeFunctionSpecialist:
    """Computes ``calc_order_risk(o_orderkey)`` for every order in scope."""

    name = "compute"

    def __init__(
        self,
        sql_executor: SqlExecutor,
        function_fqn: str = "main.sales_ops.calc_order_risk",
    ) -> None:
        self._sql = sql_executor
        self._function_fqn = function_fqn

    def handle(self, state: SupervisorState) -> SpecialistResult:
        order_keys = self._extract_order_keys(state)
        if not order_keys:
            return SpecialistResult(
                specialist=self.name,
                content="No orders to score — Genie returned no rows.",
            )

        scored: list[dict] = []
        for order_key in order_keys:
            try:
                row = self._score_one(order_key)
                if row is not None:
                    scored.append(row)
            except Exception:
                logger.exception("Risk scoring failed for order %s", order_key)

        if not scored:
            return SpecialistResult(
                specialist=self.name,
                content="Could not compute risk scores for the matching orders.",
            )

        bullet_lines = [
            f"- Order {row['o_orderkey']}: risk {row['score']:.2f} "
            f"(top driver: {row['top_driver']})"
            for row in scored
        ]
        return SpecialistResult(
            specialist=self.name,
            content="Risk scores:\n" + "\n".join(bullet_lines),
            citations=[
                Citation(
                    source=f"{self._function_fqn}",
                    detail=f"order_key {row['o_orderkey']}",
                )
                for row in scored
            ],
            data={"scored": scored},
        )

    @staticmethod
    def _extract_order_keys(state: SupervisorState) -> list[int]:
        for prior in state.results:
            if prior.specialist == "genie":
                rows = prior.data.get("rows", [])
                return [int(r["o_orderkey"]) for r in rows if "o_orderkey" in r]
        return []

    def _score_one(self, order_key: int) -> dict | None:
        """Invoke the UC function for a single order and return a normalized dict.

        UC scalar functions return a STRUCT, so the result dict has a single
        column. We unpack it defensively into a flat shape.
        """
        rows = self._sql.execute(
            statement=f"SELECT * FROM {self._function_fqn}(:order_key) AS r",
            parameters={"order_key": order_key},
        )
        if not rows:
            return None
        record = rows[0]
        # The function returns a struct named after the function or 'r' depending on
        # how the SQL engine names it; handle both.
        struct = next(iter(record.values()))
        if isinstance(struct, dict):
            score = float(struct.get("score", 0.0))
            drivers = list(struct.get("drivers", []) or [])
        else:
            # Fallback if the SDK already flattened the struct into separate columns.
            score = float(record.get("score", 0.0))
            drivers = list(record.get("drivers", []) or [])
        return {
            "o_orderkey": order_key,
            "score": score,
            "drivers": drivers,
            "top_driver": drivers[0] if drivers else "n/a",
        }
