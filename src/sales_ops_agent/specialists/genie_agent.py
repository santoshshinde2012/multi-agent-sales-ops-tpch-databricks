"""Genie specialist — translates natural language to SQL over the sales_ops schema.

Implementation of **Step 2** in the article.

Single Responsibility: this module only knows how to ask Genie a question and
turn the response into a ``SpecialistResult``. It does not route, classify, or
synthesize.

Dependency Inversion: the underlying invoker is injected as a callable, so
tests substitute a fake without needing a real Genie space. An optional
``ResponseCache`` is also injected — production wires in an ``InMemoryResponseCache``
(or a Redis/Lakebase one) so that asking the same question 3 times in a row
returns the same answer instead of three slightly different ones.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .base import Citation, SpecialistResult, SupervisorState
from .response_cache import ResponseCache, normalize_question

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient

logger = logging.getLogger(__name__)


GenieInvoker = Callable[[str], dict[str, Any]]
"""A callable taking the user message and returning a Genie response dict."""


class GenieSpecialist:
    """Routes structured-data questions to a Genie space.

    Parameters
    ----------
    invoker : callable
        Calls Genie and returns the raw response dict.
    cache : ResponseCache, optional
        If provided, identical questions within the cache TTL return the
        cached answer instead of re-calling Genie. This keeps user-facing
        responses **consistent across repeated calls** — the canonical fix
        for "I asked the same thing 3 times and got 3 different answers".
    """

    name = "genie"

    def __init__(
        self,
        invoker: GenieInvoker,
        cache: ResponseCache | None = None,
    ) -> None:
        self._invoke = invoker
        self._cache = cache

    @classmethod
    def from_workspace(
        cls,
        workspace_client: WorkspaceClient,
        genie_space_id: str,
        cache: ResponseCache | None = None,
    ) -> GenieSpecialist:
        """Construct a GenieSpecialist that calls a real Genie space.

        See https://docs.databricks.com/aws/en/generative-ai/agent-framework/multi-agent-genie
        """
        from databricks_langchain import GenieAgent

        agent = GenieAgent(
            genie_space_id=genie_space_id,
            client=workspace_client,
            include_context=True,  # we want the SQL summary for citations
        )

        def invoke(message: str) -> dict[str, Any]:
            # GenieAgent.invoke takes a LangChain-style messages list and
            # returns a dict that includes 'messages' and (with include_context)
            # 'context'. Each call is **stateless** — we deliberately do not
            # carry conversation history between invocations, because that
            # would let prior questions silently bias the answer to the next.
            return agent.invoke({"messages": [{"role": "user", "content": message}]})

        return cls(invoker=invoke, cache=cache)

    def handle(self, state: SupervisorState) -> SpecialistResult:
        cache_key = normalize_question(state.user_message)

        # Step 1 — cache hit? Return immediately. Same question, same answer.
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.info("Genie cache hit for: %s", cache_key[:80])
                return _result_from_response(cached)

        # Step 2 — cache miss: actually call Genie.
        try:
            response = self._invoke(state.user_message)
        except Exception as exc:
            logger.exception("Genie call failed")
            return SpecialistResult(
                specialist=self.name,
                content=f"Could not retrieve structured data: {exc}",
            )

        # Step 3 — store the response so the next 2 (or 200) repeats are cheap
        # AND consistent.
        if self._cache is not None:
            self._cache.set(cache_key, response)

        return _result_from_response(response)


# ─── Pure response-to-result mapping (kept module-level for testability) ──────
def _result_from_response(response: dict[str, Any]) -> SpecialistResult:
    text = _extract_answer_text(response)
    rows = _extract_rows(response)
    sql_summary = _extract_sql_summary(response)

    return SpecialistResult(
        specialist=GenieSpecialist.name,
        content=text or "Genie returned no answer.",
        citations=[
            Citation(
                source="samples.tpch (via Genie)",
                detail=sql_summary or "generated SQL",
            )
        ],
        data={"rows": rows},
    )


def _extract_answer_text(response: dict[str, Any]) -> str:
    """Pull the assistant text out of a GenieAgent response."""
    messages = response.get("messages") or []
    for msg in reversed(messages):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "type", None)
        if role in ("assistant", "ai"):
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list) and content:
                first = content[0]
                return first.get("text", "") if isinstance(first, dict) else str(first)
    return ""


def _extract_rows(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract structured rows so downstream specialists can join on o_orderkey.

    GenieAgent surfaces rows in a few different shapes depending on version.
    We look in three places, returning the first non-empty list we find.
    """
    rows = response.get("rows")
    if isinstance(rows, list) and rows:
        return [r for r in rows if isinstance(r, dict)]

    ctx = response.get("context") or {}
    qr = ctx.get("query_result") or {}
    rows = qr.get("rows") or qr.get("data")
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        return rows

    return []


def _extract_sql_summary(response: dict[str, Any]) -> str:
    """Summarize the executed SQL into a short citation string."""
    ctx = response.get("context") or {}
    sql = ctx.get("sql") or ctx.get("query")
    if isinstance(sql, str) and sql.strip():
        compact = re.sub(r"\s+", " ", sql).strip()
        return compact[:120]
    return ""
