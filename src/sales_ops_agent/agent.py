"""SalesOpsAgent — the MLflow ``ResponsesAgent`` wrapper around the Supervisor.

This is the implementation of **Step 6** in the article. Wrapping the supervisor
in MLflow's ``ResponsesAgent`` interface gives us four things for free:

* Streaming output via ``predict_stream``
* Standardized tool-call message history
* AI Playground and Agent Evaluation compatibility
* Built-in tracing — every invocation produces an MLflow trace

Tests build a SalesOpsAgent with fake specialists and call ``predict`` directly.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Generator
from typing import Any

from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from sales_ops_agent.specialists import SupervisorState

logger = logging.getLogger(__name__)


SupervisorInvocable = Callable[[str], SupervisorState]


class SalesOpsAgent(ResponsesAgent):
    """The single class served by Model Serving and Databricks Apps.

    The constructor takes a *Supervisor invocable* — a callable that takes a
    user message and returns a SupervisorState. Production wires this to the
    LangGraph-compiled supervisor; tests substitute a stub.
    """

    def __init__(self, supervisor: SupervisorInvocable) -> None:
        super().__init__()
        self._supervisor = supervisor

    # ─── Sync ───────────────────────────────────────────────────────────────
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        message = self._extract_user_message(request)
        logger.info("predict received message: %s", message[:120])
        state = self._supervisor(message)
        text = state.final_response or ""
        # ResponsesAgent provides a helper that returns a properly-typed OutputItem
        # of type "message" containing the text. We wrap it in the required list.
        item = self.create_text_output_item(text=text, id=str(uuid.uuid4()))
        return ResponsesAgentResponse(output=[item])

    # ─── Streaming ──────────────────────────────────────────────────────────
    def predict_stream(
        self,
        request: ResponsesAgentRequest,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        message = self._extract_user_message(request)
        logger.info("predict_stream received message: %s", message[:120])
        state = self._supervisor(message)
        text = state.final_response or ""
        # The default supervisor returns the full response in one shot; emit it
        # as a single delta. A future improvement is to stream per-specialist.
        item_id = str(uuid.uuid4())
        delta = self.create_text_delta(delta=text, item_id=item_id)
        yield ResponsesAgentStreamEvent(
            type="response.output_text.delta",
            **delta,
        )
        # Closing event so clients know the response is complete.
        completed_item = self.create_text_output_item(text=text, id=item_id)
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=completed_item,
        )

    # ─── Helpers ────────────────────────────────────────────────────────────
    @staticmethod
    def _extract_user_message(request: ResponsesAgentRequest) -> str:
        """Pull the latest user-role message from the input list.

        Accepts both Pydantic-typed message items and plain dicts (the latter
        is what the test harness passes in).
        """
        for item in reversed(request.input or []):
            role = _get(item, "role")
            if role != "user":
                continue
            content: Any = _get(item, "content")
            if isinstance(content, str):
                return content
            if isinstance(content, list) and content:
                first = content[0]
                text = _get(first, "text")
                if text is not None:
                    return str(text)
                return str(first)
        raise ValueError("No user message found in request input")


def _get(obj: Any, key: str) -> Any:
    """Read a field from a dict OR a Pydantic model uniformly."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
