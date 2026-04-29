"""Composition root — wires concrete dependencies into the supervisor.

This is the only place in the codebase that knows how to construct *all* the
real specialists, the real router, the real SQL executor, and the real
ResponsesAgent. Everywhere else depends on abstractions, which is the
**Dependency Inversion** principle in action.

If you want to swap one component (a different LLM, a different ticket
service, a different Genie space) this is the only file you change.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks

from .agent import SalesOpsAgent
from .config import get_settings
from .specialists import (
    ActionMCPSpecialist,
    ComputeFunctionSpecialist,
    DatabricksSqlExecutor,
    GenieSpecialist,
    KnowledgeSpecialist,
    SupervisorState,
)
from .supervisor import IntentRouter, build_default_supervisor
from .supervisor.router import IntentClassifier, IntentLabel

logger = logging.getLogger(__name__)


# ─── A real, LLM-backed intent classifier ─────────────────────────────────────
class LLMIntentClassifier(IntentClassifier):
    """Calls the configured Databricks LLM endpoint with a small classification prompt."""

    _ALLOWED = {label.value for label in IntentLabel}

    def __init__(self, llm_endpoint: str) -> None:
        self._llm = ChatDatabricks(endpoint=llm_endpoint, temperature=0.0)

    def classify(self, message: str) -> IntentLabel:
        response = self._llm.invoke(
            [
                {"role": "system", "content": IntentRouter.SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ]
        )
        text = ((response.content or "").strip().lower()
                 if isinstance(response.content, str) else "")
        # Be defensive: small LLMs sometimes wrap the label in punctuation.
        for label in self._ALLOWED:
            if label in text:
                return IntentLabel(label)
        # Unknown label is handled gracefully by IntentRouter (falls back to compose).
        return IntentLabel.COMPOSE


# ─── Composition root ────────────────────────────────────────────────────────
def build_production_agent() -> SalesOpsAgent:
    """Construct a fully wired SalesOpsAgent ready for serving."""
    settings = get_settings()
    ws = WorkspaceClient(host=settings.DATABRICKS_HOST, token=settings.DATABRICKS_TOKEN)

    sql_executor = DatabricksSqlExecutor(
        workspace_client=ws,
        warehouse_id=settings.WAREHOUSE_ID,
    )

    genie = GenieSpecialist.from_workspace(
        workspace_client=ws,
        genie_space_id=settings.GENIE_SPACE_ID,
    )
    knowledge = KnowledgeSpecialist(
        sql_executor=sql_executor,
        function_fqn=settings.search_function_fqn,
    )
    compute = ComputeFunctionSpecialist(
        sql_executor=sql_executor,
        function_fqn=settings.risk_function_fqn,
    )
    action = ActionMCPSpecialist()  # ticket_creator wired by the deployment if used

    classifier = LLMIntentClassifier(llm_endpoint=settings.LLM_ENDPOINT)
    router = IntentRouter(classifier=classifier)

    supervisor: Callable[[str], SupervisorState] = build_default_supervisor(
        router=router,
        specialists=[genie, knowledge, compute, action],
    )

    logger.info("SalesOpsAgent constructed successfully.")
    return SalesOpsAgent(supervisor=supervisor)
