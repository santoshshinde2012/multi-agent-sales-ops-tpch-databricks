"""End-to-end tests for the SupervisorBuilder and SalesOpsAgent.

These prove the SOLID design pays off: we wire fake specialists and a fake
classifier into the real supervisor and the real ResponsesAgent wrapper, then
exercise the whole flow in-process.
"""

from __future__ import annotations

from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

from sales_ops_agent.agent import SalesOpsAgent
from sales_ops_agent.specialists.base import Citation
from sales_ops_agent.supervisor import IntentRouter, SupervisorBuilder
from sales_ops_agent.supervisor.router import IntentLabel

from .fakes import FakeIntentClassifier, FakeSpecialist


def _output_text(response: ResponsesAgentResponse) -> str:
    """Pull the assistant text out of a ResponsesAgentResponse."""
    assert response.output, "expected at least one output item"
    item = response.output[0]
    # output items are dicts shaped like {"role":"assistant","content":[{"text": "..."}]}
    content = item["content"] if isinstance(item, dict) else item.content  # type: ignore[index]
    return content[0]["text"] if isinstance(content[0], dict) else content[0].text


def _build_agent(intent: IntentLabel) -> SalesOpsAgent:
    """Construct an agent with three fake specialists."""
    genie = FakeSpecialist(
        name="genie",
        content="3 URGENT open orders",
        data={"rows": [{"o_orderkey": 11396166}, {"o_orderkey": 12005471}]},
        citations=[Citation(source="samples.tpch.orders", detail="rows=3")],
    )
    knowledge = FakeSpecialist(
        name="knowledge",
        content="Clerks mention shipping delays.",
        citations=[Citation(source="comment_index:11396166", detail="score 0.83")],
    )
    compute = FakeSpecialist(
        name="compute",
        content="Risk scores: 0.87, 0.82",
        data={"scored": [{"o_orderkey": 11396166, "score": 0.87, "top_driver": "high price"}]},
    )
    action = FakeSpecialist(name="action", content="(would open tickets)")

    router = IntentRouter(classifier=FakeIntentClassifier(label=intent))
    invoke = (
        SupervisorBuilder()
        .with_router(router)
        .with_specialist(genie)
        .with_specialist(knowledge)
        .with_specialist(compute)
        .with_specialist(action)
        .build()
    )
    return SalesOpsAgent(supervisor=invoke)


def test_compose_intent_runs_three_specialists_and_synthesizes() -> None:
    agent = _build_agent(IntentLabel.COMPOSE)
    request = ResponsesAgentRequest(
        input=[{"role": "user", "content": "Briefing on at-risk URGENT orders"}]
    )

    text = _output_text(agent.predict(request))

    assert "URGENT" in text or "shipping delays" in text
    assert "samples.tpch.orders" in text  # citation present


def test_structured_intent_runs_only_genie() -> None:
    agent = _build_agent(IntentLabel.STRUCTURED)
    request = ResponsesAgentRequest(
        input=[{"role": "user", "content": "How many URGENT orders this week?"}]
    )

    text = _output_text(agent.predict(request))

    assert "3 URGENT open orders" in text


def test_action_intent_routes_to_action_only() -> None:
    agent = _build_agent(IntentLabel.ACTION)
    request = ResponsesAgentRequest(
        input=[{"role": "user", "content": "Open tickets for the top 3"}]
    )

    text = _output_text(agent.predict(request))

    assert "would open tickets" in text
