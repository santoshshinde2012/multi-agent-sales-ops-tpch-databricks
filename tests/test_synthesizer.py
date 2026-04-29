"""Tests for the CitedSynthesizer."""

from __future__ import annotations

from sales_ops_agent.specialists.base import (
    Citation,
    SpecialistResult,
    SupervisorState,
)
from sales_ops_agent.supervisor.synthesizer import CitedSynthesizer


def test_synthesizer_returns_friendly_message_when_no_results() -> None:
    state = SupervisorState(user_message="hi")

    out = CitedSynthesizer().synthesize(state)

    assert "wasn't able" in out


def test_synthesizer_renders_each_specialists_content_with_citations() -> None:
    state = SupervisorState(user_message="briefing")
    state.results.extend(
        [
            SpecialistResult(
                specialist="genie",
                content="3 at-risk URGENT orders",
                citations=[Citation(source="samples.tpch.orders", detail="rows=3")],
            ),
            SpecialistResult(
                specialist="compute",
                content="Risk scores: 0.87, 0.82, 0.74",
                citations=[Citation(source="sales_ops.calc_order_risk", detail="3 calls")],
            ),
        ]
    )

    out = CitedSynthesizer().synthesize(state)

    # Both specialist sections appear
    assert "Genie" in out and "Compute" in out
    # Citations block is rendered
    assert "samples.tpch.orders" in out
    assert "sales_ops.calc_order_risk" in out
