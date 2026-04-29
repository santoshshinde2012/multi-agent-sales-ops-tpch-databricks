"""Tests for the Specialist abstraction.

These tests do not exercise real Databricks calls. They prove that the system
holds together at the Protocol level — exactly the value of using a small
interface instead of a fat one.
"""

from __future__ import annotations

from sales_ops_agent.specialists.base import (
    Specialist,
    SpecialistResult,
    SupervisorState,
)

from .fakes import FakeSpecialist


def test_fake_specialist_satisfies_protocol() -> None:
    fake: Specialist = FakeSpecialist(name="genie")  # passes type-check
    state = SupervisorState(user_message="any")

    result = fake.handle(state)

    assert isinstance(result, SpecialistResult)
    assert result.specialist == "genie"


def test_specialist_is_pure_with_respect_to_state() -> None:
    """A specialist must not mutate the SupervisorState it receives."""
    fake = FakeSpecialist(name="genie", content="three rows", data={"rows": [{"o_orderkey": 1}]})
    state = SupervisorState(user_message="briefing")

    fake.handle(state)

    # The contract: state is untouched — only the supervisor mutates state.
    assert state.results == []
    assert state.intent is None
