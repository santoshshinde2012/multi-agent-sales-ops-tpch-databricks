"""Shared test doubles.

These fakes implement the same Protocols as the real components, which is
exactly what Liskov Substitution and Dependency Inversion enable: the system
under test never knows the difference.
"""

from __future__ import annotations

from sales_ops_agent.specialists.base import (
    Citation,
    SpecialistResult,
    SupervisorState,
)
from sales_ops_agent.supervisor.router import IntentLabel


class FakeSpecialist:
    """A specialist that returns a canned result.

    Construct with a name and a content string; optionally pass `data` so
    downstream specialists can read what a real Genie call would have produced.
    """

    def __init__(
        self,
        name: str,
        content: str = "ok",
        data: dict | None = None,
        citations: list[Citation] | None = None,
    ) -> None:
        self.name = name
        self._content = content
        self._data = data or {}
        self._citations = citations or []
        self.calls: list[SupervisorState] = []

    def handle(self, state: SupervisorState) -> SpecialistResult:
        self.calls.append(state)
        return SpecialistResult(
            specialist=self.name,
            content=self._content,
            citations=self._citations,
            data=self._data,
        )


class FakeIntentClassifier:
    """An IntentClassifier that always returns the configured label."""

    def __init__(self, label: IntentLabel = IntentLabel.STRUCTURED) -> None:
        self.label = label
        self.calls: list[str] = []

    def classify(self, message: str) -> IntentLabel:
        self.calls.append(message)
        return self.label
