"""Tests for the IntentRouter."""

from __future__ import annotations

from sales_ops_agent.supervisor.router import IntentLabel, IntentRouter

from .fakes import FakeIntentClassifier


def test_router_returns_classifier_label() -> None:
    classifier = FakeIntentClassifier(label=IntentLabel.COMPOSE)
    router = IntentRouter(classifier=classifier)

    result = router.classify("Briefing on at-risk orders please")

    assert result is IntentLabel.COMPOSE
    assert classifier.calls == ["Briefing on at-risk orders please"]


def test_router_falls_back_when_classifier_raises() -> None:
    class BoomClassifier:
        def classify(self, message: str) -> IntentLabel:
            raise RuntimeError("LLM is down")

    router = IntentRouter(classifier=BoomClassifier())

    assert router.classify("anything") is IntentLabel.COMPOSE
