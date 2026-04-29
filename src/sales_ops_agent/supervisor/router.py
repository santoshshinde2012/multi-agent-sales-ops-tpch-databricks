"""Router — classifies the user message into a closed set of intent labels.

The article (Step 5) makes a clear point: a small classifier with a closed label
set beats free-form "decide what to do" prompts. Keeping the label set tight is
what makes the supervisor's behavior predictable and easy to evaluate.

Single Responsibility: this module only classifies. It does not call specialists,
combine results, or stream output.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Protocol

logger = logging.getLogger(__name__)


class IntentLabel(StrEnum):
    """The closed set of intents the supervisor can route on.

    Adding a new label is a one-line change here plus a one-line edit to the
    LangGraph wiring. That's deliberately the only friction.
    """

    STRUCTURED = "structured"
    KNOWLEDGE = "knowledge"
    COMPUTE = "compute"
    COMPOSE = "compose"  # fan-out to multiple specialists
    ACTION = "action"


class IntentClassifier(Protocol):
    """Anything that can map a message string to an IntentLabel."""

    def classify(self, message: str) -> IntentLabel:  # pragma: no cover
        ...


class IntentRouter:
    """LLM-backed router that calls a small classifier prompt.

    Tests inject a fake IntentClassifier; production wires in a real LLM-backed one.
    """

    SYSTEM_PROMPT = (
        "You classify a Sales Ops question into one of: "
        "structured, knowledge, compute, compose, action. "
        "Reply with only the label, nothing else.\n"
        "- structured: pure SQL question (counts, totals, listings).\n"
        "- knowledge: search for blockers / notes in clerk comments.\n"
        "- compute: deterministic risk score for known orders.\n"
        "- compose: needs SQL plus comments plus risk (most morning briefings).\n"
        "- action: explicit request to open or create tickets."
    )

    def __init__(self, classifier: IntentClassifier) -> None:
        self._classifier = classifier

    def classify(self, message: str) -> IntentLabel:
        try:
            label = self._classifier.classify(message)
        except Exception:
            logger.exception("Router classification failed; defaulting to compose")
            return IntentLabel.COMPOSE

        # Defensive: if a model returns something off-vocabulary, fall back.
        try:
            return IntentLabel(label)
        except ValueError:
            logger.warning("Router returned unknown label %r — defaulting to compose", label)
            return IntentLabel.COMPOSE
