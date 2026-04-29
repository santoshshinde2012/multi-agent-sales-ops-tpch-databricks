"""Specialist abstractions that the supervisor depends on.

This module exists to satisfy two SOLID principles at once:

* **Open/Closed** — the supervisor is closed for modification but open for extension.
  Adding a new specialist (e.g. a calendar specialist, a CRM specialist) means
  implementing the ``Specialist`` protocol. The supervisor's code does not change.

* **Dependency Inversion** — the supervisor depends on this abstraction, not on the
  concrete Genie / Knowledge / UC / MCP classes. Concrete specialists are passed in
  at construction time, which makes unit tests trivial: tests substitute fakes that
  implement the same protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Citation:
    """A single piece of provenance for a specialist's claim."""

    source: str  # e.g. "samples.tpch.orders" or "comment_index:11396166"
    detail: str  # short human-readable note, e.g. "row count 12" or "chunk score 0.83"


@dataclass
class SpecialistResult:
    """The structured output of a single specialist call.

    Specialists return this rather than free-form text so the supervisor can
    synthesize a grounded answer with citations attached to specific claims.
    """

    specialist: str  # which specialist produced this result
    content: str  # the human-readable answer fragment
    citations: list[Citation] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)  # structured payload, optional


@dataclass
class SupervisorState:
    """Shared state passed between supervisor nodes.

    Each specialist appends a ``SpecialistResult`` to ``results``. The synthesizer
    reads ``results`` and produces the final cited response. ``user_message`` is
    immutable for the lifetime of one supervisor invocation.
    """

    user_message: str
    intent: str | None = None
    results: list[SpecialistResult] = field(default_factory=list)
    final_response: str | None = None


class Specialist(Protocol):
    """The contract every specialist agent must satisfy.

    The protocol is intentionally tiny — name plus a single ``handle`` method —
    so any concrete specialist can drop in without touching supervisor code.
    """

    name: str

    def handle(self, state: SupervisorState) -> SpecialistResult:
        """Process the user message and return a structured result.

        Implementations MUST be pure with respect to ``state``: they read it but
        never mutate it. The supervisor is the only component that mutates state,
        which keeps the data flow easy to reason about.
        """
        ...
