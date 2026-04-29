"""Action specialist — proposes follow-up tickets via an external MCP server.

This specialist is intentionally cautious: it never opens tickets on the
first user message. It returns a *proposal* the supervisor presents to the
user. Only after explicit confirmation does ``confirm`` actually call the
ticket-creation tool.

This shape avoids two failure modes:

* Side effects on ambiguous queries (the agent shouldn't open ten tickets
  just because the user asked an information question).
* Unscoped privilege escalation — the action runs under the on-behalf-of-user
  auth declared in databricks.yml.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from .base import SpecialistResult, SupervisorState

logger = logging.getLogger(__name__)


TicketCreator = Callable[[dict[str, Any]], dict[str, Any]]
"""A callable that takes a ticket payload and returns the created ticket dict.

Production wires this to an MCP-backed function (Jira, Linear, ServiceNow, …).
Tests inject a stub. Either way the specialist depends only on this callable.
"""


class ActionMCPSpecialist:
    """Proposes follow-up ticket creation; commits only on explicit confirm."""

    name = "action"

    def __init__(self, ticket_creator: TicketCreator | None = None) -> None:
        self._create = ticket_creator

    def handle(self, state: SupervisorState) -> SpecialistResult:
        """Return a confirmation prompt — never a side effect."""
        scored = self._extract_scored(state)
        if not scored:
            return SpecialistResult(
                specialist=self.name,
                content="No risky orders identified — nothing to escalate.",
            )

        summary = ", ".join(str(s["o_orderkey"]) for s in scored[:3])
        return SpecialistResult(
            specialist=self.name,
            content=(
                f"I can open follow-up tickets for the top {len(scored)} risky orders "
                f"(starting with {summary}). Reply 'confirm' to proceed."
            ),
            data={"pending_tickets": scored},
        )

    def confirm(self, scored: list[dict]) -> SpecialistResult:
        """Actually create tickets. Invoked only after the user confirms."""
        if self._create is None:
            return SpecialistResult(
                specialist=self.name,
                content="Ticket creation is not configured in this environment.",
            )

        opened: list[dict] = []
        for entry in scored:
            try:
                ticket = self._create(_payload(entry))
                opened.append({"o_orderkey": entry["o_orderkey"], **ticket})
            except Exception:
                logger.exception("Failed to open ticket for %s", entry["o_orderkey"])

        if not opened:
            return SpecialistResult(
                specialist=self.name,
                content="No tickets were opened (ticket service may be unavailable).",
            )

        bullet_lines = [
            f"- Order {t['o_orderkey']} → ticket {t.get('ticket_id', '?')}" for t in opened
        ]
        return SpecialistResult(
            specialist=self.name,
            content=f"Opened {len(opened)} follow-up tickets:\n" + "\n".join(bullet_lines),
            data={"opened": opened},
        )

    @staticmethod
    def _extract_scored(state: SupervisorState) -> list[dict]:
        for prior in state.results:
            if prior.specialist == "compute":
                return prior.data.get("scored", [])
        return []


def _payload(entry: dict[str, Any]) -> dict[str, Any]:
    """Render a ticket payload from a scored order entry."""
    drivers = entry.get("drivers") or [entry.get("top_driver", "n/a")]
    return {
        "title": f"Risk follow-up for order {entry['o_orderkey']}",
        "body": (
            f"Risk score {entry.get('score', 0.0):.2f}. "
            f"Top drivers: {', '.join(drivers)}. "
            "Review supplier and ship priority."
        ),
        "labels": ["sales-ops", "risk-followup"],
    }
