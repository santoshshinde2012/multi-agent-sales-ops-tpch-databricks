"""Synthesizer — combines specialist results into one grounded response.

The synthesizer is the only component that produces user-facing text from
multiple specialist outputs. Keeping this responsibility separate means tests
can verify citation accuracy without spinning up any specialist.
"""

from __future__ import annotations

from typing import Protocol

from sales_ops_agent.specialists import SpecialistResult, SupervisorState


class Synthesizer(Protocol):
    """Anything that can turn a list of specialist results into one response."""

    def synthesize(self, state: SupervisorState) -> str:  # pragma: no cover
        ...


class CitedSynthesizer:
    """Plain-prose synthesizer that always attaches citations.

    The implementation is intentionally simple — it concatenates each
    specialist's content with a citation block at the end. Plug in an
    LLM-backed synthesizer later by implementing the same Protocol.
    """

    def synthesize(self, state: SupervisorState) -> str:
        if not state.results:
            return "I wasn't able to gather any information for that question."

        sections = [self._render_result(r) for r in state.results]
        body = "\n\n".join(sections)
        citations_block = self._render_citations(state.results)
        if citations_block:
            return f"{body}\n\n---\n**Sources**\n{citations_block}"
        return body

    @staticmethod
    def _render_result(result: SpecialistResult) -> str:
        """Each specialist's content gets a small heading for clarity."""
        heading = result.specialist.replace("_", " ").title()
        return f"**{heading}**\n{result.content.strip()}"

    @staticmethod
    def _render_citations(results: list[SpecialistResult]) -> str:
        rendered = []
        for r in results:
            for c in r.citations:
                rendered.append(f"- `{c.source}` — {c.detail}")
        return "\n".join(rendered)
