"""LangGraph wiring — the Supervisor as a state graph.

This is the implementation of **Step 5** in the article. The graph is intentionally
small:

    router → (genie | knowledge | compute | action) → synthesize → END

For ``compose`` intents, the router fans out to multiple specialists in sequence,
and the synthesizer combines their results.

This module follows two SOLID principles closely:

* **Dependency Inversion** — the supervisor holds Specialists by Protocol type,
  not by concrete class. Different deployments wire different specialists in.
* **Open/Closed** — to add a new specialist, you implement the Specialist Protocol
  and call ``with_specialist`` on the builder. The graph definition does not need
  to know about your concrete class.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from langgraph.graph import END, StateGraph

from sales_ops_agent.specialists import Specialist, SupervisorState

from .router import IntentLabel, IntentRouter
from .synthesizer import CitedSynthesizer, Synthesizer

logger = logging.getLogger(__name__)


class SupervisorBuilder:
    """Fluent builder that constructs a LangGraph ``StateGraph`` from injected parts.

    Why a builder? It separates *configuration* (which specialists, which router,
    which synthesizer) from *execution* (the compiled graph). Tests construct
    builders with fakes; production constructs builders with real components.
    """

    def __init__(self) -> None:
        self._specialists: dict[str, Specialist] = {}
        self._router: IntentRouter | None = None
        self._synthesizer: Synthesizer = CitedSynthesizer()

    # ─── Fluent API ─────────────────────────────────────────────────────────
    def with_specialist(self, specialist: Specialist) -> SupervisorBuilder:
        """Register a specialist by its ``name`` attribute."""
        self._specialists[specialist.name] = specialist
        return self

    def with_router(self, router: IntentRouter) -> SupervisorBuilder:
        self._router = router
        return self

    def with_synthesizer(self, synthesizer: Synthesizer) -> SupervisorBuilder:
        self._synthesizer = synthesizer
        return self

    # ─── Build ──────────────────────────────────────────────────────────────
    def build(self) -> Callable[[str], SupervisorState]:
        """Compile the graph and return a callable ``invoke(message)`` function."""
        if self._router is None:
            raise ValueError("Supervisor needs a router; call .with_router() first.")
        if not self._specialists:
            raise ValueError("Supervisor needs at least one specialist.")

        graph = StateGraph(SupervisorState)
        graph.add_node("route", self._route_node)
        for name in self._specialists:
            graph.add_node(name, self._make_specialist_node(name))
        graph.add_node("synthesize", self._synthesize_node)

        graph.set_entry_point("route")
        graph.add_conditional_edges(
            "route",
            lambda s: s.intent or IntentLabel.COMPOSE.value,
            self._edge_map(),
        )

        # Compose chains specialists in sequence so each can see prior results.
        # Other intents go from their single specialist straight to synthesize.
        if "knowledge" in self._specialists and "compute" in self._specialists:
            graph.add_conditional_edges(
                "genie",
                lambda s: "knowledge" if s.intent == IntentLabel.COMPOSE.value else "synthesize",
                {"knowledge": "knowledge", "synthesize": "synthesize"},
            )
            graph.add_conditional_edges(
                "knowledge",
                lambda s: "compute" if s.intent == IntentLabel.COMPOSE.value else "synthesize",
                {"compute": "compute", "synthesize": "synthesize"},
            )
            graph.add_edge("compute", "synthesize")
        else:
            for name in self._specialists:
                graph.add_edge(name, "synthesize")

        # Action and any other specialist go directly to synthesize.
        for name in self._specialists:
            if name not in {"genie", "knowledge", "compute"}:
                graph.add_edge(name, "synthesize")

        graph.add_edge("synthesize", END)

        compiled = graph.compile()

        def invoke(user_message: str) -> SupervisorState:
            initial = SupervisorState(user_message=user_message)
            final = compiled.invoke(initial)
            # LangGraph hands back a dict (the dataclass fields keyed by name).
            # Coerce it back to SupervisorState so the rest of the system can
            # rely on attribute access. This is the only place we tolerate that
            # representation gap.
            if isinstance(final, dict):
                return SupervisorState(**final)
            return final  # type: ignore[no-any-return]

        return invoke

    # ─── Node definitions (private) ─────────────────────────────────────────
    def _route_node(self, state: SupervisorState) -> SupervisorState:
        assert self._router is not None
        intent = self._router.classify(state.user_message)
        logger.info("Router classified message as %s", intent)
        state.intent = intent.value
        return state

    def _make_specialist_node(self, name: str) -> Callable[[SupervisorState], SupervisorState]:
        """Build the LangGraph node for a single specialist.

        Each specialist node always invokes its underlying specialist when the
        graph reaches it. The supervisor's job is to route correctly via the
        conditional edges in ``_edge_map``; once we land on a node, we run.

        For ``compose`` intent we still want genie → knowledge → compute to run
        in sequence so each subsequent specialist can read the prior results.
        We achieve that by routing ``compose`` to ``genie`` and then chaining
        knowledge and compute behind it via additional edges in :meth:`build`.
        """
        specialist = self._specialists[name]

        def _node(state: SupervisorState) -> SupervisorState:
            result = specialist.handle(state)
            state.results.append(result)
            return state

        return _node

    def _synthesize_node(self, state: SupervisorState) -> SupervisorState:
        state.final_response = self._synthesizer.synthesize(state)
        return state

    def _edge_map(self) -> dict[str, str]:
        """Conditional routing: intent label → next node name."""
        # COMPOSE always starts with genie so subsequent specialists can join on its rows.
        return {
            IntentLabel.STRUCTURED.value: "genie",
            IntentLabel.KNOWLEDGE.value: "knowledge",
            IntentLabel.COMPUTE.value: "compute",
            IntentLabel.COMPOSE.value: "genie",
            IntentLabel.ACTION.value: "action",
        }


def build_default_supervisor(
    *,
    router: IntentRouter,
    specialists: list[Specialist],
) -> Callable[[str], SupervisorState]:
    """Convenience factory used by ``SalesOpsAgent`` and the CLI."""
    builder = SupervisorBuilder().with_router(router)
    for s in specialists:
        builder = builder.with_specialist(s)
    return builder.build()
