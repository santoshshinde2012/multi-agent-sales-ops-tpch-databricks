"""Agent Monitoring — Step 9.

Wraps the Databricks Agent Monitoring API with sensible defaults. Every metric
listed here corresponds to a real production failure mode the team has seen at
least once.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

DEFAULT_METRICS = ("groundedness", "tool_call_accuracy", "latency_ms")
DEFAULT_SAMPLE_RATE = 0.10


def enable_agent_monitoring(
    endpoint_name: str,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    alert_on_regression: bool = True,
) -> None:
    """Turn on monitoring for a deployed agent endpoint.

    Args:
        endpoint_name: The Model Serving / Apps endpoint name.
        metrics: Metrics to track. Defaults are recommended for any agent.
        sample_rate: Fraction of production traffic to sample for evaluation.
        alert_on_regression: If True, fire alerts when a metric drops.
    """
    # NB: imported lazily so unit tests can run without the databricks-agents extra.
    from databricks.agents import enable_monitoring  # type: ignore[import-not-found]

    logger.info(
        "Enabling Agent Monitoring on %s with metrics=%s, sample_rate=%.2f",
        endpoint_name,
        metrics,
        sample_rate,
    )
    enable_monitoring(
        endpoint_name=endpoint_name,
        metrics=list(metrics),
        sample_rate=sample_rate,
        alert_on_regression=alert_on_regression,
    )
