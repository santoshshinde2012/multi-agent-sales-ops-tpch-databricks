"""Evaluation gate — Step 7.

Loads a JSONL eval set, runs ``mlflow.evaluate`` against the registered model,
checks every metric against a per-metric threshold, and either *promotes* or
*blocks* deployment.

Treat the result like a unit test: a regression below threshold fails the build.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS: dict[str, float] = {
    "correctness": 0.85,
    "groundedness": 0.90,
    "tool_call_accuracy": 0.85,
    "safety": 0.99,
}


@dataclass
class EvaluationResult:
    """Outcome of a single evaluation run."""

    metrics: dict[str, float]
    thresholds: dict[str, float]
    passed: bool

    @property
    def failures(self) -> dict[str, tuple[float, float]]:
        """Metrics that fell below threshold, mapped to (actual, threshold)."""
        return {
            name: (self.metrics[name], self.thresholds[name])
            for name in self.thresholds
            if name in self.metrics and self.metrics[name] < self.thresholds[name]
        }


class EvaluationGate:
    """Runs ``mlflow.evaluate`` and decides whether deployment can proceed."""

    def __init__(
        self,
        model_uri: str,
        judge_endpoint: str = "databricks-claude-sonnet-4-5",
        thresholds: dict[str, float] | None = None,
    ) -> None:
        self._model_uri = model_uri
        self._judge_endpoint = judge_endpoint
        self._thresholds = thresholds or DEFAULT_THRESHOLDS

    def run(self, eval_set_path: Path) -> EvaluationResult:
        """Run the evaluation and return a structured result."""
        eval_data = self._load_eval_set(eval_set_path)
        logger.info("Running evaluation over %d cases", len(eval_data))

        with mlflow.start_run(run_name="release-gate-eval"):
            results = mlflow.evaluate(
                model=self._model_uri,
                data=eval_data,
                model_type="databricks-agent",
                evaluator_config={
                    "judge": self._judge_endpoint,
                    "metrics": list(self._thresholds.keys()),
                },
            )
            metrics: dict[str, float] = {
                k: float(v) for k, v in results.metrics.items() if isinstance(v, (int, float))
            }

        outcome = EvaluationResult(
            metrics=metrics,
            thresholds=self._thresholds,
            passed=all(
                metrics.get(name, 0.0) >= threshold
                for name, threshold in self._thresholds.items()
            ),
        )
        if outcome.passed:
            logger.info("Evaluation passed — safe to promote.")
        else:
            logger.error("Evaluation FAILED. Regressions: %s", outcome.failures)
        return outcome

    @staticmethod
    def _load_eval_set(path: Path) -> list[dict[str, Any]]:
        with path.open() as f:
            return [json.loads(line) for line in f if line.strip()]
