"""Command-line interface — run the agent locally without deploying.

Useful during development; not used in production (Databricks Apps invokes the
``SalesOpsAgent`` directly through Model Serving).

Example::

    python -m sales_ops_agent.cli "Which URGENT open orders are at risk this week?"
"""

from __future__ import annotations

import logging
import sys

from mlflow.types.responses import ResponsesAgentRequest

from .bootstrap import build_production_agent

logger = logging.getLogger(__name__)


def main() -> int:
    """Entry point invoked by ``python -m sales_ops_agent.cli`` or the console_script."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python -m sales_ops_agent.cli '<your question>'", file=sys.stderr)
        return 2

    question = " ".join(sys.argv[1:])
    agent = build_production_agent()
    request = ResponsesAgentRequest(
        input=[{"role": "user", "content": question}]
    )
    response = agent.predict(request)
    print(response.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
