"""Centralized, typed configuration.

This module exists so that no other part of the codebase reads ``os.environ`` directly.
Every value is validated at startup, which means a missing or malformed setting fails
fast with a clear error rather than crashing deep inside an agent call.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or a .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ─── Workspace ─────────────────────────────────────────────────────────
    DATABRICKS_HOST: str = Field(..., description="Workspace URL")
    DATABRICKS_TOKEN: str | None = Field(default=None, description="Optional PAT for local runs")

    # ─── SQL Warehouse ─────────────────────────────────────────────────────
    # Required to execute the UC functions via the Statement Execution API.
    # Find this in the SQL Warehouse UI: it's the value in the connection details.
    WAREHOUSE_ID: str = Field(..., description="SQL Warehouse ID for UC function calls")

    # ─── Genie space ───────────────────────────────────────────────────────
    GENIE_SPACE_ID: str = Field(..., description="Curated Genie space for sales_ops schema")

    # ─── Vector Search ─────────────────────────────────────────────────────
    VECTOR_SEARCH_ENDPOINT: str = Field(default="customer_ops_endpoint")
    VECTOR_SEARCH_INDEX: str = Field(default="comment_index_vs")

    # ─── LLM ───────────────────────────────────────────────────────────────
    LLM_ENDPOINT: str = Field(default="databricks-claude-sonnet-4-5")

    # ─── Working schema ────────────────────────────────────────────────────
    WORKING_CATALOG: str = Field(default="main")
    WORKING_SCHEMA: str = Field(default="sales_ops")

    # ─── Observability ─────────────────────────────────────────────────────
    MLFLOW_EXPERIMENT: str = Field(default="/Shared/sales_ops_agent")

    # ─── Genie response cache ──────────────────────────────────────────────
    # How long (in seconds) to return a cached Genie answer for the same
    # question. 0 disables the cache entirely. The default of 5 minutes
    # absorbs "user asks the same thing 3 times in a row" bursts while
    # keeping data reasonably fresh for slower-changing tables.
    GENIE_CACHE_TTL_SECONDS: float = Field(default=300.0)

    # ─── Derived ───────────────────────────────────────────────────────────
    @property
    def working_schema_fqn(self) -> str:
        """Fully qualified name for the working schema (catalog.schema)."""
        return f"{self.WORKING_CATALOG}.{self.WORKING_SCHEMA}"

    @property
    def search_function_fqn(self) -> str:
        return f"{self.working_schema_fqn}.search_order_comments"

    @property
    def risk_function_fqn(self) -> str:
        return f"{self.working_schema_fqn}.calc_order_risk"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance.

    The lru_cache ensures Settings is constructed once per process, which means
    we hit the .env file and validate types only on the first call.
    """
    return Settings()  # type: ignore[call-arg]
