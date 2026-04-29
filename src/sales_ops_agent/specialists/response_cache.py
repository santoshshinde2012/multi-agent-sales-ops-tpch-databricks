"""Response cache abstraction.

Solves a real production issue: foundation-model-backed agents like Genie can
return slightly different answers to the same question across calls. Caching
identical questions for a short TTL window restores user-facing consistency
without sacrificing freshness.

The ``ResponseCache`` protocol exists so we can swap implementations:

* ``InMemoryResponseCache`` — single-process default, fine for tests and small
  deployments.
* For production at scale, plug in a Redis-backed or Lakebase-backed
  implementation that satisfies the same protocol; ``GenieSpecialist`` does
  not need to change.

Why a TTL? Two reasons:

1. Underlying data does change (TPC-H is static, your real tables won't be).
   A 5-minute window is long enough to absorb a "user asks 3 times in a row"
   burst but short enough that fresh data flows through.
2. The cache key is deliberately simple: the normalized question string. We
   do not key on the supervisor state, because identical questions should
   yield identical Genie answers regardless of who asked.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol


class ResponseCache(Protocol):
    """A small key-value store with TTL semantics."""

    def get(self, key: str) -> Any | None:
        ...

    def set(self, key: str, value: Any) -> None:
        ...


@dataclass
class _Entry:
    value: Any
    expires_at: float


class InMemoryResponseCache:
    """Thread-safe, process-local TTL cache.

    Suitable for a single replica. For multi-replica deployments, swap for a
    Redis or Lakebase implementation that satisfies the ``ResponseCache``
    protocol.
    """

    def __init__(self, ttl_seconds: float = 300.0, max_entries: int = 256) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")
        if max_entries <= 0:
            raise ValueError("max_entries must be > 0")
        self._ttl = ttl_seconds
        self._max = max_entries
        self._lock = threading.Lock()
        self._store: dict[str, _Entry] = {}

    def get(self, key: str) -> Any | None:
        now = time.monotonic()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if entry.expires_at <= now:
                # Lazy eviction — drop expired entry on next read.
                self._store.pop(key, None)
                return None
            return entry.value

    def set(self, key: str, value: Any) -> None:
        now = time.monotonic()
        with self._lock:
            # Simple bounded cache: evict the oldest entry if at capacity.
            if len(self._store) >= self._max and key not in self._store:
                oldest = min(self._store, key=lambda k: self._store[k].expires_at)
                self._store.pop(oldest, None)
            self._store[key] = _Entry(value=value, expires_at=now + self._ttl)


def normalize_question(question: str) -> str:
    """Normalize a user question into a stable cache key.

    Folds whitespace and case so that "  How many URGENT orders?  " and
    "how many urgent orders?" hit the same cache entry.
    """
    return " ".join(question.split()).lower()
