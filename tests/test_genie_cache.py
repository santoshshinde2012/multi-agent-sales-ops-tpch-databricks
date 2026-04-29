"""Tests for the Genie response cache and consistency guarantees.

These directly cover the production issue: "same question 3 times → 3
different answers". After the fix, the same question within the cache TTL
returns the same response, and the underlying invoker is only called once.
"""

from __future__ import annotations

import time

import pytest

from sales_ops_agent.specialists import (
    GenieSpecialist,
    InMemoryResponseCache,
    SupervisorState,
)
from sales_ops_agent.specialists.response_cache import normalize_question


# ─── Cache primitive tests ────────────────────────────────────────────────────
def test_in_memory_cache_returns_value_within_ttl() -> None:
    cache = InMemoryResponseCache(ttl_seconds=60.0)
    cache.set("k", {"answer": "42"})

    assert cache.get("k") == {"answer": "42"}


def test_in_memory_cache_expires_after_ttl() -> None:
    cache = InMemoryResponseCache(ttl_seconds=0.05)  # 50ms
    cache.set("k", "v")

    time.sleep(0.1)

    assert cache.get("k") is None


def test_in_memory_cache_evicts_oldest_when_at_capacity() -> None:
    cache = InMemoryResponseCache(ttl_seconds=60.0, max_entries=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)  # should evict the oldest entry ("a")

    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("How many URGENT orders?", "how many urgent orders?"),
        ("  How   many  URGENT   orders?  ", "how many urgent orders?"),
        ("HOW MANY URGENT ORDERS?", "how many urgent orders?"),
    ],
)
def test_normalize_question_folds_whitespace_and_case(raw: str, expected: str) -> None:
    assert normalize_question(raw) == expected


def test_in_memory_cache_rejects_nonsense_config() -> None:
    with pytest.raises(ValueError):
        InMemoryResponseCache(ttl_seconds=0)
    with pytest.raises(ValueError):
        InMemoryResponseCache(ttl_seconds=60.0, max_entries=0)


# ─── Genie specialist consistency tests ──────────────────────────────────────
class _CountingInvoker:
    """A fake Genie invoker that returns slightly different rows each call.

    This is exactly what the user reported: same question, different result.
    With the cache enabled, only the first call should reach this invoker.
    """

    def __init__(self) -> None:
        self.call_count = 0

    def __call__(self, message: str) -> dict:
        self.call_count += 1
        # Different rows per call — the bug we are fixing.
        return {
            "messages": [
                {"role": "user", "content": message},
                {"role": "assistant", "content": f"answer #{self.call_count}"},
            ],
            "rows": [{"o_orderkey": self.call_count}],
            "context": {"sql": f"SELECT {self.call_count}"},
        }


def test_genie_returns_consistent_answer_across_repeated_calls_when_cached() -> None:
    invoker = _CountingInvoker()
    cache = InMemoryResponseCache(ttl_seconds=60.0)
    specialist = GenieSpecialist(invoker=invoker, cache=cache)

    state = SupervisorState(user_message="Which URGENT orders are at risk this week?")
    first  = specialist.handle(state)
    second = specialist.handle(state)
    third  = specialist.handle(state)

    # Genie was called exactly once — the next two reads came from cache.
    assert invoker.call_count == 1
    # All three responses are identical at the user-facing level.
    assert first.content == second.content == third.content
    assert first.data == second.data == third.data
    # And the answer is the *first* one — the user sees the same answer
    # they saw the first time, not a new one.
    assert "answer #1" in first.content


def test_genie_cache_hit_is_case_and_whitespace_insensitive() -> None:
    invoker = _CountingInvoker()
    cache = InMemoryResponseCache(ttl_seconds=60.0)
    specialist = GenieSpecialist(invoker=invoker, cache=cache)

    specialist.handle(SupervisorState(user_message="How many URGENT orders?"))
    specialist.handle(SupervisorState(user_message="  how   many   urgent orders?  "))
    specialist.handle(SupervisorState(user_message="HOW MANY URGENT ORDERS?"))

    assert invoker.call_count == 1


def test_genie_calls_underlying_invoker_again_after_ttl_expires() -> None:
    invoker = _CountingInvoker()
    cache = InMemoryResponseCache(ttl_seconds=0.05)
    specialist = GenieSpecialist(invoker=invoker, cache=cache)

    specialist.handle(SupervisorState(user_message="Q"))
    time.sleep(0.1)  # TTL expires
    specialist.handle(SupervisorState(user_message="Q"))

    assert invoker.call_count == 2


def test_genie_without_cache_calls_invoker_every_time() -> None:
    """Sanity check: removing the cache restores the original (broken) behavior.

    This documents the contrast: cache=None means each call re-invokes Genie.
    """
    invoker = _CountingInvoker()
    specialist = GenieSpecialist(invoker=invoker, cache=None)

    state = SupervisorState(user_message="Same question")
    a = specialist.handle(state)
    b = specialist.handle(state)
    c = specialist.handle(state)

    assert invoker.call_count == 3
    # The fake invoker returns different content each call, so without the
    # cache the answers differ — exactly the user-reported issue.
    assert a.content != b.content != c.content


def test_genie_does_not_cache_failures() -> None:
    """A transient failure must not poison the cache for subsequent calls."""
    state = SupervisorState(user_message="Same question")

    class FlakyInvoker:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, message: str) -> dict:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient")
            return {
                "messages": [{"role": "assistant", "content": "ok"}],
                "rows": [],
                "context": {"sql": "SELECT 1"},
            }

    invoker = FlakyInvoker()
    cache = InMemoryResponseCache(ttl_seconds=60.0)
    specialist = GenieSpecialist(invoker=invoker, cache=cache)

    failed = specialist.handle(state)
    recovered = specialist.handle(state)

    assert "Could not retrieve structured data" in failed.content
    assert recovered.content == "ok"
    assert invoker.calls == 2
