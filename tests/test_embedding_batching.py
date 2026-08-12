"""Deterministic rate-aware scheduling tests; no network or licensed content."""

from __future__ import annotations

import threading
from collections import deque

import pytest

import wsj_embeddings.batching as batching_module
from wsj_embeddings.adapters import (
    JinaBatchItemOutcome,
    JinaEmbeddedVector,
    JinaEmbeddingBatchResponse,
    JinaEmbeddingInput,
    JinaHostedAdapterError,
    JinaResponseMetadata,
)
from wsj_embeddings.batching import BatchPolicy, execute_rate_aware_batches


def _vector(index: int) -> JinaEmbeddedVector:
    values = (1.0, *(0.0 for _ in range(2047)))
    return JinaEmbeddedVector(index, values, values, "a" * 64, "b" * 64)


def _response(
    count: int,
    *,
    retryable_indexes: tuple[int, ...] = (),
    headers: dict[str, float] | None = None,
    usage: dict[str, int | float] | None = None,
) -> JinaEmbeddingBatchResponse:
    return JinaEmbeddingBatchResponse(
        items=tuple(
            JinaBatchItemOutcome(
                index,
                None if index in retryable_indexes else _vector(index),
                "missing_response_item" if index in retryable_indexes else None,
                retryable=index in retryable_indexes,
                status_code=200 if index in retryable_indexes else None,
                retry_after_seconds=(headers or {}).get("retry-after"),
            )
            for index in range(count)
        ),
        usage={} if usage is None else usage,
        response_metadata=JinaResponseMetadata(
            200, "jina-embeddings-v4", headers or {}
        ),
    )


class ScenarioBatchAdapter:
    def __init__(self, responses):
        self.responses = deque(responses)
        self.calls: list[tuple[JinaEmbeddingInput, ...]] = []
        self.active = 0
        self.max_active = 0
        self.gate: threading.Barrier | None = None
        self.lock = threading.Lock()

    def embed_batch(self, inputs, *, limits):
        with self.lock:
            self.calls.append(tuple(inputs))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.gate is not None:
                self.gate.wait(timeout=2)
            response = self.responses.popleft()
            if isinstance(response, BaseException):
                raise response
            return response(len(inputs)) if callable(response) else response
        finally:
            with self.lock:
                self.active -= 1


def test_scheduler_packs_item_token_and_encoded_byte_limits_independently():
    """Break caught: batching treats one payload budget as a proxy for all three."""

    adapter = ScenarioBatchAdapter([_response(1), _response(1), _response(2)])
    inputs = (
        JinaEmbeddingInput.text("abc"),
        JinaEmbeddingInput.image_base64("aW1hZ2U="),
        JinaEmbeddingInput.text("wxyz"),
        JinaEmbeddingInput.image_base64("YQ=="),
    )

    result = execute_rate_aware_batches(
        adapter,
        inputs,
        policy=BatchPolicy(3, 6, 8, 1, 2, 1.0, 8.0),
        sleep=lambda _seconds: None,
        jitter=lambda _attempt: 0.0,
        monotonic=iter((10.0, 12.5)).__next__,
    )

    assert adapter.calls == [inputs[:1], inputs[1:2], inputs[2:]]
    assert all(item.vector is not None for item in result.items)
    assert result.requests == 3
    assert result.elapsed_seconds == 2.5


def test_scheduler_honors_retry_after_backoff_jitter_usage_and_partial_retry():
    """Break caught: throttle delay is ignored or a partial success is repurchased."""

    sleeps: list[float] = []
    adapter = ScenarioBatchAdapter(
        [
            lambda count: _response(
                count,
                retryable_indexes=(1,),
                headers={
                    "retry-after": 5.0,
                    "x-ratelimit-remaining-requests": 0.0,
                    "x-ratelimit-reset-requests": 4.0,
                },
                usage={"prompt_tokens": 7, "total_tokens": 7},
            ),
            lambda count: _response(
                count,
                usage={"prompt_tokens": 3, "total_tokens": 3},
            ),
        ]
    )
    inputs = (JinaEmbeddingInput.text("one"), JinaEmbeddingInput.text("two"))

    result = execute_rate_aware_batches(
        adapter,
        inputs,
        policy=BatchPolicy(2, 10, 10, 2, 3, 1.0, 8.0),
        sleep=sleeps.append,
        jitter=lambda attempt: 0.25 * attempt,
        monotonic=iter((1.0, 2.0)).__next__,
    )

    assert adapter.calls == [inputs, inputs[1:]]
    assert sleeps == [5.25]
    assert result.retries == 1
    assert result.throttles == 1
    assert result.usage == {"prompt_tokens": 10, "total_tokens": 10}
    assert all(item.vector is not None for item in result.items)


def test_scheduler_retries_timeout_without_losing_item_identity():
    """Break caught: a transport timeout becomes terminal or reorders its item."""

    adapter = ScenarioBatchAdapter(
        [
            JinaHostedAdapterError("timeout", retryable=True),
            lambda count: _response(count),
        ]
    )
    inputs = (JinaEmbeddingInput.text("generated timeout fixture"),)

    result = execute_rate_aware_batches(
        adapter,
        inputs,
        policy=BatchPolicy(1, 100, 100, 1, 2, 0.0, 0.0),
        sleep=lambda _seconds: None,
        jitter=lambda _attempt: 0.0,
        monotonic=iter((1.0, 2.0)).__next__,
    )

    assert adapter.calls == [inputs, inputs]
    assert result.retries == 1
    assert result.items[0].index == 0
    assert result.items[0].vector is not None


def test_scheduler_reduces_throttled_concurrency_without_exceeding_maximum():
    """Break caught: throttle signals leave fan-out unchanged or exceed its ceiling."""

    throttle = JinaHostedAdapterError(
        "rate_limit",
        retryable=True,
        status_code=429,
        retry_after_seconds=0.5,
        rate_limit_headers={"x-ratelimit-remaining-requests": 0.0},
    )
    adapter = ScenarioBatchAdapter(
        [
            throttle,
            lambda count: _response(count),
            lambda count: _response(count),
            lambda count: _response(count),
            lambda count: _response(count),
        ]
    )
    adapter.gate = threading.Barrier(2)
    inputs = tuple(JinaEmbeddingInput.text(str(index)) for index in range(4))

    result = execute_rate_aware_batches(
        adapter,
        inputs,
        policy=BatchPolicy(1, 10, 10, 2, 2, 0.1, 1.0),
        sleep=lambda _seconds: setattr(adapter, "gate", None),
        jitter=lambda _attempt: 0.0,
        monotonic=iter((1.0, 2.0)).__next__,
    )

    assert adapter.max_active == 2
    assert result.max_concurrency_observed == 2
    assert result.throttles == 1
    assert result.retries == 1
    assert all(item.vector is not None for item in result.items)


def test_scheduler_waits_for_rate_reset_before_deferred_work():
    """Break caught: zero remaining requests immediately starts another wave."""

    sleeps: list[float] = []
    adapter = ScenarioBatchAdapter(
        [
            _response(
                1,
                headers={
                    "x-ratelimit-remaining-requests": 0.0,
                    "x-ratelimit-reset-requests": 3.0,
                },
            ),
            _response(1),
        ]
    )

    result = execute_rate_aware_batches(
        adapter,
        (JinaEmbeddingInput.text("a"), JinaEmbeddingInput.text("b")),
        policy=BatchPolicy(1, 10, 10, 1, 2, 1.0, 8.0),
        sleep=sleeps.append,
        jitter=lambda _attempt: 0.0,
        monotonic=iter((1.0, 2.0)).__next__,
    )

    assert sleeps == [3.0]
    assert result.throttles == 1
    assert result.retries == 0


def test_scheduler_emits_each_wave_outcome_before_later_fatal_request():
    """Break caught: an earlier success remains volatile until all waves finish."""

    adapter = ScenarioBatchAdapter(
        [
            _response(1),
            JinaHostedAdapterError(
                "authentication", retryable=False, status_code=401
            ),
        ]
    )
    committed: list[tuple[int, int, bool]] = []

    with pytest.raises(JinaHostedAdapterError, match="authentication"):
        execute_rate_aware_batches(
            adapter,
            (JinaEmbeddingInput.text("first"), JinaEmbeddingInput.text("second")),
            policy=BatchPolicy(1, 10, 10, 1, 2, 1.0, 8.0),
            on_outcome=lambda index, attempt, outcome, final: committed.append(
                (index, attempt, outcome.vector is not None and final)
            ),
            jitter=lambda _attempt: 0.0,
        )

    assert committed == [(0, 1, True)]


def test_scheduler_uses_durable_prior_attempt_count_as_total_budget():
    """Break caught: replay starts at attempt one and purchases past the durable cap."""

    adapter = ScenarioBatchAdapter(
        [JinaHostedAdapterError("timeout", retryable=True, status_code=504)]
    )
    observed: list[tuple[int, int, str | None, bool]] = []

    result = execute_rate_aware_batches(
        adapter,
        (JinaEmbeddingInput.text("entering attempt two"),),
        policy=BatchPolicy(1, 100, 100, 1, 2, 1.0, 8.0),
        prior_attempt_counts=(1,),
        on_outcome=lambda index, attempt, outcome, final: observed.append(
            (index, attempt, outcome.error_code, final)
        ),
        sleep=lambda _seconds: None,
        jitter=lambda _attempt: 0.0,
    )

    assert len(adapter.calls) == 1
    assert result.retries == 0
    assert observed == [(0, 2, "timeout", True)]


def test_scheduler_caps_jittered_retry_and_rate_reset_delays():
    """Break caught: jitter or a usable rate header pushes sleep past policy max."""

    sleeps: list[float] = []
    adapter = ScenarioBatchAdapter(
        [
            JinaHostedAdapterError(
                "rate_limit",
                retryable=True,
                status_code=429,
                retry_after_seconds=100.0,
            ),
            _response(
                1,
                headers={
                    "x-ratelimit-remaining-requests": 0.0,
                    "x-ratelimit-reset-requests": 200.0,
                },
            ),
            _response(1),
        ]
    )

    execute_rate_aware_batches(
        adapter,
        (JinaEmbeddingInput.text("retry"), JinaEmbeddingInput.text("deferred")),
        policy=BatchPolicy(1, 100, 100, 1, 2, 1.0, 3.0),
        sleep=sleeps.append,
        jitter=lambda _attempt: 50.0,
    )

    assert sleeps == [3.0, 3.0]


def test_scheduler_default_retry_jitter_is_real_and_bounded(monkeypatch):
    """Break caught: production retries synchronize because default jitter is zero."""

    sleeps: list[float] = []
    monkeypatch.setattr(
        batching_module._SYSTEM_RANDOM, "uniform", lambda _low, _high: 0.4
    )
    adapter = ScenarioBatchAdapter(
        [JinaHostedAdapterError("timeout", retryable=True), _response(1)]
    )

    execute_rate_aware_batches(
        adapter,
        (JinaEmbeddingInput.text("default jitter"),),
        policy=BatchPolicy(1, 100, 100, 1, 2, 1.0, 3.0),
        sleep=sleeps.append,
    )

    assert sleeps == [1.4]


def test_scheduler_honors_item_retry_after_metadata_without_numeric_header():
    """Break caught: parsed HTTP-date delay is lost between adapter and scheduler."""

    sleeps: list[float] = []
    retryable = JinaEmbeddingBatchResponse(
        items=(
            JinaBatchItemOutcome(
                0,
                None,
                "missing_response_item",
                retryable=True,
                status_code=200,
                retry_after_seconds=5.0,
            ),
        ),
        usage={},
        response_metadata=JinaResponseMetadata(200, "jina-embeddings-v4", {}),
    )
    adapter = ScenarioBatchAdapter([retryable, _response(1)])

    execute_rate_aware_batches(
        adapter,
        (JinaEmbeddingInput.text("item retry"),),
        policy=BatchPolicy(1, 100, 100, 1, 2, 1.0, 8.0),
        sleep=sleeps.append,
        jitter=lambda _attempt: 0.0,
    )

    assert sleeps == [5.0]


@pytest.mark.parametrize("code", ("authentication", "authorization"))
def test_scheduler_stops_run_scope_without_retrying_auth_failures(code):
    """Break caught: credential or permission failures enter transient retries."""

    failure = JinaHostedAdapterError(code, retryable=False, status_code=401)
    adapter = ScenarioBatchAdapter([failure])

    with pytest.raises(JinaHostedAdapterError) as raised:
        execute_rate_aware_batches(
            adapter,
            (JinaEmbeddingInput.text("generated"),),
            policy=BatchPolicy(1, 10, 10, 1, 3, 1.0, 8.0),
            sleep=lambda _seconds: None,
            jitter=lambda _attempt: 0.0,
        )

    assert raised.value.code == code
    assert len(adapter.calls) == 1
