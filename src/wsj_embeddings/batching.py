"""Bounded, rate-aware scheduling for synchronous hosted Jina requests."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from wsj_embeddings.adapters import (
    JinaBatchItemOutcome,
    JinaBatchLimits,
    JinaEmbeddingBatchResponse,
    JinaEmbeddingInput,
    JinaHostedAdapterError,
)

_RUN_FATAL_CODES = frozenset(
    {
        "authentication",
        "authorization",
        "deterministic_request",
        "missing_credential",
        "request_failure",
    }
)


@dataclass(frozen=True, slots=True)
class BatchPolicy:
    """Immutable request, concurrency, and retry ceilings."""

    max_items: int
    max_estimated_tokens: int
    max_encoded_bytes: int
    max_concurrency: int
    max_attempts: int
    initial_backoff_seconds: float
    max_backoff_seconds: float

    def __post_init__(self) -> None:
        integer_values = (
            self.max_items,
            self.max_estimated_tokens,
            self.max_encoded_bytes,
            self.max_concurrency,
            self.max_attempts,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in integer_values
        ):
            raise ValueError("batch integer limits must be positive")
        if (
            self.initial_backoff_seconds < 0
            or self.max_backoff_seconds < self.initial_backoff_seconds
        ):
            raise ValueError("batch backoff limits are invalid")

    @property
    def request_limits(self) -> JinaBatchLimits:
        return JinaBatchLimits(
            self.max_items,
            self.max_estimated_tokens,
            self.max_encoded_bytes,
        )


@dataclass(frozen=True, slots=True)
class BatchExecutionResult:
    """Input-ordered outcomes and content-free scheduling observations."""

    items: tuple[JinaBatchItemOutcome, ...]
    requests: int
    retries: int
    usage: dict[str, int | float]
    throttles: int
    elapsed_seconds: float
    max_concurrency_observed: int


@dataclass(frozen=True, slots=True)
class _PendingItem:
    index: int
    value: JinaEmbeddingInput
    attempt: int = 1


def execute_rate_aware_batches(
    adapter: object,
    inputs: Sequence[JinaEmbeddingInput],
    *,
    policy: BatchPolicy,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[int], float] = lambda _attempt: 0.0,
    monotonic: Callable[[], float] = time.monotonic,
    before_attempt: Callable[[int, int], None] | None = None,
) -> BatchExecutionResult:
    """Run bounded waves, retaining every independently validated success."""

    started_at = monotonic()
    pending = [_PendingItem(index, value) for index, value in enumerate(inputs)]
    final: dict[int, JinaBatchItemOutcome] = {}
    usage: dict[str, int | float] = {}
    requests = 0
    retries = 0
    throttles = 0
    concurrency = policy.max_concurrency
    observed_concurrency = 0
    while pending:
        batches = _pack(pending, policy)
        wave = batches[:concurrency]
        deferred = [item for batch in batches[concurrency:] for item in batch]
        observed_concurrency = max(observed_concurrency, len(wave))
        if before_attempt is not None:
            for batch in wave:
                for item in batch:
                    before_attempt(item.index, item.attempt)
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    adapter.embed_batch,
                    tuple(item.value for item in batch),
                    limits=policy.request_limits,
                )
                for batch in wave
            ]
            exchanges: list[
                tuple[
                    list[_PendingItem],
                    JinaEmbeddingBatchResponse | JinaHostedAdapterError,
                ]
            ] = []
            for batch, future in zip(wave, futures, strict=True):
                try:
                    exchange = future.result()
                except JinaHostedAdapterError as error:
                    exchange = error
                exchanges.append((batch, exchange))
        requests += len(wave)
        retry_delays: list[float] = []
        next_pending = list(deferred)
        throttled_wave = False
        for batch, exchange in exchanges:
            if isinstance(exchange, JinaHostedAdapterError):
                if exchange.code in _RUN_FATAL_CODES or not exchange.retryable:
                    raise exchange
                throttled = exchange.code == "rate_limit"
                if throttled:
                    throttles += 1
                    throttled_wave = True
                for item in batch:
                    if item.attempt >= policy.max_attempts:
                        final[item.index] = JinaBatchItemOutcome(
                            item.index, None, exchange.code, retryable=True
                        )
                    else:
                        retries += 1
                        next_pending.append(
                            _PendingItem(item.index, item.value, item.attempt + 1)
                        )
                        retry_delays.append(
                            _retry_delay(
                                item.attempt,
                                policy,
                                jitter,
                                retry_after=exchange.retry_after_seconds,
                                headers=exchange.rate_limit_headers,
                            )
                        )
                continue
            _add_usage(usage, exchange.usage)
            headers = exchange.response_metadata.rate_limit_headers
            throttled = _is_throttled(headers)
            if throttled:
                throttles += 1
                throttled_wave = True
                retry_delays.append(_rate_header_delay(headers))
            if len(exchange.items) != len(batch):
                raise JinaHostedAdapterError("invalid_response", retryable=False)
            for item, outcome in zip(batch, exchange.items, strict=True):
                if outcome.index != batch.index(item):
                    raise JinaHostedAdapterError("invalid_response", retryable=False)
                if outcome.vector is not None or not outcome.retryable:
                    final[item.index] = JinaBatchItemOutcome(
                        item.index,
                        outcome.vector,
                        outcome.error_code,
                        retryable=False,
                    )
                    continue
                if item.attempt >= policy.max_attempts:
                    final[item.index] = JinaBatchItemOutcome(
                        item.index, None, outcome.error_code, retryable=True
                    )
                    continue
                retries += 1
                next_pending.append(
                    _PendingItem(item.index, item.value, item.attempt + 1)
                )
                retry_delays.append(
                    _retry_delay(
                        item.attempt,
                        policy,
                        jitter,
                        retry_after=headers.get("retry-after"),
                        headers=headers,
                    )
                )
        if throttled_wave:
            concurrency = max(1, concurrency // 2)
        if retry_delays:
            sleep(max(retry_delays))
        pending = sorted(next_pending, key=lambda item: item.index)
    elapsed = max(0.0, monotonic() - started_at)
    return BatchExecutionResult(
        items=tuple(final[index] for index in range(len(inputs))),
        requests=requests,
        retries=retries,
        usage=usage,
        throttles=throttles,
        elapsed_seconds=elapsed,
        max_concurrency_observed=observed_concurrency,
    )


def _pack(
    pending: Sequence[_PendingItem], policy: BatchPolicy
) -> list[list[_PendingItem]]:
    batches: list[list[_PendingItem]] = []
    current: list[_PendingItem] = []
    tokens = 0
    encoded_bytes = 0
    for item in pending:
        item_tokens = item.value.estimated_tokens
        item_encoded_bytes = item.value.encoded_bytes
        if (
            item_tokens > policy.max_estimated_tokens
            or item_encoded_bytes > policy.max_encoded_bytes
        ):
            raise JinaHostedAdapterError("deterministic_request", retryable=False)
        exceeds = current and (
            len(current) + 1 > policy.max_items
            or tokens + item_tokens > policy.max_estimated_tokens
            or encoded_bytes + item_encoded_bytes > policy.max_encoded_bytes
        )
        if exceeds:
            batches.append(current)
            current = []
            tokens = 0
            encoded_bytes = 0
        current.append(item)
        tokens += item_tokens
        encoded_bytes += item_encoded_bytes
    if current:
        batches.append(current)
    return batches


def _retry_delay(
    attempt: int,
    policy: BatchPolicy,
    jitter: Callable[[int], float],
    *,
    retry_after: float | None,
    headers: dict[str, float] | object,
) -> float:
    backoff = min(
        policy.max_backoff_seconds,
        policy.initial_backoff_seconds * (2 ** (attempt - 1)),
    )
    header_values = [backoff]
    if retry_after is not None:
        header_values.append(retry_after)
    if isinstance(headers, dict):
        if headers.get("x-ratelimit-remaining-requests") == 0:
            header_values.append(headers.get("x-ratelimit-reset-requests", 0.0))
        if headers.get("x-ratelimit-remaining-tokens") == 0:
            header_values.append(headers.get("x-ratelimit-reset-tokens", 0.0))
    jitter_seconds = jitter(attempt)
    if jitter_seconds < 0:
        raise ValueError("batch retry jitter must be nonnegative")
    return min(policy.max_backoff_seconds, max(header_values)) + jitter_seconds


def _is_throttled(headers: object) -> bool:
    return isinstance(headers, dict) and (
        headers.get("x-ratelimit-remaining-requests") == 0
        or headers.get("x-ratelimit-remaining-tokens") == 0
    )


def _rate_header_delay(headers: object) -> float:
    if not isinstance(headers, dict):
        return 0.0
    values = [headers.get("retry-after", 0.0)]
    if headers.get("x-ratelimit-remaining-requests") == 0:
        values.append(headers.get("x-ratelimit-reset-requests", 0.0))
    if headers.get("x-ratelimit-remaining-tokens") == 0:
        values.append(headers.get("x-ratelimit-reset-tokens", 0.0))
    return max(values)


def _add_usage(
    aggregate: dict[str, int | float], usage: dict[str, int | float]
) -> None:
    for name, value in usage.items():
        aggregate[name] = aggregate.get(name, 0) + value
