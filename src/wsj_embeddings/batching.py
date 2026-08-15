"""Bounded, rate-aware scheduling for synchronous hosted Jina requests."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from random import SystemRandom

from wsj_embeddings.adapters import (
    JinaBatchItemOutcome,
    JinaBatchLimits,
    JinaEmbeddingBatchResponse,
    JinaEmbeddingInput,
    JinaHostedAdapterError,
)
from wsj_embeddings.run_metrics import normalize_safe_usage

_RUN_FATAL_CODES = frozenset(
    {
        "authentication",
        "authorization",
        "deterministic_request",
        "missing_credential",
        "request_failure",
    }
)
_SYSTEM_RANDOM = SystemRandom()


def _production_jitter(_attempt: int) -> float:
    return _SYSTEM_RANDOM.uniform(0.0, 1.0)


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
    max_response_bytes: int = 2_000_000

    def __post_init__(self) -> None:
        integer_values = (
            self.max_items,
            self.max_estimated_tokens,
            self.max_encoded_bytes,
            self.max_concurrency,
            self.max_attempts,
            self.max_response_bytes,
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
            self.max_response_bytes,
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


class BatchExecutionFatal(JinaHostedAdapterError):
    """Fatal hosted error carrying all safe observations completed before it."""

    def __init__(
        self,
        error: JinaHostedAdapterError,
        *,
        requests: int,
        retries: int,
        usage: dict[str, int | float],
        throttles: int,
        elapsed_seconds: float,
    ) -> None:
        super().__init__(
            error.code,
            retryable=error.retryable,
            status_code=error.status_code,
            retry_after_seconds=error.retry_after_seconds,
            rate_limit_headers=error.rate_limit_headers,
        )
        self.requests = requests
        self.retries = retries
        self.usage = dict(usage)
        self.throttles = throttles
        self.elapsed_seconds = elapsed_seconds


@dataclass(frozen=True, slots=True)
class _PendingItem:
    index: int
    value: JinaEmbeddingInput
    attempt: int = 1
    attempt_limit: int = 1


def execute_rate_aware_batches(
    adapter: object,
    inputs: Sequence[JinaEmbeddingInput],
    *,
    policy: BatchPolicy,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[int], float] = _production_jitter,
    monotonic: Callable[[], float] = time.monotonic,
    before_attempt: Callable[[int, int], None] | None = None,
    before_wave: (
        Callable[[tuple[tuple[JinaEmbeddingInput, ...], ...]], None] | None
    ) = None,
    after_wave: Callable[[tuple[dict[str, int | float], ...]], None] | None = None,
    on_outcome: Callable[[int, int, JinaBatchItemOutcome, bool], None] | None = None,
    prior_attempt_counts: Sequence[int] | None = None,
    attempt_limits: Sequence[int] | None = None,
) -> BatchExecutionResult:
    """Run bounded waves, retaining every independently validated success."""

    started_at = monotonic()
    if prior_attempt_counts is None:
        prior_attempt_counts = (0,) * len(inputs)
    if attempt_limits is None:
        attempt_limits = (policy.max_attempts,) * len(inputs)
    if len(prior_attempt_counts) != len(inputs) or any(
        isinstance(count, bool) or not isinstance(count, int) or count < 0
        for count in prior_attempt_counts
    ):
        raise ValueError("prior attempt counts must be nonnegative integers per input")
    if len(attempt_limits) != len(inputs) or any(
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
        for limit in attempt_limits
    ):
        raise ValueError("attempt limits must be positive integers per input")
    pending: list[_PendingItem] = []
    final: dict[int, JinaBatchItemOutcome] = {}
    for index, (value, prior_attempts, attempt_limit) in enumerate(
        zip(inputs, prior_attempt_counts, attempt_limits, strict=True)
    ):
        if prior_attempts >= attempt_limit:
            outcome = JinaBatchItemOutcome(
                index, None, "attempt_limit_exhausted", retryable=False
            )
            final[index] = outcome
            if on_outcome is not None:
                on_outcome(index, prior_attempts, outcome, True)
        else:
            pending.append(
                _PendingItem(index, value, prior_attempts + 1, attempt_limit)
            )
    usage: dict[str, int | float] = {}
    requests = 0
    retries = 0
    throttles = 0
    concurrency = policy.max_concurrency
    observed_concurrency = 0
    while pending:
        batches, rejected = _pack(pending, policy)
        for item in rejected:
            outcome = JinaBatchItemOutcome(
                item.index,
                None,
                "deterministic_request",
                retryable=False,
            )
            final[item.index] = outcome
            if on_outcome is not None:
                on_outcome(item.index, item.attempt, outcome, True)
        if not batches:
            break
        wave = batches[:concurrency]
        deferred = [item for batch in batches[concurrency:] for item in batch]
        observed_concurrency = max(observed_concurrency, len(wave))
        if before_wave is not None:
            before_wave(
                tuple(tuple(item.value for item in batch) for batch in wave)
            )
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
        if after_wave is not None:
            after_wave(
                tuple(
                    dict(exchange.usage)
                    if isinstance(exchange, JinaEmbeddingBatchResponse)
                    else {}
                    for _batch, exchange in exchanges
                )
            )
        requests += len(wave)
        retry_delays: list[float] = []
        next_pending = list(deferred)
        throttled_wave = False
        fatal_errors: list[JinaHostedAdapterError] = []
        for batch, exchange in exchanges:
            if isinstance(exchange, JinaHostedAdapterError):
                if exchange.code in _RUN_FATAL_CODES or not exchange.retryable:
                    if exchange.code == "deterministic_request":
                        for item in batch:
                            outcome = JinaBatchItemOutcome(
                                item.index,
                                None,
                                exchange.code,
                                retryable=False,
                                status_code=exchange.status_code,
                                retry_after_seconds=exchange.retry_after_seconds,
                            )
                            final[item.index] = outcome
                            if on_outcome is not None:
                                on_outcome(item.index, item.attempt, outcome, True)
                    fatal_errors.append(exchange)
                    continue
                throttled = exchange.code == "rate_limit"
                if throttled:
                    throttles += 1
                    throttled_wave = True
                for item in batch:
                    if item.attempt >= item.attempt_limit:
                        outcome = JinaBatchItemOutcome(
                            item.index,
                            None,
                            exchange.code,
                            retryable=False,
                            status_code=exchange.status_code,
                            retry_after_seconds=exchange.retry_after_seconds,
                        )
                        final[item.index] = outcome
                        if on_outcome is not None:
                            on_outcome(item.index, item.attempt, outcome, True)
                    else:
                        outcome = JinaBatchItemOutcome(
                            item.index,
                            None,
                            exchange.code,
                            retryable=True,
                            status_code=exchange.status_code,
                            retry_after_seconds=exchange.retry_after_seconds,
                        )
                        if on_outcome is not None:
                            on_outcome(item.index, item.attempt, outcome, False)
                        retries += 1
                        next_pending.append(
                            _PendingItem(
                                item.index,
                                item.value,
                                item.attempt + 1,
                                item.attempt_limit,
                            )
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
                retry_delays.append(
                    _rate_header_delay(headers, policy.max_backoff_seconds)
                )
            if len(exchange.items) != len(batch):
                fatal_errors.append(
                    JinaHostedAdapterError("invalid_response", retryable=False)
                )
                continue
            malformed_indexes = any(
                outcome.index != local_index
                for local_index, outcome in enumerate(exchange.items)
            )
            if malformed_indexes:
                fatal_errors.append(
                    JinaHostedAdapterError("invalid_response", retryable=False)
                )
                continue
            for item, outcome in zip(batch, exchange.items, strict=True):
                deterministic_image_retry = (
                    outcome.vector is None
                    and outcome.error_code == "deterministic_request"
                    and item.value.kind == "image"
                )
                retryable_item = outcome.retryable or deterministic_image_retry
                if outcome.vector is not None or not retryable_item:
                    delivered = JinaBatchItemOutcome(
                        item.index,
                        outcome.vector,
                        outcome.error_code,
                        retryable=False,
                        status_code=outcome.status_code,
                        retry_after_seconds=outcome.retry_after_seconds,
                    )
                    final[item.index] = delivered
                    if on_outcome is not None:
                        on_outcome(item.index, item.attempt, delivered, True)
                    continue
                if item.attempt >= item.attempt_limit:
                    delivered = JinaBatchItemOutcome(
                        item.index,
                        None,
                        outcome.error_code,
                        retryable=False,
                        status_code=outcome.status_code,
                        retry_after_seconds=outcome.retry_after_seconds,
                    )
                    final[item.index] = delivered
                    if on_outcome is not None:
                        on_outcome(item.index, item.attempt, delivered, True)
                    continue
                if on_outcome is not None:
                    on_outcome(
                        item.index,
                        item.attempt,
                        JinaBatchItemOutcome(
                            item.index,
                            None,
                            outcome.error_code,
                            retryable=True,
                            status_code=outcome.status_code,
                            retry_after_seconds=outcome.retry_after_seconds,
                        ),
                        False,
                    )
                retries += 1
                next_pending.append(
                    _PendingItem(
                        item.index,
                        item.value,
                        item.attempt + 1,
                        item.attempt_limit,
                    )
                )
                retry_delays.append(
                    _retry_delay(
                        item.attempt,
                        policy,
                        jitter,
                        retry_after=outcome.retry_after_seconds,
                        headers=headers,
                    )
                )
        if fatal_errors:
            raise BatchExecutionFatal(
                fatal_errors[0],
                requests=requests,
                retries=retries,
                usage=usage,
                throttles=throttles,
                elapsed_seconds=max(0.0, monotonic() - started_at),
            ) from None
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
) -> tuple[list[list[_PendingItem]], list[_PendingItem]]:
    batches: list[list[_PendingItem]] = []
    rejected: list[_PendingItem] = []
    current: list[_PendingItem] = []
    tokens = 0
    encoded_bytes = 0
    for item in pending:
        try:
            item.value.as_request_item()
        except JinaHostedAdapterError:
            rejected.append(item)
            continue
        item_tokens = item.value.estimated_tokens
        item_encoded_bytes = item.value.encoded_bytes
        if (
            item_tokens > policy.max_estimated_tokens
            or item_encoded_bytes > policy.max_encoded_bytes
        ):
            rejected.append(item)
            continue
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
    return batches, rejected


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
    return min(policy.max_backoff_seconds, max(header_values) + jitter_seconds)


def _is_throttled(headers: object) -> bool:
    return isinstance(headers, dict) and (
        headers.get("x-ratelimit-remaining-requests") == 0
        or headers.get("x-ratelimit-remaining-tokens") == 0
    )


def _rate_header_delay(headers: object, maximum: float) -> float:
    if not isinstance(headers, dict):
        return 0.0
    values = [headers.get("retry-after", 0.0)]
    if headers.get("x-ratelimit-remaining-requests") == 0:
        values.append(headers.get("x-ratelimit-reset-requests", 0.0))
    if headers.get("x-ratelimit-remaining-tokens") == 0:
        values.append(headers.get("x-ratelimit-reset-tokens", 0.0))
    return min(maximum, max(values))


def _add_usage(
    aggregate: dict[str, int | float], usage: dict[str, int | float]
) -> None:
    for name, value in normalize_safe_usage(usage).items():
        aggregate[name] = aggregate.get(name, 0) + value
