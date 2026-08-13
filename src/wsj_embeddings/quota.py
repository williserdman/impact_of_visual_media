"""Pure hosted-quota policy and rolling-window calculations."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Protocol

MAX_SAFE_OBSERVED_INPUT_TOKENS = 1_000_000_000


class HostedQuotaProfile(Protocol):
    """The immutable profile fields that affect hosted admission."""

    batch_max_concurrency: int
    batch_max_estimated_tokens: int
    quota_max_requests: int
    quota_max_estimated_tokens: int
    quota_window_seconds: int


def hosted_quota_policy_is_valid(profile: HostedQuotaProfile) -> bool:
    """Return whether one immutable batch/quota policy is coherent."""

    values = (
        profile.batch_max_concurrency,
        profile.batch_max_estimated_tokens,
        profile.quota_max_requests,
        profile.quota_max_estimated_tokens,
        profile.quota_window_seconds,
    )
    return (
        all(
            not isinstance(value, bool)
            and isinstance(value, int)
            and value > 0
            for value in values
        )
        and profile.batch_max_concurrency <= profile.quota_max_requests
        and profile.batch_max_concurrency * profile.batch_max_estimated_tokens
        <= profile.quota_max_estimated_tokens
    )


def safe_observed_input_tokens(usage: object) -> int | None:
    """Extract one bounded integral provider input-token observation."""

    if not isinstance(usage, Mapping):
        return None
    observed = usage.get("input_tokens", usage.get("prompt_tokens"))
    if isinstance(observed, bool) or not isinstance(observed, (int, float)):
        return None
    try:
        number = float(observed)
    except (OverflowError, TypeError, ValueError):
        return None
    if (
        not math.isfinite(number)
        or number < 0
        or number > MAX_SAFE_OBSERVED_INPUT_TOKENS
        or not number.is_integer()
    ):
        return None
    return int(number)


def rolling_quota_retry_after(
    reservations: Sequence[tuple[datetime, int]],
    proposed_costs: Sequence[int],
    *,
    now: datetime,
    profile: HostedQuotaProfile,
) -> float:
    """Return zero for admission or the earliest delay that fits all limits."""

    costs = tuple(proposed_costs)
    if not hosted_quota_policy_is_valid(profile):
        raise ValueError("hosted quota policy is invalid")
    if (
        not costs
        or len(costs) > profile.batch_max_concurrency
        or any(
            isinstance(cost, bool)
            or not isinstance(cost, int)
            or cost < 0
            or cost > profile.batch_max_estimated_tokens
            for cost in costs
        )
        or len(costs) > profile.quota_max_requests
        or sum(costs) > profile.quota_max_estimated_tokens
    ):
        raise ValueError("hosted wave exceeds quota policy")
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("hosted quota timestamp must be timezone-aware")

    ordered = sorted(reservations, key=lambda reservation: reservation[0])
    if len(ordered) > profile.quota_max_requests or any(
        isinstance(tokens, bool)
        or not isinstance(tokens, int)
        or tokens < 0
        or tokens > MAX_SAFE_OBSERVED_INPUT_TOKENS
        or reserved_at.tzinfo is None
        or reserved_at.utcoffset() is None
        for reserved_at, tokens in ordered
    ):
        raise ValueError("hosted quota reservation state is invalid")

    proposed_requests = len(costs)
    proposed_tokens = sum(costs)
    remaining_requests = len(ordered)
    remaining_tokens = sum(tokens for _reserved_at, tokens in ordered)
    if (
        remaining_requests + proposed_requests <= profile.quota_max_requests
        and remaining_tokens + proposed_tokens
        <= profile.quota_max_estimated_tokens
    ):
        return 0.0

    for reserved_at, effective_tokens in ordered:
        remaining_requests -= 1
        remaining_tokens -= effective_tokens
        if (
            remaining_requests + proposed_requests
            <= profile.quota_max_requests
            and remaining_tokens + proposed_tokens
            <= profile.quota_max_estimated_tokens
        ):
            retry_at = reserved_at + timedelta(
                seconds=profile.quota_window_seconds
            )
            return max(0.0, (retry_at - now).total_seconds())
    raise ValueError("hosted quota reservation state is invalid")
