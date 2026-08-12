"""Content-free normalization for hosted run observations."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping

SAFE_USAGE_FIELDS = frozenset(
    {"input_tokens", "output_tokens", "prompt_tokens", "total_tokens"}
)
_SAFE_USAGE_KEY = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def normalize_safe_usage(value: object) -> dict[str, int | float]:
    """Drop unknown, content-bearing, or invalid hosted usage values."""

    if not isinstance(value, Mapping):
        return {}
    normalized: dict[str, int | float] = {}
    for key, count in value.items():
        if (
            not isinstance(key, str)
            or _SAFE_USAGE_KEY.fullmatch(key) is None
            or key not in SAFE_USAGE_FIELDS
            or isinstance(count, bool)
            or not isinstance(count, (int, float))
        ):
            continue
        number = float(count)
        if not math.isfinite(number) or number < 0:
            continue
        normalized[key] = count
    return normalized


def safe_usage_is_valid(value: object) -> bool:
    """Return whether decoded JSON already has the complete safe shape."""

    return isinstance(value, dict) and normalize_safe_usage(value) == value
