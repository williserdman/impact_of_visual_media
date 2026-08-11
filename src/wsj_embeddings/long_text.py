"""Lossless deterministic planning for over-context canonical Markdown."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Protocol


class TextOffsetTokenizer(Protocol):
    """Tokenizer seam exposing exact source offsets without model inference."""

    def token_offsets(self, text: str) -> tuple[tuple[int, int], ...]: ...


@dataclass(frozen=True, slots=True)
class LongTextPart:
    """One exact Markdown substring and its content-free durable identity."""

    index: int
    text: str
    input_sha256: str
    token_count: int


class LongTextPlanningError(RuntimeError):
    """Raised when tokenizer offsets cannot prove lossless source coverage."""


_BLOCK_SEPARATOR = re.compile(r"(?:\r?\n[ \t]*){2,}")


def token_count(text: str, tokenizer: TextOffsetTokenizer) -> int:
    """Count tokens only after proving offsets cover the exact input once."""

    return len(_validated_offsets(text, tokenizer.token_offsets(text)))


def plan_long_text_parts(
    markdown: str,
    tokenizer: TextOffsetTokenizer,
    *,
    token_limit: int,
) -> tuple[LongTextPart, ...]:
    """Partition Markdown losslessly beneath a positive token ceiling."""

    if token_limit < 1:
        raise ValueError("token_limit must be positive")
    if not markdown:
        raise LongTextPlanningError("canonical Markdown is empty")
    if token_count(markdown, tokenizer) <= token_limit:
        return ()

    planned_text: list[str] = []
    current = ""
    for block in _markdown_blocks(markdown):
        block_tokens = token_count(block, tokenizer)
        if block_tokens > token_limit:
            if current:
                planned_text.append(current)
                current = ""
            planned_text.extend(
                _split_oversized_block(block, tokenizer, token_limit=token_limit)
            )
            continue
        candidate = current + block
        if current and token_count(candidate, tokenizer) > token_limit:
            planned_text.append(current)
            current = block
        else:
            current = candidate
    if current:
        planned_text.append(current)

    if "".join(planned_text) != markdown or any(not part for part in planned_text):
        raise LongTextPlanningError("long-text plan does not cover canonical Markdown")
    parts = tuple(
        LongTextPart(
            index=index,
            text=part,
            input_sha256=hashlib.sha256(part.encode("utf-8")).hexdigest(),
            token_count=token_count(part, tokenizer),
        )
        for index, part in enumerate(planned_text)
    )
    if any(part.token_count > token_limit for part in parts):
        raise LongTextPlanningError("long-text part exceeds configured token ceiling")
    return parts


def _markdown_blocks(markdown: str) -> tuple[str, ...]:
    blocks: list[str] = []
    start = 0
    for match in _BLOCK_SEPARATOR.finditer(markdown):
        blocks.append(markdown[start : match.end()])
        start = match.end()
    if start < len(markdown):
        blocks.append(markdown[start:])
    return tuple(blocks)


def _split_oversized_block(
    block: str,
    tokenizer: TextOffsetTokenizer,
    *,
    token_limit: int,
) -> tuple[str, ...]:
    parts: list[str] = []
    remaining = block
    while token_count(remaining, tokenizer) > token_limit:
        offsets = _validated_offsets(remaining, tokenizer.token_offsets(remaining))
        boundary_index = min(token_limit, len(offsets)) - 1
        boundary = 0
        while boundary_index >= 0:
            candidate = offsets[boundary_index][1]
            next_start = (
                offsets[boundary_index + 1][0]
                if boundary_index + 1 < len(offsets)
                else len(remaining)
            )
            if (
                next_start >= candidate
                and token_count(remaining[:candidate], tokenizer) <= token_limit
            ):
                boundary = candidate
                break
            boundary_index -= 1
        if boundary_index < 0 or boundary <= 0:
            raise LongTextPlanningError("token boundary cannot satisfy token ceiling")
        parts.append(remaining[:boundary])
        remaining = remaining[boundary:]
    if remaining:
        parts.append(remaining)
    if "".join(parts) != block:
        raise LongTextPlanningError("oversized block split is not reversible")
    return tuple(parts)


def _validated_offsets(
    text: str,
    offsets: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    if not text:
        if offsets:
            raise LongTextPlanningError("tokenizer offsets do not cover input exactly")
        return offsets
    covered_end = 0
    previous_start = -1
    previous_end = -1
    for offset in offsets:
        if (
            len(offset) != 2
            or isinstance(offset[0], bool)
            or isinstance(offset[1], bool)
            or not isinstance(offset[0], int)
            or not isinstance(offset[1], int)
        ):
            raise LongTextPlanningError("tokenizer returned malformed offsets")
        start, end = offset
        if (
            start < 0
            or end <= start
            or end > len(text)
            or start < previous_start
            or end < previous_end
            or start > covered_end
        ):
            raise LongTextPlanningError("tokenizer offsets do not cover input exactly")
        previous_start = start
        previous_end = end
        covered_end = max(covered_end, end)
    if not offsets or covered_end != len(text):
        raise LongTextPlanningError("tokenizer offsets do not cover input exactly")
    return offsets
