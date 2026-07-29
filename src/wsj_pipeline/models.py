"""Shared immutable records for pipeline boundaries."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SourceCandidate:
    """One raw HTML source and its optional companion image."""

    relative_html_path: str
    html_size: int
    html_mtime_ns: int
    relative_image_path: str | None
    image_size: int | None
    image_mtime_ns: int | None
    warnings: tuple[str, ...] = ()
