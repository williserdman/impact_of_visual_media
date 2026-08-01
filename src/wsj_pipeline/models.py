"""Shared immutable records for pipeline boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True, slots=True)
class SourceCandidate:
    """One raw HTML source and its optional companion image."""

    relative_html_path: str
    html_size: int
    html_mtime_ns: int
    relative_header_image_path: str | None
    header_image_size: int | None
    header_image_mtime_ns: int | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PublicationDates:
    """UTC and New York representations of one publication event."""

    published_at_utc: datetime
    publication_date_utc: date
    published_at_new_york: datetime
    publication_date_new_york: date
    publication_year_ny: int
    publication_month_ny: int


@dataclass(frozen=True, slots=True)
class ExtractedArticle:
    """Normalized representation of one raw source snapshot."""

    article_id: str
    relative_html_path: str
    source_html_sha256: str
    extractor_version: str
    canonical_url: str | None
    published_at_utc: datetime
    publication_date_new_york: date
    updated_at_utc: datetime | None
    markdown_text: str
    editorial_character_count: int
    editorial_block_count: int
    metadata_completeness: int
    relative_header_image_path: str | None
    inline_image_urls: tuple[str, ...]
    warnings: tuple[str, ...]
