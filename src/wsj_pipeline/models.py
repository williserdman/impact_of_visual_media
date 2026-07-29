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
    relative_image_path: str | None
    image_size: int | None
    image_mtime_ns: int | None
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

    relative_html_path: str
    source_html_sha256: str
    extractor_version: str
    wsj_article_id: str | None
    canonical_url: str | None
    headline: str | None
    original_headline: str | None
    summary: str | None
    authors: tuple[str, ...]
    keywords: tuple[str, ...]
    section: str | None
    page: str | None
    article_type: str | None
    access: str | None
    published_at_utc: datetime
    publication_date_utc: date
    published_at_new_york: datetime
    publication_date_new_york: date
    publication_year_ny: int
    publication_month_ny: int
    updated_at_utc: datetime | None
    created_at_utc: datetime | None
    last_published_at_utc: datetime | None
    body_paragraphs: tuple[str, ...]
    body_text: str
    body_word_count: int
    relative_image_path: str | None
    image_present: bool
    image_media_type: str | None
    image_size_bytes: int | None
    image_caption: str | None
    image_credit: str | None
    image_creator: str | None
    image_alt: str | None
    image_width: int | None
    image_height: int | None
    metadata_completeness: int
    warnings: tuple[str, ...]
    raw_metadata_json: str
