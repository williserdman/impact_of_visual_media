"""Immutable records at the embedding coordinator boundary."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True, slots=True)
class CanonicalArticle:
    """Canonical metadata consumed by embedding publication and validation."""

    article_id: str
    cleaned_markdown_path: str
    published_at_utc: datetime
    publication_date_new_york: date


@dataclass(frozen=True, slots=True)
class EmbeddingProfile:
    """Identity and output contract for one embedding configuration."""

    model: str
    task: str
    dimensions: int
    output_type: str = "float"
    normalization: str = "l2-client-v1"


@dataclass(frozen=True, slots=True)
class EmbeddingRunResult:
    """Content-free counts published by one embedding run."""

    articles: int
    embeddings: int
