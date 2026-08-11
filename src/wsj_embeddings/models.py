"""Immutable records at the embedding coordinator boundary."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum


class EmbeddingModality(StrEnum):
    """Research-facing vector modalities shared by coordination and validation."""

    ARTICLE_TEXT = "article_text"
    HEADER_IMAGE = "header_image"
    MULTIMODAL_ARTICLE = "multimodal_article"


class WorkState(StrEnum):
    """Durable lifecycle vocabulary for one modality/configuration item."""

    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    RETRYABLE = "retryable"
    TERMINAL = "terminal"
    INTERRUPTED = "interrupted"
    NOT_APPLICABLE = "not_applicable"

    @property
    def is_failure(self) -> bool:
        return self in {WorkState.RETRYABLE, WorkState.TERMINAL}

    @property
    def is_reusable_success(self) -> bool:
        return self is WorkState.SUCCEEDED


class SupersessionReason(StrEnum):
    """Stable causes for retaining superseded vector provenance."""

    INPUT_CHANGED = "input_changed"
    EXPLICIT_REPROCESS = "explicit_reprocess"


@dataclass(frozen=True, slots=True)
class CanonicalArticle:
    """Canonical metadata consumed by embedding publication and validation."""

    article_id: str
    cleaned_markdown_path: str
    published_at_utc: datetime
    publication_date_new_york: date
    cleaned_markdown_sha256: str
    header_image_path: str | None


@dataclass(frozen=True, slots=True)
class EmbeddingProfile:
    """Identity and output contract for one embedding configuration."""

    model: str
    task: str
    dimensions: int
    output_type: str = "float"
    normalization: str = "l2-client-float32-v1"
    observed_model: str = "jina-embeddings-v4"
    observed_api_version: str = "2026.07.27.1603"
    tokenizer_revision: str = "jinaai/jina-embeddings-v4-hosted-2026-08-11"
    context_token_limit: int = 32_768
    context_rules: str = "complete-input-truncate-false-late-chunking-false-v1"
    long_text_aggregation: str = "single-input-provider-pooling-v1"
    image_input_rules: str = "base64-source-bytes-no-remote-v1"
    image_transform: str = "source-bytes-no-transform-v1"
    multimodal_formula: str = "l2-normalize-0.5-text-0.5-image-v1"
    client_configuration_version: str = "wsj-embeddings-config-v1"


@dataclass(frozen=True, slots=True)
class EmbeddingRunResult:
    """Content-free counts published by one embedding run."""

    articles: int
    embeddings: int
    reused: int = 0
    attempted: int = 0
    succeeded: int = 0
    retryable: int = 0
    terminal: int = 0
    interrupted: int = 0
    header_absent: int = 0
    header_failed: int = 0


@dataclass(frozen=True, slots=True)
class EmbeddingInventoryResult:
    """Content-free eligibility counts from the canonical article catalog."""

    canonical_articles: int
    eligible_text: int
    header_present: int
    header_absent: int
