"""Immutable records at the embedding coordinator boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum


class EmbeddingModality(StrEnum):
    """Research-facing vector modalities shared by coordination and validation."""

    ARTICLE_TEXT = "article_text"
    HEADER_IMAGE = "header_image"
    MULTIMODAL_ARTICLE = "multimodal_article"


class VectorDerivationKind(StrEnum):
    """Truthful origin of the bytes represented by one stored vector."""

    HOSTED_RESPONSE = "hosted_response"
    LONG_TEXT_AGGREGATE = "long_text_aggregate"
    MULTIMODAL_MIDPOINT = "multimodal_midpoint"
    SYNTHETIC_FIXTURE = "synthetic_fixture"


class WorkState(StrEnum):
    """Durable lifecycle vocabulary for one modality/configuration item."""

    ELIGIBLE = "eligible"
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    RETRYABLE = "retryable"
    TERMINAL = "terminal"
    INTERRUPTED = "interrupted"
    NOT_APPLICABLE = "not_applicable"
    STALE_INPUT = "stale_input"
    STALE_CONFIGURATION = "stale_configuration"

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
    observed_model: str | None = None
    output_type: str = "float"
    normalization: str = "l2-client-float32-v1"
    client_api_contract_version: str = "openapi-2026.07.27.1603"
    tokenizer_revision: str = (
        "jinaai/jina-embeddings-v4@"
        "d1e5d70b7b34d927a8cddac458583c4fbe50a914:tokenizer.json"
        "#sha256=9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa"
    )
    tokenizer_engine: str = "tokenizers-0.21.4"
    context_token_limit: int = 8_000
    context_rules: str = "markdown-block-greedy-no-overlap-truncate-false-v1"
    long_text_aggregation: str = "l2-token-count-weighted-mean-float32-v1"
    long_text_part_attempt_limit: int = 3
    batch_max_items: int = 16
    batch_max_estimated_tokens: int = 45_000
    batch_max_encoded_bytes: int = 8_000_000
    batch_max_response_bytes: int = 2_000_000
    batch_max_concurrency: int = 2
    batch_max_attempts: int = 3
    batch_initial_backoff_seconds: float = 1.0
    batch_max_backoff_seconds: float = 30.0
    quota_max_requests: int = 90
    quota_max_estimated_tokens: int = 90_000
    quota_window_seconds: int = 60
    image_input_rules: str = (
        "static-jpeg-png-webp-max5000000b-max20000000px-decode-max40000000px-"
        "exact-source-v1"
    )
    image_transform: str = (
        "pillow-11.3.0-exif-transpose-alpha-white-rgb-jpeg-q85-420-lanczos-"
        "if-over20000000px-optimize0-progressive0-metadata-none-v1"
    )
    multimodal_formula: str = "l2-normalize-0.5-text-0.5-image-v1"
    client_configuration_version: str = "wsj-embeddings-config-v6"


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
    hosted_requests: int = 0
    retries: int = 0
    usage: dict[str, int | float] = field(default_factory=dict)
    throttles: int = 0
    elapsed_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class EmbeddingInventoryResult:
    """Content-free eligibility counts from the canonical article catalog."""

    canonical_articles: int
    eligible_text: int
    header_present: int
    header_absent: int
