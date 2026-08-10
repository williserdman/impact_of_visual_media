"""Read-only preprocessing consumption and article-text embedding publication."""

from __future__ import annotations

import hashlib
import math
import os
import struct
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import date, datetime
from pathlib import Path
from uuid import uuid4

import duckdb

from wsj_embeddings.adapters import EmbeddingAdapter
from wsj_embeddings.catalog import EmbeddingCatalog, configuration_id
from wsj_embeddings.config import EmbeddingPipelineConfig
from wsj_embeddings.models import EmbeddingRunResult
from wsj_pipeline.catalog import Catalog, CatalogError

_ARTICLE_TEXT_MODALITY = "article_text"
_VECTOR_DIMENSIONS = 2048


class EmbeddingPipelineError(RuntimeError):
    """Content-free coordinator failure with a stable machine-readable code."""

    def __init__(self, code: str, article_id: str) -> None:
        self.code = code
        self.article_id = article_id
        super().__init__(f"[{code}] article_id={article_id}")


class EmbeddingPipelineLockedError(RuntimeError):
    """Raised when another embedding coordinator owns the output lock."""


@contextmanager
def embedding_pipeline_lock(lock_path: Path) -> Iterator[None]:
    """Own one exclusive lock before mutating an embedding output root."""

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as error:
        raise EmbeddingPipelineLockedError(
            f"embedding pipeline is already running; lock exists at {lock_path}"
        ) from error
    try:
        try:
            os.write(descriptor, f"{os.getpid()}\n".encode())
        finally:
            os.close(descriptor)
        yield
    finally:
        with suppress(FileNotFoundError):
            lock_path.unlink()


def run_embedding_pipeline(
    config: EmbeddingPipelineConfig,
    adapter: EmbeddingAdapter,
    *,
    limit: int,
) -> EmbeddingRunResult:
    """Publish normalized article-text vectors for a bounded canonical slice."""

    if limit < 1:
        raise ValueError("limit must be positive")
    config.validate()
    profile = adapter.profile
    if profile.dimensions != _VECTOR_DIMENSIONS:
        raise ValueError("adapter profile dimensions must be 2048")
    articles = _read_articles(config, limit)
    prepared = tuple(
        _prepare_embedding(config, adapter, article) for article in articles
    )
    profile_id = configuration_id(profile)
    with embedding_pipeline_lock(config.embedding_lock_path), EmbeddingCatalog.open(
        config.embedding_catalog
    ) as catalog, catalog.transaction():
        if not prepared:
            return EmbeddingRunResult(articles=0, embeddings=0)
        catalog.connection.execute(
            """
            INSERT INTO embedding_configurations
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (configuration_id) DO NOTHING
            """,
            [
                profile_id,
                profile.model,
                profile.task,
                profile.dimensions,
                profile.output_type,
                profile.normalization,
            ],
        )
        for article, input_sha256, vector_hash, vector in prepared:
            catalog.connection.execute(
                """
                INSERT INTO embeddings
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (article_id, modality, configuration_id)
                DO UPDATE SET
                    published_at_utc = excluded.published_at_utc,
                    publication_date_new_york = excluded.publication_date_new_york,
                    dimensions = excluded.dimensions,
                    input_sha256 = excluded.input_sha256,
                    stored_vector_sha256 = excluded.stored_vector_sha256,
                    vector = excluded.vector
                """,
                [
                    article[0],
                    _ARTICLE_TEXT_MODALITY,
                    profile_id,
                    article[2],
                    article[3],
                    profile.dimensions,
                    input_sha256,
                    vector_hash,
                    list(vector),
                ],
            )
        catalog.connection.execute(
            """
            INSERT INTO runs VALUES (?, ?, ?, ?, current_timestamp)
            """,
            [str(uuid4()), profile_id, len(prepared), len(prepared)],
        )
    return EmbeddingRunResult(articles=len(prepared), embeddings=len(prepared))


def _read_articles(
    config: EmbeddingPipelineConfig,
    limit: int,
) -> tuple[tuple[str, str, datetime, date], ...]:
    try:
        with Catalog.read_only(config.preprocessing_catalog) as db:
            Catalog.validate_schema(db)
            rows = db.execute(
                """
                SELECT article_id, cleaned_markdown_path, published_at_utc,
                       publication_date_new_york
                FROM articles
                ORDER BY article_id
                LIMIT ?
                """,
                [limit],
            ).fetchall()
    except (CatalogError, duckdb.Error, OSError) as error:
        raise EmbeddingPipelineError("preprocessing_catalog", "unknown") from error
    return tuple(rows)


def _prepare_embedding(
    config: EmbeddingPipelineConfig,
    adapter: EmbeddingAdapter,
    article: tuple[str, str, datetime, date],
) -> tuple[tuple[str, str, datetime, date], str, str, tuple[float, ...]]:
    article_id, relative_markdown_path, _published_at, _publication_date = article
    path = _resolve_markdown_path(
        config.preprocessing_output_root,
        article_id,
        relative_markdown_path,
    )
    try:
        markdown_bytes = path.read_bytes()
    except OSError as error:
        raise EmbeddingPipelineError("read_markdown", article_id) from error
    try:
        markdown = markdown_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise EmbeddingPipelineError("invalid_utf8", article_id) from error
    try:
        embedded = adapter.embed_text(markdown)
    except Exception as error:
        raise EmbeddingPipelineError("embed_text", article_id) from error
    vector = _normalized_vector(embedded, article_id)
    packed = b"".join(struct.pack("<f", value) for value in vector)
    return article, hashlib.sha256(markdown_bytes).hexdigest(), hashlib.sha256(
        packed
    ).hexdigest(), vector


def _resolve_markdown_path(
    root: Path,
    article_id: str,
    relative_markdown_path: str,
) -> Path:
    candidate = Path(relative_markdown_path)
    if candidate.is_absolute():
        raise EmbeddingPipelineError("unsafe_markdown_path", article_id)
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise EmbeddingPipelineError("unsafe_markdown_path", article_id)
    return resolved


def _normalized_vector(
    values: tuple[float, ...],
    article_id: str,
) -> tuple[float, ...]:
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise EmbeddingPipelineError("invalid_vector", article_id) from error
    if len(vector) != _VECTOR_DIMENSIONS:
        raise EmbeddingPipelineError("invalid_dimensions", article_id)
    if not all(math.isfinite(value) for value in vector):
        raise EmbeddingPipelineError("nonfinite_vector", article_id)
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm == 0:
        raise EmbeddingPipelineError("zero_vector", article_id)
    normalized = tuple(value / norm for value in vector)
    return tuple(
        struct.unpack("<f", struct.pack("<f", value))[0] for value in normalized
    )
