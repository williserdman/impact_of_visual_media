"""Read-only preprocessing consumption and article-text embedding publication."""

from __future__ import annotations

import hashlib
import math
import os
import stat
import struct
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from uuid import uuid4

import duckdb

from wsj_embeddings.adapters import EmbeddingAdapter
from wsj_embeddings.canonical_markdown import (
    CanonicalMarkdownError,
    read_canonical_markdown,
)
from wsj_embeddings.catalog import EmbeddingCatalog, configuration_id
from wsj_embeddings.config import EmbeddingPipelineConfig
from wsj_embeddings.models import (
    CanonicalArticle,
    EmbeddingInventoryResult,
    EmbeddingRunResult,
)
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


class EmbeddingOutputRootError(RuntimeError):
    """Raised before mutation when the embedding output root is unsafe."""


class HostedProcessingAuthorizationError(RuntimeError):
    """Raised before licensed Markdown can reach a hosted adapter."""

    code = "hosted_processing_not_authorized"

    def __init__(self) -> None:
        super().__init__("hosted processing is not authorized")


_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
_LOCK_CREATE_FLAGS = (
    os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
)


def _open_embedding_output_root(output_root: Path) -> int:
    """Create and anchor the output root without following path components."""

    absolute_root = Path(os.path.abspath(output_root))
    if absolute_root == Path(absolute_root.anchor) or not absolute_root.name:
        raise EmbeddingOutputRootError("unsafe embedding output root")
    try:
        descriptor = os.open(absolute_root.anchor, _DIRECTORY_FLAGS)
    except OSError as error:
        raise EmbeddingOutputRootError("unsafe embedding output root") from error
    try:
        for component in absolute_root.parts[1:-1]:
            next_descriptor = os.open(
                component,
                _DIRECTORY_FLAGS,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        with suppress(FileExistsError):
            os.mkdir(absolute_root.name, mode=0o700, dir_fd=descriptor)
        output_descriptor = os.open(
            absolute_root.name,
            _DIRECTORY_FLAGS,
            dir_fd=descriptor,
        )
        output_stat = os.fstat(output_descriptor)
        path_stat = os.stat(
            absolute_root.name,
            dir_fd=descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(path_stat.st_mode)
            or (path_stat.st_dev, path_stat.st_ino)
            != (output_stat.st_dev, output_stat.st_ino)
        ):
            os.close(output_descriptor)
            raise EmbeddingOutputRootError("unsafe embedding output root")
        return output_descriptor
    except EmbeddingOutputRootError:
        raise
    except OSError as error:
        raise EmbeddingOutputRootError("unsafe embedding output root") from error
    finally:
        os.close(descriptor)


@contextmanager
def embedding_pipeline_lock(lock_path: Path) -> Iterator[int]:
    """Own one exclusive lock before mutating an embedding output root."""

    output_descriptor = _open_embedding_output_root(lock_path.parent)
    try:
        try:
            lock_descriptor = os.open(
                lock_path.name,
                _LOCK_CREATE_FLAGS,
                0o600,
                dir_fd=output_descriptor,
            )
        except FileExistsError as error:
            raise EmbeddingPipelineLockedError(
                f"embedding pipeline is already running; lock exists at {lock_path}"
            ) from error
        lock_stat = os.fstat(lock_descriptor)
        try:
            if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1:
                raise EmbeddingOutputRootError("unsafe embedding pipeline lock")
            os.write(lock_descriptor, f"{os.getpid()}\n".encode())
            yield output_descriptor
        finally:
            try:
                current_stat = os.stat(
                    lock_path.name,
                    dir_fd=output_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                if (
                    stat.S_ISREG(current_stat.st_mode)
                    and (current_stat.st_dev, current_stat.st_ino)
                    == (lock_stat.st_dev, lock_stat.st_ino)
                ):
                    os.unlink(lock_path.name, dir_fd=output_descriptor)
            os.close(lock_descriptor)
    finally:
        os.close(output_descriptor)


def run_embedding_pipeline(
    config: EmbeddingPipelineConfig,
    adapter: EmbeddingAdapter,
    *,
    limit: int,
) -> EmbeddingRunResult:
    """Publish normalized article-text vectors for a bounded canonical slice."""

    if not config.hosted_processing_authorized:
        raise HostedProcessingAuthorizationError
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
    configuration_identifier = configuration_id(profile)
    with (
        embedding_pipeline_lock(config.embedding_lock_path) as output_descriptor,
        EmbeddingCatalog.open_in(output_descriptor) as catalog,
        catalog.transaction(),
    ):
        if not prepared:
            return EmbeddingRunResult(articles=0, embeddings=0)
        catalog.connection.execute(
            """
            INSERT INTO embedding_configurations
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (configuration_id) DO NOTHING
            """,
            [
                configuration_identifier,
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
                    article.article_id,
                    _ARTICLE_TEXT_MODALITY,
                    configuration_identifier,
                    article.published_at_utc,
                    article.publication_date_new_york,
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
            [
                str(uuid4()),
                configuration_identifier,
                len(prepared),
                len(prepared),
            ],
        )
    return EmbeddingRunResult(articles=len(prepared), embeddings=len(prepared))


def inventory_embedding_articles(
    config: EmbeddingPipelineConfig,
) -> EmbeddingInventoryResult:
    """Count canonical text/header eligibility without constructing an adapter."""

    config.validate()
    try:
        with Catalog.read_only(config.preprocessing_catalog) as db:
            Catalog.validate_schema(db)
            row = db.execute(
                """
                SELECT count(*) AS canonical_articles,
                       count(cleaned_markdown_path) AS eligible_text,
                       count(header_image_path) AS header_present,
                       count(*) - count(header_image_path) AS header_absent
                FROM articles
                """
            ).fetchone()
    except (CatalogError, duckdb.Error, OSError) as error:
        raise EmbeddingPipelineError("preprocessing_catalog", "unknown") from error
    assert row is not None
    return EmbeddingInventoryResult(*(int(value) for value in row))


def _read_articles(
    config: EmbeddingPipelineConfig,
    limit: int,
) -> tuple[CanonicalArticle, ...]:
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
    return tuple(CanonicalArticle(*row) for row in rows)


def _prepare_embedding(
    config: EmbeddingPipelineConfig,
    adapter: EmbeddingAdapter,
    article: CanonicalArticle,
) -> tuple[CanonicalArticle, str, str, tuple[float, ...]]:
    try:
        markdown_bytes = read_canonical_markdown(
            config.preprocessing_output_root,
            article.cleaned_markdown_path,
        )
    except CanonicalMarkdownError as error:
        code = "read_markdown" if error.status == "missing" else "unsafe_markdown_path"
        raise EmbeddingPipelineError(code, article.article_id) from error
    try:
        markdown = markdown_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise EmbeddingPipelineError("invalid_utf8", article.article_id) from error
    embedded = adapter.embed_text(markdown)
    vector = _normalized_vector(embedded, article.article_id)
    packed = b"".join(struct.pack("<f", value) for value in vector)
    return article, hashlib.sha256(markdown_bytes).hexdigest(), hashlib.sha256(
        packed
    ).hexdigest(), vector
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
