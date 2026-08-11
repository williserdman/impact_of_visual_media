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

from wsj_embeddings.adapters import EmbeddingAdapter, JinaHostedAdapterError
from wsj_embeddings.canonical_markdown import (
    CanonicalMarkdownError,
    read_canonical_markdown,
)
from wsj_embeddings.catalog import EmbeddingCatalog, configuration_id
from wsj_embeddings.config import EmbeddingPipelineConfig
from wsj_embeddings.models import (
    CanonicalArticle,
    EmbeddingInventoryResult,
    EmbeddingProfile,
    EmbeddingRunResult,
)
from wsj_pipeline.catalog import Catalog, CatalogError

_ARTICLE_TEXT_MODALITY = "article_text"
_VECTOR_DIMENSIONS = 2048
_QUEUED = "queued"
_IN_PROGRESS = "in_progress"
_SUCCEEDED = "succeeded"
_RETRYABLE = "retryable"
_TERMINAL = "terminal"
_INTERRUPTED = "interrupted"
_WORK_STATES = {
    _QUEUED,
    _IN_PROGRESS,
    _SUCCEEDED,
    _RETRYABLE,
    _TERMINAL,
    _INTERRUPTED,
}
_RUN_COUNT_COLUMNS = {
    "reused",
    "attempted",
    "succeeded",
    "retryable",
    "terminal",
    "interrupted",
}


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
    configuration_identifier = configuration_id(profile)
    run_id = str(uuid4())
    with (
        embedding_pipeline_lock(config.embedding_lock_path) as output_descriptor,
        EmbeddingCatalog.open_in(output_descriptor) as catalog,
    ):
        with catalog.transaction():
            _begin_run(
                catalog,
                run_id,
                configuration_identifier,
                profile,
                len(articles),
            )
        for article in articles:
            with catalog.transaction():
                _register_work(
                    catalog,
                    run_id,
                    configuration_identifier,
                    article,
                )
        for article in articles:
            state = _work_state(catalog, article, configuration_identifier)
            if state == _SUCCEEDED:
                with catalog.transaction():
                    _increment_run(catalog, run_id, "reused")
                continue
            if state == _TERMINAL:
                with catalog.transaction():
                    _increment_run(catalog, run_id, "terminal")
                continue
            with catalog.transaction():
                catalog.connection.execute(
                    """
                    UPDATE embedding_work_items
                    SET state = ?, attempt_count = attempt_count + 1,
                        error_code = NULL, status_code = NULL,
                        retry_after_seconds = NULL, last_run_id = ?,
                        updated_at = current_timestamp
                    WHERE article_id = ? AND modality = ?
                      AND configuration_id = ?
                    """,
                    [
                        _IN_PROGRESS,
                        run_id,
                        article.article_id,
                        _ARTICLE_TEXT_MODALITY,
                        configuration_identifier,
                    ],
                )
                _increment_run(catalog, run_id, "attempted")
            try:
                input_sha256, vector_hash, vector = _prepare_embedding(
                    config,
                    adapter,
                    article,
                )
            except JinaHostedAdapterError as error:
                failure_state = _RETRYABLE if error.retryable else _TERMINAL
                with catalog.transaction():
                    catalog.connection.execute(
                        """
                        UPDATE embedding_work_items
                        SET state = ?, error_code = ?, status_code = ?,
                            retry_after_seconds = ?, last_run_id = ?,
                            updated_at = current_timestamp
                        WHERE article_id = ? AND modality = ?
                          AND configuration_id = ?
                        """,
                        [
                            failure_state,
                            error.code,
                            error.status_code,
                            error.retry_after_seconds,
                            run_id,
                            article.article_id,
                            _ARTICLE_TEXT_MODALITY,
                            configuration_identifier,
                        ],
                    )
                    _increment_run(catalog, run_id, failure_state)
                raise
            with catalog.transaction():
                catalog.connection.execute(
                    """
                    INSERT INTO embeddings
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (article_id, modality, configuration_id)
                    DO UPDATE SET
                        published_at_utc = excluded.published_at_utc,
                        publication_date_new_york =
                            excluded.publication_date_new_york,
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
                    UPDATE embedding_work_items
                    SET state = ?, input_sha256 = ?, error_code = NULL,
                        status_code = NULL, retry_after_seconds = NULL,
                        last_run_id = ?, updated_at = current_timestamp
                    WHERE article_id = ? AND modality = ?
                      AND configuration_id = ?
                    """,
                    [
                        _SUCCEEDED,
                        input_sha256,
                        run_id,
                        article.article_id,
                        _ARTICLE_TEXT_MODALITY,
                        configuration_identifier,
                    ],
                )
                catalog.connection.execute(
                    """
                    UPDATE runs
                    SET embeddings = embeddings + 1,
                        succeeded = succeeded + 1
                    WHERE run_id = ?
                    """,
                    [run_id],
                )
        row = catalog.connection.execute(
            """
            SELECT articles, embeddings, reused, attempted, succeeded,
                   retryable, terminal, interrupted
            FROM runs
            WHERE run_id = ?
            """,
            [run_id],
        ).fetchone()
    assert row is not None
    return EmbeddingRunResult(*(int(value) for value in row))


def _begin_run(
    catalog: EmbeddingCatalog,
    run_id: str,
    configuration_identifier: str,
    profile: EmbeddingProfile,
    article_count: int,
) -> None:
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
    catalog.connection.execute(
        """
        INSERT INTO runs (
            run_id, configuration_id, articles, embeddings, reused,
            attempted, succeeded, retryable, terminal, interrupted, started_at
        )
        VALUES (?, ?, ?, 0, 0, 0, 0, 0, 0, 0, current_timestamp)
        """,
        [run_id, configuration_identifier, article_count],
    )


def _register_work(
    catalog: EmbeddingCatalog,
    run_id: str,
    configuration_identifier: str,
    article: CanonicalArticle,
) -> None:
    row = catalog.connection.execute(
        """
        SELECT state, input_sha256
        FROM embedding_work_items
        WHERE article_id = ? AND modality = ? AND configuration_id = ?
        """,
        [article.article_id, _ARTICLE_TEXT_MODALITY, configuration_identifier],
    ).fetchone()
    if row is None:
        catalog.connection.execute(
            """
            INSERT INTO embedding_work_items
            VALUES (?, ?, ?, ?, ?, 0, NULL, NULL, NULL, ?, current_timestamp)
            """,
            [
                article.article_id,
                _ARTICLE_TEXT_MODALITY,
                configuration_identifier,
                article.cleaned_markdown_sha256,
                _QUEUED,
                run_id,
            ],
        )
        return
    state, input_sha256 = row
    if state not in _WORK_STATES:
        raise EmbeddingPipelineError("invalid_work_state", article.article_id)
    if input_sha256 != article.cleaned_markdown_sha256:
        catalog.connection.execute(
            """
            DELETE FROM embeddings
            WHERE article_id = ? AND modality = ? AND configuration_id = ?
              AND input_sha256 <> ?
            """,
            [
                article.article_id,
                _ARTICLE_TEXT_MODALITY,
                configuration_identifier,
                article.cleaned_markdown_sha256,
            ],
        )
        catalog.connection.execute(
            """
            UPDATE embedding_work_items
            SET input_sha256 = ?, state = ?, attempt_count = 0,
                error_code = NULL, status_code = NULL,
                retry_after_seconds = NULL, last_run_id = ?,
                updated_at = current_timestamp
            WHERE article_id = ? AND modality = ? AND configuration_id = ?
            """,
            [
                article.cleaned_markdown_sha256,
                _QUEUED,
                run_id,
                article.article_id,
                _ARTICLE_TEXT_MODALITY,
                configuration_identifier,
            ],
        )
        return
    proven_success = state == _SUCCEEDED and catalog.connection.execute(
        """
        SELECT count(*)
        FROM embeddings
        WHERE article_id = ? AND modality = ? AND configuration_id = ?
          AND input_sha256 = ?
        """,
        [
            article.article_id,
            _ARTICLE_TEXT_MODALITY,
            configuration_identifier,
            article.cleaned_markdown_sha256,
        ],
    ).fetchone()[0]
    if state == _IN_PROGRESS or (state == _SUCCEEDED and not proven_success):
        catalog.connection.execute(
            """
            UPDATE embedding_work_items
            SET state = ?, last_run_id = ?, updated_at = current_timestamp
            WHERE article_id = ? AND modality = ? AND configuration_id = ?
            """,
            [
                _INTERRUPTED,
                run_id,
                article.article_id,
                _ARTICLE_TEXT_MODALITY,
                configuration_identifier,
            ],
        )
        _increment_run(catalog, run_id, "interrupted")


def _work_state(
    catalog: EmbeddingCatalog,
    article: CanonicalArticle,
    configuration_identifier: str,
) -> str:
    row = catalog.connection.execute(
        """
        SELECT state
        FROM embedding_work_items
        WHERE article_id = ? AND modality = ? AND configuration_id = ?
        """,
        [article.article_id, _ARTICLE_TEXT_MODALITY, configuration_identifier],
    ).fetchone()
    assert row is not None
    return str(row[0])


def _increment_run(catalog: EmbeddingCatalog, run_id: str, column: str) -> None:
    if column not in _RUN_COUNT_COLUMNS:
        raise AssertionError("unsupported run count")
    catalog.connection.execute(
        f"UPDATE runs SET {column} = {column} + 1 WHERE run_id = ?",
        [run_id],
    )


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
                       publication_date_new_york, cleaned_markdown_sha256
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
) -> tuple[str, str, tuple[float, ...]]:
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
    return (
        hashlib.sha256(markdown_bytes).hexdigest(),
        hashlib.sha256(packed).hexdigest(),
        vector,
    )


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
