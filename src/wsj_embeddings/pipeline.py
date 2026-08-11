"""Read-only preprocessing consumption and multimodal embedding publication."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import stat
import struct
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
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
    EmbeddingModality,
    EmbeddingProfile,
    EmbeddingRunResult,
    WorkState,
)
from wsj_embeddings.source_image import SourceImageError, read_source_image
from wsj_pipeline.catalog import Catalog, CatalogError

_ARTICLE_TEXT_MODALITY = EmbeddingModality.ARTICLE_TEXT.value
_HEADER_IMAGE_MODALITY = EmbeddingModality.HEADER_IMAGE.value
_MULTIMODAL_ARTICLE_MODALITY = EmbeddingModality.MULTIMODAL_ARTICLE.value
_VECTOR_DIMENSIONS = 2048
_DETERMINISTIC_IMAGE_ATTEMPT_LIMIT = 3
_MULTIMODAL_FORMULA = "l2-normalize-0.5-text-0.5-image-v1"


@dataclass(frozen=True, slots=True)
class _PreparedHeaderImage:
    source_relative_path: str
    input_sha256: str
    data: bytes


@dataclass(frozen=True, slots=True)
class _MissingHeaderImage:
    source_relative_path: str


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


def _ensure_source_root(source_root: Path) -> None:
    """Reject an unavailable archive root before any item or output work."""

    try:
        descriptor = os.open(source_root, _DIRECTORY_FLAGS)
    except OSError as error:
        raise EmbeddingPipelineError("source_root", "unknown") from error
    else:
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
    reprocess: bool = False,
) -> EmbeddingRunResult:
    """Publish source and derived vectors for a bounded canonical slice."""

    if not config.hosted_processing_authorized:
        raise HostedProcessingAuthorizationError
    if limit < 1:
        raise ValueError("limit must be positive")
    if not isinstance(reprocess, bool):
        raise ValueError("reprocess must be boolean")
    config.validate()
    _ensure_source_root(config.source_root)
    profile = adapter.profile
    if profile.dimensions != _VECTOR_DIMENSIONS:
        raise ValueError("adapter profile dimensions must be 2048")
    articles = _read_articles(config, limit)
    prepared_images = {
        article.article_id: _prepare_header_image(config, article)
        for article in articles
    }
    configuration_identifier = configuration_id(profile)
    run_id = str(uuid4())
    with (
        embedding_pipeline_lock(config.embedding_lock_path) as output_descriptor,
        EmbeddingCatalog.open_in(output_descriptor) as catalog,
    ):
        with catalog.transaction():
            catalog.begin_run(
                run_id, configuration_identifier, profile, len(articles)
            )
        registered_states: dict[tuple[str, str], WorkState] = {}
        for article in articles:
            with catalog.transaction():
                registered_states[(article.article_id, _ARTICLE_TEXT_MODALITY)] = (
                    _register_work(
                        catalog,
                        run_id,
                        configuration_identifier,
                        article,
                        reprocess=reprocess,
                    )
                )
            with catalog.transaction():
                registered_states[(article.article_id, _HEADER_IMAGE_MODALITY)] = (
                    _register_header_work(
                        catalog,
                        run_id,
                        configuration_identifier,
                        article,
                        prepared_images[article.article_id],
                    )
                )
        for article in articles:
            for modality, prepared_image in (
                (_ARTICLE_TEXT_MODALITY, None),
                (_HEADER_IMAGE_MODALITY, prepared_images[article.article_id]),
            ):
                state = registered_states[(article.article_id, modality)]
                if state is WorkState.NOT_APPLICABLE:
                    continue
                if state.is_reusable_success:
                    with catalog.transaction():
                        catalog.increment_run(run_id, "reused")
                    continue
                if state is WorkState.TERMINAL:
                    continue
                if isinstance(prepared_image, _MissingHeaderImage):
                    continue
                with catalog.transaction():
                    catalog.start_attempt(
                        run_id=run_id,
                        article_id=article.article_id,
                        modality=modality,
                        configuration_identifier=configuration_identifier,
                    )
                try:
                    input_sha256, vector_hash, vector = _prepare_embedding(
                        config,
                        adapter,
                        article,
                        modality=modality,
                        prepared_image=prepared_image,
                    )
                except JinaHostedAdapterError as error:
                    with catalog.transaction():
                        attempt_count = catalog.attempt_count(
                            article_id=article.article_id,
                            modality=modality,
                            configuration_identifier=configuration_identifier,
                        )
                        bounded_deterministic_image = (
                            modality == _HEADER_IMAGE_MODALITY
                            and error.code == "deterministic_request"
                            and attempt_count < _DETERMINISTIC_IMAGE_ATTEMPT_LIMIT
                        )
                        failure_state = (
                            WorkState.RETRYABLE
                            if error.retryable or bounded_deterministic_image
                            else WorkState.TERMINAL
                        )
                        catalog.record_failure(
                            run_id=run_id,
                            article_id=article.article_id,
                            modality=modality,
                            configuration_identifier=configuration_identifier,
                            state=failure_state,
                            error_code=error.code,
                            status_code=error.status_code,
                            retry_after_seconds=error.retry_after_seconds,
                        )
                        if modality == _HEADER_IMAGE_MODALITY:
                            catalog.increment_run(run_id, "header_failed")
                    if modality == _HEADER_IMAGE_MODALITY:
                        continue
                    raise
                source_relative_path = (
                    prepared_image.source_relative_path
                    if prepared_image is not None
                    else None
                )
                with catalog.transaction():
                    catalog.publish_success(
                        run_id=run_id,
                        article_id=article.article_id,
                        modality=modality,
                        configuration_identifier=configuration_identifier,
                        source_relative_path=source_relative_path,
                        published_at_utc=article.published_at_utc,
                        publication_date_new_york=article.publication_date_new_york,
                        dimensions=profile.dimensions,
                        input_sha256=input_sha256,
                        stored_vector_sha256=vector_hash,
                        vector=vector,
                    )
        for article in articles:
            with catalog.transaction():
                _publish_multimodal_article(
                    catalog,
                    run_id,
                    configuration_identifier,
                    profile,
                    article,
                )
        result = catalog.run_result(run_id)
    return result


def _register_work(
    catalog: EmbeddingCatalog,
    run_id: str,
    configuration_identifier: str,
    article: CanonicalArticle,
    *,
    reprocess: bool,
) -> WorkState:
    try:
        return catalog.register_work(
            run_id=run_id,
            article_id=article.article_id,
            modality=_ARTICLE_TEXT_MODALITY,
            configuration_identifier=configuration_identifier,
            source_relative_path=None,
            input_sha256=article.cleaned_markdown_sha256,
            published_at_utc=article.published_at_utc,
            publication_date_new_york=article.publication_date_new_york,
            reprocess=reprocess,
        )
    except ValueError as error:
        raise EmbeddingPipelineError(
            "invalid_work_state", article.article_id
        ) from error


def _register_header_work(
    catalog: EmbeddingCatalog,
    run_id: str,
    configuration_identifier: str,
    article: CanonicalArticle,
    prepared_image: _PreparedHeaderImage | _MissingHeaderImage | None,
) -> WorkState:
    try:
        if prepared_image is None:
            return catalog.register_not_applicable(
                run_id=run_id,
                article_id=article.article_id,
                modality=_HEADER_IMAGE_MODALITY,
                configuration_identifier=configuration_identifier,
            )
        if isinstance(prepared_image, _MissingHeaderImage):
            return catalog.register_missing_header(
                run_id=run_id,
                article_id=article.article_id,
                configuration_identifier=configuration_identifier,
                source_relative_path=prepared_image.source_relative_path,
            )
        return catalog.register_work(
            run_id=run_id,
            article_id=article.article_id,
            modality=_HEADER_IMAGE_MODALITY,
            configuration_identifier=configuration_identifier,
            source_relative_path=prepared_image.source_relative_path,
            input_sha256=prepared_image.input_sha256,
            published_at_utc=article.published_at_utc,
            publication_date_new_york=article.publication_date_new_york,
            reprocess=False,
        )
    except ValueError as error:
        raise EmbeddingPipelineError(
            "invalid_work_state", article.article_id
        ) from error


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
                       publication_date_new_york, cleaned_markdown_sha256,
                       header_image_path
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
    *,
    modality: str,
    prepared_image: _PreparedHeaderImage | _MissingHeaderImage | None,
) -> tuple[str, str, tuple[float, ...]]:
    if modality == _HEADER_IMAGE_MODALITY:
        assert isinstance(prepared_image, _PreparedHeaderImage)
        embedded = adapter.embed_image(
            base64.b64encode(prepared_image.data).decode("ascii")
        )
        vector = _normalized_vector(embedded, article.article_id)
        return (
            prepared_image.input_sha256,
            _vector_hash(vector),
            vector,
        )
    assert modality == _ARTICLE_TEXT_MODALITY
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
    input_sha256 = hashlib.sha256(markdown_bytes).hexdigest()
    if input_sha256 != article.cleaned_markdown_sha256:
        raise EmbeddingPipelineError(
            "canonical_input_hash_mismatch", article.article_id
        )
    embedded = adapter.embed_text(markdown)
    vector = _normalized_vector(embedded, article.article_id)
    return (
        input_sha256,
        _vector_hash(vector),
        vector,
    )


def _prepare_header_image(
    config: EmbeddingPipelineConfig,
    article: CanonicalArticle,
) -> _PreparedHeaderImage | _MissingHeaderImage | None:
    if article.header_image_path is None:
        return None
    try:
        image_bytes = read_source_image(config.source_root, article.header_image_path)
    except SourceImageError as error:
        if error.status == "missing":
            return _MissingHeaderImage(article.header_image_path)
        code = (
            "missing_header_image"
            if error.status == "missing"
            else "unsafe_header_image_path"
        )
        raise EmbeddingPipelineError(code, article.article_id) from error
    return _PreparedHeaderImage(
        source_relative_path=article.header_image_path,
        input_sha256=hashlib.sha256(image_bytes).hexdigest(),
        data=image_bytes,
    )


def _publish_multimodal_article(
    catalog: EmbeddingCatalog,
    run_id: str,
    configuration_identifier: str,
    profile: EmbeddingProfile,
    article: CanonicalArticle,
) -> None:
    """Publish the derived vector only from two current successful sources."""

    sources = catalog.multimodal_sources(
        article_id=article.article_id,
        configuration_identifier=configuration_identifier,
    )
    if sources is None:
        return
    formula_version = str(profile.multimodal_formula)
    if formula_version != _MULTIMODAL_FORMULA:
        raise EmbeddingPipelineError(
            "unsupported_multimodal_formula",
            article.article_id,
        )
    (
        text_generation_run_id,
        text_vector_sha256,
        text_values,
        image_generation_run_id,
        image_vector_sha256,
        image_values,
    ) = sources
    text_vector = _verified_source_vector(
        text_values,
        str(text_vector_sha256),
        article.article_id,
    )
    image_vector = _verified_source_vector(
        image_values,
        str(image_vector_sha256),
        article.article_id,
    )
    vector = _multimodal_vector(text_vector, image_vector, article.article_id)
    provenance_payload = json.dumps(
        {
            "formula_version": formula_version,
            "header_image_generation_run_id": str(image_generation_run_id),
            "header_image_stored_vector_sha256": str(image_vector_sha256),
            "text_generation_run_id": str(text_generation_run_id),
            "text_stored_vector_sha256": str(text_vector_sha256),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    input_sha256 = hashlib.sha256(provenance_payload).hexdigest()
    try:
        state = catalog.register_work(
            run_id=run_id,
            article_id=article.article_id,
            modality=_MULTIMODAL_ARTICLE_MODALITY,
            configuration_identifier=configuration_identifier,
            source_relative_path=None,
            input_sha256=input_sha256,
            published_at_utc=article.published_at_utc,
            publication_date_new_york=article.publication_date_new_york,
            reprocess=False,
        )
    except ValueError as error:
        raise EmbeddingPipelineError(
            "invalid_work_state", article.article_id
        ) from error
    if state.is_reusable_success:
        generation_row = catalog.connection.execute(
            """
            SELECT generation_run_id
            FROM embedding_work_items
            WHERE article_id = ? AND modality = ? AND configuration_id = ?
            """,
            [
                article.article_id,
                _MULTIMODAL_ARTICLE_MODALITY,
                configuration_identifier,
            ],
        ).fetchone()
        if generation_row is None or not catalog.multimodal_provenance_matches(
            article_id=article.article_id,
            configuration_identifier=configuration_identifier,
            generation_run_id=str(generation_row[0]),
            text_generation_run_id=str(text_generation_run_id),
            text_stored_vector_sha256=str(text_vector_sha256),
            header_image_generation_run_id=str(image_generation_run_id),
            header_image_stored_vector_sha256=str(image_vector_sha256),
            formula_version=formula_version,
        ):
            raise EmbeddingPipelineError(
                "invalid_multimodal_provenance", article.article_id
            )
        catalog.increment_run(run_id, "reused")
        return
    catalog.publish_multimodal_success(
        run_id=run_id,
        article_id=article.article_id,
        configuration_identifier=configuration_identifier,
        published_at_utc=article.published_at_utc,
        publication_date_new_york=article.publication_date_new_york,
        dimensions=_VECTOR_DIMENSIONS,
        input_sha256=input_sha256,
        stored_vector_sha256=_vector_hash(vector),
        vector=vector,
        text_generation_run_id=str(text_generation_run_id),
        text_stored_vector_sha256=str(text_vector_sha256),
        header_image_generation_run_id=str(image_generation_run_id),
        header_image_stored_vector_sha256=str(image_vector_sha256),
        formula_version=formula_version,
    )


def _verified_source_vector(
    values: Sequence[float],
    expected_hash: str,
    article_id: str,
) -> tuple[float, ...]:
    """Reject a persisted source that is not the exact normalized generation."""

    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise EmbeddingPipelineError("invalid_source_vector", article_id) from error
    if (
        len(vector) != _VECTOR_DIMENSIONS
        or not all(math.isfinite(value) for value in vector)
        or _vector_hash(vector) != expected_hash
    ):
        raise EmbeddingPipelineError("invalid_source_vector", article_id)
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise EmbeddingPipelineError("invalid_source_vector", article_id)
    return vector


def _multimodal_vector(
    text_vector: tuple[float, ...],
    image_vector: tuple[float, ...],
    article_id: str,
) -> tuple[float, ...]:
    """Return the float32 L2-normalized equal midpoint of normalized sources."""

    normalized_text = _normalized_vector(text_vector, article_id)
    normalized_image = _normalized_vector(image_vector, article_id)
    midpoint = tuple(
        0.5 * text_value + 0.5 * image_value
        for text_value, image_value in zip(
            normalized_text,
            normalized_image,
            strict=True,
        )
    )
    return _normalized_vector(midpoint, article_id)


def _vector_hash(vector: tuple[float, ...]) -> str:
    packed = b"".join(struct.pack("<f", value) for value in vector)
    return hashlib.sha256(packed).hexdigest()


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
