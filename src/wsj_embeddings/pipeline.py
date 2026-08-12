"""Read-only preprocessing consumption and multimodal embedding publication."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import stat
import struct
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import duckdb

from wsj_embeddings.adapters import (
    EmbeddingAdapter,
    JinaBatchItemOutcome,
    JinaEmbeddingInput,
    JinaHostedAdapterError,
)
from wsj_embeddings.batching import (
    BatchExecutionResult,
    BatchPolicy,
    execute_rate_aware_batches,
)
from wsj_embeddings.canonical_markdown import (
    CanonicalMarkdownError,
    read_canonical_markdown,
)
from wsj_embeddings.catalog import EmbeddingCatalog, configuration_id
from wsj_embeddings.config import EmbeddingPipelineConfig
from wsj_embeddings.image_rendition import (
    ImageCodec,
    ImageCodecError,
    PillowImageCodec,
    PreparedImageInput,
    cleanup_orphan_image_renditions,
    install_image_rendition,
    prepare_image_input,
    verify_image_rendition,
)
from wsj_embeddings.long_text import (
    LongTextPart,
    LongTextPlanningError,
    plan_long_text_parts,
)
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
_TERMINAL_IMAGE_PREPARATION_CODES = frozenset(
    {
        "corrupt_image",
        "derived_image_oversized",
        "image_encode_failure",
        "unsafe_image",
        "unsupported_image",
    }
)


@dataclass(frozen=True, slots=True)
class _PreparedHeaderImage:
    source_relative_path: str
    input_sha256: str
    data: bytes
    image_input: PreparedImageInput | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingPipelineFailpoints:
    """Content-free interruption hooks for durable-boundary verification."""

    after_long_text_part_attempt_checkpoint: (
        Callable[[int, int], None] | None
    ) = None
    after_long_text_terminal_transition: Callable[[int], None] | None = None
    after_image_rendition_install: Callable[[str], None] | None = None
    after_batched_item_outcome_commit: Callable[[int, bool], None] | None = None
    after_full_discovery_page: Callable[[int], None] | None = None
    after_full_processing_page: Callable[[int], None] | None = None
    during_full_reconciliation_page: Callable[[int], None] | None = None
    before_full_rendition_cleanup: Callable[[], None] | None = None


@dataclass(frozen=True, slots=True)
class _MissingHeaderImage:
    source_relative_path: str


@dataclass(frozen=True, slots=True)
class _TerminalHeaderImage:
    source_relative_path: str
    source_sha256: str
    error_code: str


@dataclass(frozen=True, slots=True)
class _SuccessfulLongTextPart:
    part: LongTextPart
    generation_run_id: str
    stored_vector_sha256: str
    vector: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _PreparedEmbedding:
    input_sha256: str
    stored_vector_sha256: str
    vector: tuple[float, ...]
    long_text_parts: tuple[_SuccessfulLongTextPart, ...] = ()


@dataclass(frozen=True, slots=True)
class _BatchedHostedWork:
    article: CanonicalArticle
    modality: str
    embedding_input: JinaEmbeddingInput
    input_sha256: str
    prepared_image: _PreparedHeaderImage | None = None
    part: LongTextPart | None = None
    article_input_sha256: str | None = None


class EmbeddingPipelineError(RuntimeError):
    """Content-free coordinator failure with a stable machine-readable code."""

    def __init__(self, code: str, article_id: str) -> None:
        self.code = code
        self.article_id = article_id
        super().__init__(f"[{code}] article_id={article_id}")


class _LongTextTerminal(RuntimeError):
    """Internal control flow after a durable terminal part transition."""


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
    limit: int | None = None,
    full: bool = False,
    reprocess: bool = False,
    failpoints: EmbeddingPipelineFailpoints | None = None,
    image_codec: ImageCodec | None = None,
    batch_sleep: Callable[[float], None] | None = None,
    batch_jitter: Callable[[int], float] | None = None,
    monotonic: Callable[[], float] | None = None,
    page_size: int = 100,
) -> EmbeddingRunResult:
    """Publish source and derived vectors for a bounded canonical slice."""

    run_clock = time.monotonic if monotonic is None else monotonic
    run_started_at = run_clock()
    if not config.hosted_processing_authorized:
        raise HostedProcessingAuthorizationError
    if not isinstance(full, bool):
        raise ValueError("full must be boolean")
    if (limit is None) == (not full):
        raise ValueError("exactly one positive limit or full scope is required")
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
    ):
        raise ValueError("limit must be positive")
    if not isinstance(reprocess, bool):
        raise ValueError("reprocess must be boolean")
    if isinstance(page_size, bool) or not isinstance(page_size, int) or page_size < 1:
        raise ValueError("page size must be positive")
    config.validate()
    _ensure_source_root(config.source_root)
    active_failpoints = failpoints or EmbeddingPipelineFailpoints()
    if full:
        return _run_full_embedding_pipeline(
            config,
            adapter,
            reprocess=reprocess,
            failpoints=active_failpoints,
            image_codec=image_codec,
            batch_sleep=batch_sleep,
            batch_jitter=batch_jitter,
            monotonic=monotonic,
            page_size=page_size,
        )
    assert limit is not None
    articles = _read_articles(config, limit)
    prepared_images = {
        article.article_id: _prepare_header_image(config, article)
        for article in articles
    }
    active_image_codec = image_codec
    if active_image_codec is None:
        active_image_codec = getattr(adapter, "image_codec", None)
    bind_image_codec = getattr(adapter, "bind_image_codec", None)
    if active_image_codec is None and (
        callable(bind_image_codec)
        or any(
            isinstance(image, _PreparedHeaderImage)
            for image in prepared_images.values()
        )
    ):
        try:
            active_image_codec = PillowImageCodec()
        except ImageCodecError as error:
            raise EmbeddingPipelineError(error.code, "image_preparation") from error
    if callable(bind_image_codec) and active_image_codec is not None:
        try:
            bind_image_codec(active_image_codec)
        except ImageCodecError as error:
            raise EmbeddingPipelineError(error.code, "image_preparation") from error
    profile = adapter.profile
    if profile.dimensions != _VECTOR_DIMENSIONS:
        raise ValueError("adapter profile dimensions must be 2048")
    if profile.long_text_part_attempt_limit < 1:
        raise ValueError("long-text part attempt limit must be positive")
    _validate_batch_profile(profile)
    configuration_identifier = configuration_id(profile)
    run_id = str(uuid4())
    with (
        embedding_pipeline_lock(config.embedding_lock_path) as output_descriptor,
        EmbeddingCatalog.open_in(output_descriptor) as catalog,
    ):
        if active_image_codec is not None:
            prepared_images = {
                article_id: _prepare_image_input(
                    prepared_image,
                    profile,
                    active_image_codec,
                    output_descriptor,
                    active_failpoints,
                )
                for article_id, prepared_image in prepared_images.items()
            }
            try:
                for prepared_image in prepared_images.values():
                    if (
                        isinstance(prepared_image, _PreparedHeaderImage)
                        and prepared_image.image_input is not None
                        and prepared_image.image_input.rendition_identity is not None
                    ):
                        verify_image_rendition(
                            output_descriptor,
                            prepared_image.image_input.rendition_identity,
                            prepared_image.image_input.data,
                            prepared_image.image_input.embedded_info,
                            active_image_codec,
                        )
            except ImageCodecError as error:
                raise EmbeddingPipelineError(
                    error.code, "image_preparation"
                ) from error
        with catalog.transaction():
            catalog.begin_run(
                run_id,
                configuration_identifier,
                profile,
                len(articles),
                scope="full" if full else "limited",
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
        if (
            getattr(adapter, "rate_aware_batching", False)
            and callable(getattr(adapter, "embed_batch", None))
        ):
            metrics = _run_rate_aware_work(
                config,
                adapter,
                catalog,
                run_id,
                configuration_identifier,
                profile,
                articles,
                prepared_images,
                registered_states,
                reprocess=reprocess,
                failpoints=active_failpoints,
                batch_sleep=batch_sleep,
                batch_jitter=batch_jitter,
                monotonic=monotonic,
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
            with catalog.transaction():
                catalog.finish_run_metrics(
                    run_id,
                    hosted_requests=metrics.requests,
                    hosted_retries=metrics.retries,
                    usage=metrics.usage,
                    throttles=metrics.throttles,
                    elapsed_seconds=max(0.0, run_clock() - run_started_at),
                )
                catalog.finish_run(run_id, reconciled=False)
            return catalog.run_result(run_id)
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
                if modality == _ARTICLE_TEXT_MODALITY and not reprocess:
                    with catalog.transaction():
                        exhausted = catalog.terminalize_exhausted_long_text_work(
                            run_id=run_id,
                            article_id=article.article_id,
                            configuration_identifier=configuration_identifier,
                            article_input_sha256=(
                                article.cleaned_markdown_sha256
                            ),
                            attempt_limit=profile.long_text_part_attempt_limit,
                        )
                    if exhausted:
                        continue
                with catalog.transaction():
                    catalog.start_attempt(
                        run_id=run_id,
                        article_id=article.article_id,
                        modality=modality,
                        configuration_identifier=configuration_identifier,
                    )
                try:
                    prepared_embedding = _prepare_embedding(
                        config,
                        adapter,
                        catalog,
                        run_id,
                        configuration_identifier,
                        article,
                        modality=modality,
                        prepared_image=prepared_image,
                        reprocess=reprocess,
                        failpoints=active_failpoints,
                    )
                except _LongTextTerminal:
                    continue
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
                        terminal_long_text = (
                            modality == _ARTICLE_TEXT_MODALITY
                            and catalog.has_terminal_long_text_part(
                                article_id=article.article_id,
                                configuration_identifier=configuration_identifier,
                                article_input_sha256=(
                                    article.cleaned_markdown_sha256
                                ),
                            )
                        )
                        failure_state = (
                            WorkState.RETRYABLE
                            if (
                                not terminal_long_text
                                and (error.retryable or bounded_deterministic_image)
                            )
                            else WorkState.TERMINAL
                        )
                        if not terminal_long_text:
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
                    if prepared_embedding.long_text_parts:
                        catalog.publish_long_text_success(
                            run_id=run_id,
                            article_id=article.article_id,
                            configuration_identifier=configuration_identifier,
                            published_at_utc=article.published_at_utc,
                            publication_date_new_york=(
                                article.publication_date_new_york
                            ),
                            dimensions=profile.dimensions,
                            input_sha256=prepared_embedding.input_sha256,
                            stored_vector_sha256=(
                                prepared_embedding.stored_vector_sha256
                            ),
                            vector=prepared_embedding.vector,
                            aggregation_version=profile.long_text_aggregation,
                            parts=tuple(
                                (
                                    item.part.index,
                                    item.part.input_sha256,
                                    item.part.token_count,
                                    item.generation_run_id,
                                    item.stored_vector_sha256,
                                )
                                for item in prepared_embedding.long_text_parts
                            ),
                        )
                    elif modality == _HEADER_IMAGE_MODALITY:
                        assert isinstance(prepared_image, _PreparedHeaderImage)
                        assert prepared_image.image_input is not None
                        image_input = prepared_image.image_input
                        catalog.publish_image_success(
                            run_id=run_id,
                            article_id=article.article_id,
                            configuration_identifier=configuration_identifier,
                            source_relative_path=prepared_image.source_relative_path,
                            published_at_utc=article.published_at_utc,
                            publication_date_new_york=(
                                article.publication_date_new_york
                            ),
                            dimensions=profile.dimensions,
                            source_sha256=image_input.source_sha256,
                            source_format=image_input.source_info.format,
                            source_bytes=len(prepared_image.data),
                            source_width=image_input.source_info.width,
                            source_height=image_input.source_info.height,
                            source_frames=image_input.source_info.frames,
                            embedded_input_sha256=(
                                image_input.embedded_input_sha256
                            ),
                            embedded_format=image_input.embedded_info.format,
                            embedded_bytes=len(image_input.data),
                            embedded_width=image_input.embedded_info.width,
                            embedded_height=image_input.embedded_info.height,
                            embedded_frames=image_input.embedded_info.frames,
                            transform_id=image_input.transform_id,
                            rendition_relative_path=(
                                image_input.rendition_relative_path
                            ),
                            stored_vector_sha256=(
                                prepared_embedding.stored_vector_sha256
                            ),
                            vector=prepared_embedding.vector,
                        )
                    else:
                        catalog.publish_success(
                            run_id=run_id,
                            article_id=article.article_id,
                            modality=modality,
                            configuration_identifier=configuration_identifier,
                            source_relative_path=source_relative_path,
                            published_at_utc=article.published_at_utc,
                            publication_date_new_york=(
                                article.publication_date_new_york
                            ),
                            dimensions=profile.dimensions,
                            input_sha256=prepared_embedding.input_sha256,
                            stored_vector_sha256=(
                                prepared_embedding.stored_vector_sha256
                            ),
                            vector=prepared_embedding.vector,
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
        with catalog.transaction():
            catalog.finish_run(run_id, reconciled=False)
        result = catalog.run_result(run_id)
    return result


def _run_full_embedding_pipeline(
    config: EmbeddingPipelineConfig,
    adapter: EmbeddingAdapter,
    *,
    reprocess: bool,
    failpoints: EmbeddingPipelineFailpoints,
    image_codec: ImageCodec | None,
    batch_sleep: Callable[[float], None] | None,
    batch_jitter: Callable[[int], float] | None,
    monotonic: Callable[[], float] | None,
    page_size: int,
) -> EmbeddingRunResult:
    """Traverse, process, and reconcile one explicit full run in bounded pages."""

    run_clock = time.monotonic if monotonic is None else monotonic
    run_started_at = run_clock()
    active_image_codec = image_codec or getattr(adapter, "image_codec", None)
    bind_image_codec = getattr(adapter, "bind_image_codec", None)
    if active_image_codec is None and callable(bind_image_codec):
        try:
            active_image_codec = PillowImageCodec()
        except ImageCodecError as error:
            raise EmbeddingPipelineError(error.code, "image_preparation") from error
    if callable(bind_image_codec) and active_image_codec is not None:
        try:
            bind_image_codec(active_image_codec)
        except ImageCodecError as error:
            raise EmbeddingPipelineError(error.code, "image_preparation") from error
    profile = adapter.profile
    if profile.dimensions != _VECTOR_DIMENSIONS:
        raise ValueError("adapter profile dimensions must be 2048")
    if profile.long_text_part_attempt_limit < 1:
        raise ValueError("long-text part attempt limit must be positive")
    _validate_batch_profile(profile)
    configuration_identifier = configuration_id(profile)
    run_id = str(uuid4())
    totals = {
        "requests": 0,
        "retries": 0,
        "usage": {},
        "throttles": 0,
    }
    with (
        embedding_pipeline_lock(config.embedding_lock_path) as output_descriptor,
        EmbeddingCatalog.open_in(output_descriptor) as catalog,
    ):
        with catalog.transaction():
            catalog.begin_run(
                run_id,
                configuration_identifier,
                profile,
                0,
                scope="full",
            )
        try:
            with _preprocessing_snapshot(config) as preprocessing_db:
                after_article_id: str | None = None
                while True:
                    articles = _read_article_page(
                        config,
                        after_article_id=after_article_id,
                        page_size=page_size,
                        connection=preprocessing_db,
                    )
                    if not articles:
                        break
                    with catalog.transaction():
                        catalog.record_full_discovery_page(run_id, articles)
                    if failpoints.after_full_discovery_page is not None:
                        failpoints.after_full_discovery_page(len(articles))
                    after_article_id = articles[-1].article_id
                with catalog.transaction():
                    catalog.mark_full_discovery_complete(run_id)

                after_article_id = None
                while True:
                    article_ids = catalog.full_run_article_ids(
                        run_id,
                        after_article_id=after_article_id,
                        page_size=page_size,
                    )
                    if not article_ids:
                        break
                    articles = _read_articles_by_ids(
                        config,
                        article_ids,
                        connection=preprocessing_db,
                    )
                    if tuple(article.article_id for article in articles) != article_ids:
                        raise EmbeddingPipelineError(
                            "preprocessing_catalog", "unknown"
                        )
                    metrics = _process_full_article_page(
                        config,
                        adapter,
                        catalog,
                        output_descriptor,
                        run_id,
                        configuration_identifier,
                        profile,
                        articles,
                        reprocess=reprocess,
                        failpoints=failpoints,
                        image_codec=active_image_codec,
                        batch_sleep=batch_sleep,
                        batch_jitter=batch_jitter,
                        monotonic=monotonic,
                    )
                    if metrics is not None:
                        totals["requests"] += metrics.requests
                        totals["retries"] += metrics.retries
                        totals["throttles"] += metrics.throttles
                        for key, value in metrics.usage.items():
                            totals["usage"][key] = (
                                totals["usage"].get(key, 0) + value
                            )
                    if failpoints.after_full_processing_page is not None:
                        failpoints.after_full_processing_page(len(articles))
                    after_article_id = article_ids[-1]

            while True:
                with catalog.transaction():
                    reconciled = catalog.reconcile_full_page(
                        run_id,
                        page_size=page_size,
                        before_commit=(
                            failpoints.during_full_reconciliation_page
                        ),
                    )
                if not reconciled:
                    break
            referenced_renditions = {
                str(row[0])
                for row in catalog.connection.execute(
                    """
                    SELECT rendition_relative_path FROM image_input_provenance
                    WHERE rendition_relative_path IS NOT NULL
                    """
                ).fetchall()
            }
            with catalog.transaction():
                catalog.finish_run_metrics(
                    run_id,
                    hosted_requests=int(totals["requests"]),
                    hosted_retries=int(totals["retries"]),
                    usage=dict(totals["usage"]),
                    throttles=int(totals["throttles"]),
                    elapsed_seconds=max(0.0, run_clock() - run_started_at),
                )
                catalog.finish_run(run_id, reconciled=True)
            result = catalog.run_result(run_id)
            if failpoints.before_full_rendition_cleanup is not None:
                failpoints.before_full_rendition_cleanup()
            try:
                cleanup_orphan_image_renditions(
                    output_descriptor,
                    referenced_renditions,
                )
            except ImageCodecError as error:
                raise EmbeddingPipelineError(
                    error.code, "rendition_cleanup"
                ) from error
            return result
        except Exception:
            with catalog.transaction():
                catalog.fail_run(run_id)
            raise
        except BaseException:
            with catalog.transaction():
                catalog.interrupt_run(run_id)
            raise


def _process_full_article_page(
    config: EmbeddingPipelineConfig,
    adapter: EmbeddingAdapter,
    catalog: EmbeddingCatalog,
    output_descriptor: int,
    run_id: str,
    configuration_identifier: str,
    profile: EmbeddingProfile,
    articles: Sequence[CanonicalArticle],
    *,
    reprocess: bool,
    failpoints: EmbeddingPipelineFailpoints,
    image_codec: ImageCodec | None,
    batch_sleep: Callable[[float], None] | None,
    batch_jitter: Callable[[int], float] | None,
    monotonic: Callable[[], float] | None,
) -> BatchExecutionResult | None:
    """Process one already-inventoried bounded page under the full-run lock."""

    prepared_images = {
        article.article_id: _prepare_header_image(config, article)
        for article in articles
    }
    active_image_codec = image_codec
    if active_image_codec is None and any(
        isinstance(image, _PreparedHeaderImage) for image in prepared_images.values()
    ):
        try:
            active_image_codec = PillowImageCodec()
        except ImageCodecError as error:
            raise EmbeddingPipelineError(error.code, "image_preparation") from error
    if active_image_codec is not None:
        prepared_images = {
            article_id: _prepare_image_input(
                prepared_image,
                profile,
                active_image_codec,
                output_descriptor,
                failpoints,
            )
            for article_id, prepared_image in prepared_images.items()
        }
        try:
            for prepared_image in prepared_images.values():
                if (
                    isinstance(prepared_image, _PreparedHeaderImage)
                    and prepared_image.image_input is not None
                    and prepared_image.image_input.rendition_identity is not None
                ):
                    verify_image_rendition(
                        output_descriptor,
                        prepared_image.image_input.rendition_identity,
                        prepared_image.image_input.data,
                        prepared_image.image_input.embedded_info,
                        active_image_codec,
                    )
        except ImageCodecError as error:
            raise EmbeddingPipelineError(error.code, "image_preparation") from error
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
    if (
        getattr(adapter, "rate_aware_batching", False)
        and callable(getattr(adapter, "embed_batch", None))
    ):
        metrics = _run_rate_aware_work(
            config,
            adapter,
            catalog,
            run_id,
            configuration_identifier,
            profile,
            articles,
            prepared_images,
            registered_states,
            reprocess=reprocess,
            failpoints=failpoints,
            batch_sleep=batch_sleep,
            batch_jitter=batch_jitter,
            monotonic=monotonic,
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
        return metrics
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
            if modality == _ARTICLE_TEXT_MODALITY and not reprocess:
                with catalog.transaction():
                    exhausted = catalog.terminalize_exhausted_long_text_work(
                        run_id=run_id,
                        article_id=article.article_id,
                        configuration_identifier=configuration_identifier,
                        article_input_sha256=article.cleaned_markdown_sha256,
                        attempt_limit=profile.long_text_part_attempt_limit,
                    )
                if exhausted:
                    continue
            with catalog.transaction():
                catalog.start_attempt(
                    run_id=run_id,
                    article_id=article.article_id,
                    modality=modality,
                    configuration_identifier=configuration_identifier,
                )
            try:
                prepared_embedding = _prepare_embedding(
                    config,
                    adapter,
                    catalog,
                    run_id,
                    configuration_identifier,
                    article,
                    modality=modality,
                    prepared_image=prepared_image,
                    reprocess=reprocess,
                    failpoints=failpoints,
                )
            except _LongTextTerminal:
                continue
            except JinaHostedAdapterError as error:
                with catalog.transaction():
                    attempt_count = catalog.attempt_count(
                        article_id=article.article_id,
                        modality=modality,
                        configuration_identifier=configuration_identifier,
                    )
                    bounded_image = (
                        modality == _HEADER_IMAGE_MODALITY
                        and error.code == "deterministic_request"
                        and attempt_count < _DETERMINISTIC_IMAGE_ATTEMPT_LIMIT
                    )
                    failure_state = (
                        WorkState.RETRYABLE
                        if error.retryable or bounded_image
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
            with catalog.transaction():
                _publish_prepared_embedding(
                    catalog,
                    run_id,
                    configuration_identifier,
                    profile,
                    article,
                    modality,
                    prepared_image,
                    prepared_embedding,
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
    return None


def _publish_prepared_embedding(
    catalog: EmbeddingCatalog,
    run_id: str,
    configuration_identifier: str,
    profile: EmbeddingProfile,
    article: CanonicalArticle,
    modality: str,
    prepared_image: _PreparedHeaderImage | _MissingHeaderImage | None,
    prepared_embedding: _PreparedEmbedding,
) -> None:
    """Publish one validated non-batched source result inside a transaction."""

    if prepared_embedding.long_text_parts:
        catalog.publish_long_text_success(
            run_id=run_id,
            article_id=article.article_id,
            configuration_identifier=configuration_identifier,
            published_at_utc=article.published_at_utc,
            publication_date_new_york=article.publication_date_new_york,
            dimensions=profile.dimensions,
            input_sha256=prepared_embedding.input_sha256,
            stored_vector_sha256=prepared_embedding.stored_vector_sha256,
            vector=prepared_embedding.vector,
            aggregation_version=profile.long_text_aggregation,
            parts=tuple(
                (
                    item.part.index,
                    item.part.input_sha256,
                    item.part.token_count,
                    item.generation_run_id,
                    item.stored_vector_sha256,
                )
                for item in prepared_embedding.long_text_parts
            ),
        )
        return
    if modality == _HEADER_IMAGE_MODALITY:
        assert isinstance(prepared_image, _PreparedHeaderImage)
        assert prepared_image.image_input is not None
        image_input = prepared_image.image_input
        catalog.publish_image_success(
            run_id=run_id,
            article_id=article.article_id,
            configuration_identifier=configuration_identifier,
            source_relative_path=prepared_image.source_relative_path,
            published_at_utc=article.published_at_utc,
            publication_date_new_york=article.publication_date_new_york,
            dimensions=profile.dimensions,
            source_sha256=image_input.source_sha256,
            source_format=image_input.source_info.format,
            source_bytes=len(prepared_image.data),
            source_width=image_input.source_info.width,
            source_height=image_input.source_info.height,
            source_frames=image_input.source_info.frames,
            embedded_input_sha256=image_input.embedded_input_sha256,
            embedded_format=image_input.embedded_info.format,
            embedded_bytes=len(image_input.data),
            embedded_width=image_input.embedded_info.width,
            embedded_height=image_input.embedded_info.height,
            embedded_frames=image_input.embedded_info.frames,
            transform_id=image_input.transform_id,
            rendition_relative_path=image_input.rendition_relative_path,
            stored_vector_sha256=prepared_embedding.stored_vector_sha256,
            vector=prepared_embedding.vector,
        )
        return
    catalog.publish_success(
        run_id=run_id,
        article_id=article.article_id,
        modality=modality,
        configuration_identifier=configuration_identifier,
        source_relative_path=None,
        published_at_utc=article.published_at_utc,
        publication_date_new_york=article.publication_date_new_york,
        dimensions=profile.dimensions,
        input_sha256=prepared_embedding.input_sha256,
        stored_vector_sha256=prepared_embedding.stored_vector_sha256,
        vector=prepared_embedding.vector,
    )


def _validate_batch_profile(profile: EmbeddingProfile) -> None:
    try:
        BatchPolicy(
            profile.batch_max_items,
            profile.batch_max_estimated_tokens,
            profile.batch_max_encoded_bytes,
            profile.batch_max_concurrency,
            profile.batch_max_attempts,
            profile.batch_initial_backoff_seconds,
            profile.batch_max_backoff_seconds,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("adapter batch profile is invalid") from error


def _run_rate_aware_work(
    config: EmbeddingPipelineConfig,
    adapter: EmbeddingAdapter,
    catalog: EmbeddingCatalog,
    run_id: str,
    configuration_identifier: str,
    profile: EmbeddingProfile,
    articles: Sequence[CanonicalArticle],
    prepared_images: dict[
        str,
        _PreparedHeaderImage
        | _MissingHeaderImage
        | _TerminalHeaderImage
        | None,
    ],
    registered_states: dict[tuple[str, str], WorkState],
    *,
    reprocess: bool,
    failpoints: EmbeddingPipelineFailpoints,
    batch_sleep: Callable[[float], None] | None,
    batch_jitter: Callable[[int], float] | None,
    monotonic: Callable[[], float] | None,
) -> BatchExecutionResult:
    """Stage hosted-only work, then serialize each durable item transition."""

    works: list[_BatchedHostedWork] = []
    long_plans: dict[str, tuple[str, tuple[LongTextPart, ...]]] = {}
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
            if state is WorkState.TERMINAL or isinstance(
                prepared_image, _MissingHeaderImage
            ):
                continue
            if modality == _HEADER_IMAGE_MODALITY:
                assert isinstance(prepared_image, _PreparedHeaderImage)
                image_bytes = (
                    prepared_image.image_input.data
                    if prepared_image.image_input is not None
                    else prepared_image.data
                )
                works.append(
                    _BatchedHostedWork(
                        article=article,
                        modality=modality,
                        embedding_input=JinaEmbeddingInput.image_base64(
                            base64.b64encode(image_bytes).decode("ascii"),
                            estimated_tokens=(
                                math.ceil(
                                    prepared_image.image_input.embedded_info.width
                                    / 28
                                )
                                * math.ceil(
                                    prepared_image.image_input.embedded_info.height
                                    / 28
                                )
                                * 10
                            ),
                        ),
                        input_sha256=prepared_image.input_sha256,
                        prepared_image=prepared_image,
                    )
                )
                continue
            markdown, input_sha256, parts = _read_planned_article_text(
                config, adapter, article
            )
            if not parts:
                works.append(
                    _BatchedHostedWork(
                        article=article,
                        modality=modality,
                        embedding_input=JinaEmbeddingInput.text(
                            markdown,
                            estimated_tokens=len(adapter.token_offsets(markdown)),
                        ),
                        input_sha256=input_sha256,
                    )
                )
                continue
            if not reprocess:
                with catalog.transaction():
                    exhausted = catalog.terminalize_exhausted_long_text_work(
                        run_id=run_id,
                        article_id=article.article_id,
                        configuration_identifier=configuration_identifier,
                        article_input_sha256=input_sha256,
                        attempt_limit=profile.long_text_part_attempt_limit,
                    )
                if exhausted:
                    continue
            try:
                with catalog.transaction():
                    part_states = catalog.register_long_text_parts(
                        run_id=run_id,
                        article_id=article.article_id,
                        configuration_identifier=configuration_identifier,
                        article_input_sha256=input_sha256,
                        parts=parts,
                        reprocess=reprocess,
                    )
                    catalog.start_attempt(
                        run_id=run_id,
                        article_id=article.article_id,
                        modality=_ARTICLE_TEXT_MODALITY,
                        configuration_identifier=configuration_identifier,
                    )
            except ValueError as error:
                raise EmbeddingPipelineError(
                    "invalid_long_text_state", article.article_id
                ) from error
            if WorkState.TERMINAL in part_states:
                continue
            long_plans[article.article_id] = (input_sha256, parts)
            for part, part_state in zip(parts, part_states, strict=True):
                if part_state is WorkState.SUCCEEDED:
                    continue
                works.append(
                    _BatchedHostedWork(
                        article=article,
                        modality=modality,
                        embedding_input=JinaEmbeddingInput.text(
                            part.text, estimated_tokens=part.token_count
                        ),
                        input_sha256=part.input_sha256,
                        part=part,
                        article_input_sha256=input_sha256,
                    )
                )

    prior_attempt_counts: list[int] = []
    attempt_limits: list[int] = []
    for work in works:
        if work.part is None:
            prior_attempt_counts.append(
                catalog.attempt_count(
                    article_id=work.article.article_id,
                    modality=work.modality,
                    configuration_identifier=configuration_identifier,
                )
            )
            attempt_limits.append(profile.batch_max_attempts)
            continue
        assert work.article_input_sha256 is not None
        prior_attempt_counts.append(
            catalog.long_text_part_attempt_count(
                article_id=work.article.article_id,
                configuration_identifier=configuration_identifier,
                article_input_sha256=work.article_input_sha256,
                part_index=work.part.index,
            )
        )
        attempt_limits.append(
            min(profile.batch_max_attempts, profile.long_text_part_attempt_limit)
        )

    def checkpoint_attempt(index: int, _attempt: int) -> None:
        work = works[index]
        if work.part is None:
            with catalog.transaction():
                catalog.start_attempt(
                    run_id=run_id,
                    article_id=work.article.article_id,
                    modality=work.modality,
                    configuration_identifier=configuration_identifier,
                )
            return
        assert work.article_input_sha256 is not None
        with catalog.transaction():
            attempt_count = catalog.start_long_text_part_attempt(
                run_id=run_id,
                article_id=work.article.article_id,
                configuration_identifier=configuration_identifier,
                article_input_sha256=work.article_input_sha256,
                part_index=work.part.index,
            )
        if failpoints.after_long_text_part_attempt_checkpoint is not None:
            failpoints.after_long_text_part_attempt_checkpoint(
                work.part.index, attempt_count
            )

    failed_long_articles: set[str] = set()

    def commit_outcome(
        index: int,
        _attempt: int,
        outcome: JinaBatchItemOutcome,
        final: bool,
    ) -> None:
        work = works[index]
        if outcome.vector is not None:
            vector = _normalized_vector(
                outcome.vector.stored_vector, work.article.article_id
            )
            if work.part is not None:
                assert work.article_input_sha256 is not None
                with catalog.transaction():
                    catalog.publish_long_text_part_success(
                        run_id=run_id,
                        article_id=work.article.article_id,
                        configuration_identifier=configuration_identifier,
                        article_input_sha256=work.article_input_sha256,
                        part_index=work.part.index,
                        stored_vector_sha256=_vector_hash(vector),
                        vector=vector,
                    )
            else:
                _publish_batched_source_success(
                    catalog,
                    run_id,
                    configuration_identifier,
                    profile,
                    work,
                    vector,
                )
        else:
            error_code = outcome.error_code or "invalid_response"
            failure_state = (
                WorkState.TERMINAL
                if final or not outcome.retryable
                else WorkState.RETRYABLE
            )
            if work.part is not None:
                assert work.article_input_sha256 is not None
                if final:
                    failed_long_articles.add(work.article.article_id)
                with catalog.transaction():
                    if failure_state is WorkState.TERMINAL:
                        catalog.record_terminal_long_text_part_failure(
                            run_id=run_id,
                            article_id=work.article.article_id,
                            configuration_identifier=configuration_identifier,
                            article_input_sha256=work.article_input_sha256,
                            part_index=work.part.index,
                            error_code=error_code,
                            status_code=outcome.status_code,
                            retry_after_seconds=outcome.retry_after_seconds,
                        )
                    else:
                        catalog.record_long_text_part_failure(
                            run_id=run_id,
                            article_id=work.article.article_id,
                            configuration_identifier=configuration_identifier,
                            article_input_sha256=work.article_input_sha256,
                            part_index=work.part.index,
                            state=failure_state,
                            error_code=error_code,
                            status_code=outcome.status_code,
                            retry_after_seconds=outcome.retry_after_seconds,
                        )
            else:
                with catalog.transaction():
                    catalog.record_failure(
                        run_id=run_id,
                        article_id=work.article.article_id,
                        modality=work.modality,
                        configuration_identifier=configuration_identifier,
                        state=failure_state,
                        error_code=error_code,
                        status_code=outcome.status_code,
                        retry_after_seconds=outcome.retry_after_seconds,
                        count_run=final,
                    )
                    if final and work.modality == _HEADER_IMAGE_MODALITY:
                        catalog.increment_run(run_id, "header_failed")
        if failpoints.after_batched_item_outcome_commit is not None:
            failpoints.after_batched_item_outcome_commit(index, final)

    execution_kwargs: dict[str, object] = {
        "before_attempt": checkpoint_attempt,
        "on_outcome": commit_outcome,
        "prior_attempt_counts": tuple(prior_attempt_counts),
        "attempt_limits": tuple(attempt_limits),
    }
    if batch_sleep is not None:
        execution_kwargs["sleep"] = batch_sleep
    if batch_jitter is not None:
        execution_kwargs["jitter"] = batch_jitter
    metrics = execute_rate_aware_batches(
        adapter,
        tuple(work.embedding_input for work in works),
        policy=BatchPolicy(
            profile.batch_max_items,
            profile.batch_max_estimated_tokens,
            profile.batch_max_encoded_bytes,
            profile.batch_max_concurrency,
            profile.batch_max_attempts,
            profile.batch_initial_backoff_seconds,
            profile.batch_max_backoff_seconds,
        ),
        **execution_kwargs,
    )
    for article_id in failed_long_articles:
        with catalog.transaction():
            row = catalog.connection.execute(
                """
                SELECT state FROM embedding_work_items
                WHERE article_id = ? AND modality = ? AND configuration_id = ?
                """,
                [article_id, _ARTICLE_TEXT_MODALITY, configuration_identifier],
            ).fetchone()
            if row is not None and WorkState(str(row[0])) is not WorkState.TERMINAL:
                catalog.record_failure(
                    run_id=run_id,
                    article_id=article_id,
                    modality=_ARTICLE_TEXT_MODALITY,
                    configuration_identifier=configuration_identifier,
                    state=WorkState.RETRYABLE,
                    error_code="long_text_part_retryable",
                    status_code=None,
                    retry_after_seconds=None,
                )
    for article in articles:
        plan = long_plans.get(article.article_id)
        if plan is None or article.article_id in failed_long_articles:
            continue
        article_input_sha256, parts = plan
        successful_parts = tuple(
            _successful_catalog_part(
                catalog,
                article,
                configuration_identifier,
                article_input_sha256,
                part,
            )
            for part in parts
        )
        aggregate = _long_text_vector(successful_parts, article.article_id)
        with catalog.transaction():
            catalog.publish_long_text_success(
                run_id=run_id,
                article_id=article.article_id,
                configuration_identifier=configuration_identifier,
                published_at_utc=article.published_at_utc,
                publication_date_new_york=article.publication_date_new_york,
                dimensions=profile.dimensions,
                input_sha256=article_input_sha256,
                stored_vector_sha256=_vector_hash(aggregate),
                vector=aggregate,
                aggregation_version=profile.long_text_aggregation,
                parts=tuple(
                    (
                        item.part.index,
                        item.part.input_sha256,
                        item.part.token_count,
                        item.generation_run_id,
                        item.stored_vector_sha256,
                    )
                    for item in successful_parts
                ),
            )
    return metrics


def _read_planned_article_text(
    config: EmbeddingPipelineConfig,
    adapter: EmbeddingAdapter,
    article: CanonicalArticle,
) -> tuple[str, str, tuple[LongTextPart, ...]]:
    try:
        markdown_bytes = read_canonical_markdown(
            config.preprocessing_output_root, article.cleaned_markdown_path
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
    try:
        parts = plan_long_text_parts(
            markdown, adapter, token_limit=adapter.profile.context_token_limit
        )
    except LongTextPlanningError as error:
        raise EmbeddingPipelineError(
            "invalid_tokenizer_offsets", article.article_id
        ) from error
    return markdown, input_sha256, parts


def _successful_catalog_part(
    catalog: EmbeddingCatalog,
    article: CanonicalArticle,
    configuration_identifier: str,
    article_input_sha256: str,
    part: LongTextPart,
) -> _SuccessfulLongTextPart:
    try:
        generation_run_id, vector_sha256, vector_values = (
            catalog.long_text_part_generation(
                article_id=article.article_id,
                configuration_identifier=configuration_identifier,
                article_input_sha256=article_input_sha256,
                part_index=part.index,
            )
        )
        vector = _verified_source_vector(
            vector_values, str(vector_sha256), article.article_id
        )
    except (EmbeddingPipelineError, ValueError) as error:
        raise EmbeddingPipelineError(
            "invalid_long_text_state", article.article_id
        ) from error
    return _SuccessfulLongTextPart(
        part, str(generation_run_id), str(vector_sha256), vector
    )


def _publish_batched_source_success(
    catalog: EmbeddingCatalog,
    run_id: str,
    configuration_identifier: str,
    profile: EmbeddingProfile,
    work: _BatchedHostedWork,
    vector: tuple[float, ...],
) -> None:
    with catalog.transaction():
        if work.modality == _HEADER_IMAGE_MODALITY:
            assert work.prepared_image is not None
            assert work.prepared_image.image_input is not None
            image_input = work.prepared_image.image_input
            catalog.publish_image_success(
                run_id=run_id,
                article_id=work.article.article_id,
                configuration_identifier=configuration_identifier,
                source_relative_path=work.prepared_image.source_relative_path,
                published_at_utc=work.article.published_at_utc,
                publication_date_new_york=work.article.publication_date_new_york,
                dimensions=profile.dimensions,
                source_sha256=image_input.source_sha256,
                source_format=image_input.source_info.format,
                source_bytes=len(work.prepared_image.data),
                source_width=image_input.source_info.width,
                source_height=image_input.source_info.height,
                source_frames=image_input.source_info.frames,
                embedded_input_sha256=image_input.embedded_input_sha256,
                embedded_format=image_input.embedded_info.format,
                embedded_bytes=len(image_input.data),
                embedded_width=image_input.embedded_info.width,
                embedded_height=image_input.embedded_info.height,
                embedded_frames=image_input.embedded_info.frames,
                transform_id=image_input.transform_id,
                rendition_relative_path=image_input.rendition_relative_path,
                stored_vector_sha256=_vector_hash(vector),
                vector=vector,
            )
            return
        catalog.publish_success(
            run_id=run_id,
            article_id=work.article.article_id,
            modality=work.modality,
            configuration_identifier=configuration_identifier,
            source_relative_path=None,
            published_at_utc=work.article.published_at_utc,
            publication_date_new_york=work.article.publication_date_new_york,
            dimensions=profile.dimensions,
            input_sha256=work.input_sha256,
            stored_vector_sha256=_vector_hash(vector),
            vector=vector,
        )


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
    prepared_image: (
        _PreparedHeaderImage | _MissingHeaderImage | _TerminalHeaderImage | None
    ),
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
        if isinstance(prepared_image, _TerminalHeaderImage):
            state = catalog.register_work(
                run_id=run_id,
                article_id=article.article_id,
                modality=_HEADER_IMAGE_MODALITY,
                configuration_identifier=configuration_identifier,
                source_relative_path=prepared_image.source_relative_path,
                input_sha256=prepared_image.source_sha256,
                published_at_utc=article.published_at_utc,
                publication_date_new_york=article.publication_date_new_york,
                reprocess=False,
            )
            if state is not WorkState.TERMINAL:
                catalog.record_failure(
                    run_id=run_id,
                    article_id=article.article_id,
                    modality=_HEADER_IMAGE_MODALITY,
                    configuration_identifier=configuration_identifier,
                    state=WorkState.TERMINAL,
                    error_code=prepared_image.error_code,
                    status_code=None,
                    retry_after_seconds=None,
                )
                catalog.increment_run(run_id, "header_failed")
            return WorkState.TERMINAL
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
    limit: int | None,
) -> tuple[CanonicalArticle, ...]:
    try:
        with Catalog.read_only(config.preprocessing_catalog) as db:
            Catalog.validate_schema(db)
            sql = """
                SELECT article_id, cleaned_markdown_path, published_at_utc,
                       publication_date_new_york, cleaned_markdown_sha256,
                       header_image_path
                FROM articles
                ORDER BY article_id
            """
            rows = (
                db.execute(sql).fetchall()
                if limit is None
                else db.execute(f"{sql} LIMIT ?", [limit]).fetchall()
            )
    except (CatalogError, duckdb.Error, OSError) as error:
        raise EmbeddingPipelineError("preprocessing_catalog", "unknown") from error
    return tuple(CanonicalArticle(*row) for row in rows)


def _read_article_page(
    config: EmbeddingPipelineConfig,
    *,
    after_article_id: str | None,
    page_size: int,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> tuple[CanonicalArticle, ...]:
    """Read one stable lexical keyset page from the canonical catalog."""

    if connection is None:
        with _preprocessing_snapshot(config) as database:
            return _read_article_page(
                config,
                after_article_id=after_article_id,
                page_size=page_size,
                connection=database,
            )
    try:
        if after_article_id is None:
            rows = connection.execute(
                """
                SELECT article_id, cleaned_markdown_path, published_at_utc,
                       publication_date_new_york, cleaned_markdown_sha256,
                       header_image_path
                FROM articles ORDER BY article_id LIMIT ?
                """,
                [page_size],
            ).fetchmany(page_size)
        else:
            rows = connection.execute(
                """
                SELECT article_id, cleaned_markdown_path, published_at_utc,
                       publication_date_new_york, cleaned_markdown_sha256,
                       header_image_path
                FROM articles WHERE article_id > ?
                ORDER BY article_id LIMIT ?
                """,
                [after_article_id, page_size],
            ).fetchmany(page_size)
    except (CatalogError, duckdb.Error, OSError) as error:
        raise EmbeddingPipelineError("preprocessing_catalog", "unknown") from error
    return tuple(CanonicalArticle(*row) for row in rows)


def _read_articles_by_ids(
    config: EmbeddingPipelineConfig,
    article_ids: Sequence[str],
    *,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> tuple[CanonicalArticle, ...]:
    """Reopen one bounded inventory page from immutable preprocessing state."""

    if not article_ids:
        return ()
    if connection is None:
        with _preprocessing_snapshot(config) as database:
            return _read_articles_by_ids(
                config,
                article_ids,
                connection=database,
            )
    try:
        rows = connection.execute(
            """
            SELECT article_id, cleaned_markdown_path, published_at_utc,
                   publication_date_new_york, cleaned_markdown_sha256,
                   header_image_path
            FROM articles
            WHERE article_id IN (SELECT unnest(?))
            ORDER BY article_id
            """,
            [list(article_ids)],
        ).fetchmany(len(article_ids))
    except (CatalogError, duckdb.Error, OSError) as error:
        raise EmbeddingPipelineError("preprocessing_catalog", "unknown") from error
    return tuple(CanonicalArticle(*row) for row in rows)


@contextmanager
def _preprocessing_snapshot(
    config: EmbeddingPipelineConfig,
) -> Iterator[duckdb.DuckDBPyConnection]:
    """Hold one validated read-only preprocessing snapshot across full pages."""

    try:
        with Catalog.read_only(config.preprocessing_catalog) as connection:
            Catalog.validate_schema(connection)
            connection.execute("BEGIN TRANSACTION")
            try:
                yield connection
            finally:
                connection.execute("ROLLBACK")
    except (CatalogError, duckdb.Error, OSError) as error:
        raise EmbeddingPipelineError("preprocessing_catalog", "unknown") from error


def _prepare_embedding(
    config: EmbeddingPipelineConfig,
    adapter: EmbeddingAdapter,
    catalog: EmbeddingCatalog,
    run_id: str,
    configuration_identifier: str,
    article: CanonicalArticle,
    *,
    modality: str,
    prepared_image: (
        _PreparedHeaderImage | _MissingHeaderImage | _TerminalHeaderImage | None
    ),
    reprocess: bool,
    failpoints: EmbeddingPipelineFailpoints,
) -> _PreparedEmbedding:
    if modality == _HEADER_IMAGE_MODALITY:
        assert isinstance(prepared_image, _PreparedHeaderImage)
        image_bytes = (
            prepared_image.image_input.data
            if prepared_image.image_input is not None
            else prepared_image.data
        )
        embedded = adapter.embed_image(
            base64.b64encode(image_bytes).decode("ascii")
        )
        vector = _normalized_vector(embedded, article.article_id)
        return _PreparedEmbedding(
            input_sha256=prepared_image.input_sha256,
            stored_vector_sha256=_vector_hash(vector),
            vector=vector,
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
    try:
        parts = plan_long_text_parts(
            markdown,
            adapter,
            token_limit=adapter.profile.context_token_limit,
        )
    except LongTextPlanningError as error:
        raise EmbeddingPipelineError(
            "invalid_tokenizer_offsets", article.article_id
        ) from error
    if not parts:
        embedded = adapter.embed_text(markdown)
        vector = _normalized_vector(embedded, article.article_id)
        return _PreparedEmbedding(
            input_sha256=input_sha256,
            stored_vector_sha256=_vector_hash(vector),
            vector=vector,
        )
    successful_parts = _embed_long_text_parts(
        catalog,
        adapter,
        run_id=run_id,
        configuration_identifier=configuration_identifier,
        article=article,
        article_input_sha256=input_sha256,
        parts=parts,
        reprocess=reprocess,
        failpoints=failpoints,
    )
    aggregate = _long_text_vector(successful_parts, article.article_id)
    return _PreparedEmbedding(
        input_sha256=input_sha256,
        stored_vector_sha256=_vector_hash(aggregate),
        vector=aggregate,
        long_text_parts=successful_parts,
    )


def _embed_long_text_parts(
    catalog: EmbeddingCatalog,
    adapter: EmbeddingAdapter,
    *,
    run_id: str,
    configuration_identifier: str,
    article: CanonicalArticle,
    article_input_sha256: str,
    parts: tuple[LongTextPart, ...],
    reprocess: bool,
    failpoints: EmbeddingPipelineFailpoints,
) -> tuple[_SuccessfulLongTextPart, ...]:
    """Resume exact part checkpoints and purchase only unproven vectors."""

    try:
        with catalog.transaction():
            states = catalog.register_long_text_parts(
                run_id=run_id,
                article_id=article.article_id,
                configuration_identifier=configuration_identifier,
                article_input_sha256=article_input_sha256,
                parts=parts,
                reprocess=reprocess,
            )
    except ValueError as error:
        raise EmbeddingPipelineError(
            "invalid_long_text_state", article.article_id
        ) from error
    successful: list[_SuccessfulLongTextPart] = []
    for part, state in zip(parts, states, strict=True):
        if state is WorkState.TERMINAL:
            raise _LongTextTerminal
        attempt_count = catalog.long_text_part_attempt_count(
            article_id=article.article_id,
            configuration_identifier=configuration_identifier,
            article_input_sha256=article_input_sha256,
            part_index=part.index,
        )
        if (
            state is not WorkState.SUCCEEDED
            and attempt_count >= adapter.profile.long_text_part_attempt_limit
        ):
            with catalog.transaction():
                catalog.terminalize_exhausted_long_text_work(
                    run_id=run_id,
                    article_id=article.article_id,
                    configuration_identifier=configuration_identifier,
                    article_input_sha256=article_input_sha256,
                    attempt_limit=adapter.profile.long_text_part_attempt_limit,
                )
            raise _LongTextTerminal
        if state is not WorkState.SUCCEEDED:
            with catalog.transaction():
                attempt_count = catalog.start_long_text_part_attempt(
                    run_id=run_id,
                    article_id=article.article_id,
                    configuration_identifier=configuration_identifier,
                    article_input_sha256=article_input_sha256,
                    part_index=part.index,
                )
            if failpoints.after_long_text_part_attempt_checkpoint is not None:
                failpoints.after_long_text_part_attempt_checkpoint(
                    part.index,
                    attempt_count,
                )
            try:
                vector = _normalized_vector(
                    adapter.embed_text(part.text),
                    article.article_id,
                )
            except JinaHostedAdapterError as error:
                with catalog.transaction():
                    attempt_count = catalog.long_text_part_attempt_count(
                        article_id=article.article_id,
                        configuration_identifier=configuration_identifier,
                        article_input_sha256=article_input_sha256,
                        part_index=part.index,
                    )
                    failure_state = (
                        WorkState.RETRYABLE
                        if (
                            error.retryable
                            and attempt_count
                            < adapter.profile.long_text_part_attempt_limit
                        )
                        else WorkState.TERMINAL
                    )
                    if failure_state is WorkState.TERMINAL:
                        catalog.record_terminal_long_text_part_failure(
                            run_id=run_id,
                            article_id=article.article_id,
                            configuration_identifier=configuration_identifier,
                            article_input_sha256=article_input_sha256,
                            part_index=part.index,
                            error_code=error.code,
                            status_code=error.status_code,
                            retry_after_seconds=error.retry_after_seconds,
                        )
                    else:
                        catalog.record_long_text_part_failure(
                            run_id=run_id,
                            article_id=article.article_id,
                            configuration_identifier=configuration_identifier,
                            article_input_sha256=article_input_sha256,
                            part_index=part.index,
                            state=failure_state,
                            error_code=error.code,
                            status_code=error.status_code,
                            retry_after_seconds=error.retry_after_seconds,
                        )
                if (
                    failure_state is WorkState.TERMINAL
                    and failpoints.after_long_text_terminal_transition is not None
                ):
                    failpoints.after_long_text_terminal_transition(part.index)
                raise
            vector_sha256 = _vector_hash(vector)
            with catalog.transaction():
                catalog.publish_long_text_part_success(
                    run_id=run_id,
                    article_id=article.article_id,
                    configuration_identifier=configuration_identifier,
                    article_input_sha256=article_input_sha256,
                    part_index=part.index,
                    stored_vector_sha256=vector_sha256,
                    vector=vector,
                )
        try:
            generation_run_id, vector_sha256, vector_values = (
                catalog.long_text_part_generation(
                    article_id=article.article_id,
                    configuration_identifier=configuration_identifier,
                    article_input_sha256=article_input_sha256,
                    part_index=part.index,
                )
            )
            vector = _verified_source_vector(
                vector_values,
                str(vector_sha256),
                article.article_id,
            )
        except (EmbeddingPipelineError, ValueError) as error:
            raise EmbeddingPipelineError(
                "invalid_long_text_state", article.article_id
            ) from error
        successful.append(
            _SuccessfulLongTextPart(
                part=part,
                generation_run_id=str(generation_run_id),
                stored_vector_sha256=str(vector_sha256),
                vector=vector,
            )
        )
    return tuple(successful)


def _long_text_vector(
    parts: Sequence[_SuccessfulLongTextPart],
    article_id: str,
) -> tuple[float, ...]:
    """Return the normalized token-count-weighted mean of normalized parts."""

    if not parts or any(part.part.token_count < 1 for part in parts):
        raise EmbeddingPipelineError("invalid_long_text_state", article_id)
    total_tokens = sum(part.part.token_count for part in parts)
    weighted = tuple(
        sum(
            part.part.token_count * part.vector[dimension]
            for part in parts
        )
        / total_tokens
        for dimension in range(_VECTOR_DIMENSIONS)
    )
    return _normalized_vector(weighted, article_id)


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


def _prepare_image_input(
    prepared_image: _PreparedHeaderImage | _MissingHeaderImage | None,
    profile: EmbeddingProfile,
    image_codec: ImageCodec,
    output_descriptor: int,
    failpoints: EmbeddingPipelineFailpoints,
) -> _PreparedHeaderImage | _MissingHeaderImage | _TerminalHeaderImage | None:
    if not isinstance(prepared_image, _PreparedHeaderImage):
        return prepared_image
    try:
        image_input = prepare_image_input(
            prepared_image.data,
            codec=image_codec,
            expected_input_rules=profile.image_input_rules,
            expected_transform_id=profile.image_transform,
            install_rendition=lambda data, info, derived_sha256: (
                install_image_rendition(
                    output_descriptor,
                    data,
                    info,
                    derived_sha256,
                    transform_id=image_codec.transform_id,
                    codec=image_codec,
                )
            ),
        )
    except ImageCodecError as error:
        if error.code not in _TERMINAL_IMAGE_PREPARATION_CODES:
            raise EmbeddingPipelineError(
                error.code,
                "image_preparation",
            ) from error
        return _TerminalHeaderImage(
            source_relative_path=prepared_image.source_relative_path,
            source_sha256=prepared_image.input_sha256,
            error_code=error.code,
        )
    if (
        image_input.rendition_relative_path is not None
        and failpoints.after_image_rendition_install is not None
    ):
        failpoints.after_image_rendition_install(
            image_input.rendition_relative_path,
        )
    return _PreparedHeaderImage(
        source_relative_path=prepared_image.source_relative_path,
        input_sha256=prepared_image.input_sha256,
        data=prepared_image.data,
        image_input=image_input,
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
    input_sha256 = _multimodal_input_sha256(
        formula_version=formula_version,
        text_generation_run_id=str(text_generation_run_id),
        text_stored_vector_sha256=str(text_vector_sha256),
        header_image_generation_run_id=str(image_generation_run_id),
        header_image_stored_vector_sha256=str(image_vector_sha256),
    )
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


def _multimodal_input_sha256(
    *,
    formula_version: str,
    text_generation_run_id: str,
    text_stored_vector_sha256: str,
    header_image_generation_run_id: str,
    header_image_stored_vector_sha256: str,
) -> str:
    """Hash the complete content-free identity for one derived generation."""

    provenance_payload = json.dumps(
        {
            "formula_version": formula_version,
            "header_image_generation_run_id": header_image_generation_run_id,
            "header_image_stored_vector_sha256": (
                header_image_stored_vector_sha256
            ),
            "text_generation_run_id": text_generation_run_id,
            "text_stored_vector_sha256": text_stored_vector_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(provenance_payload).hexdigest()


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
