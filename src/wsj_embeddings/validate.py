"""Credential-free integrity validation for published article embeddings."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import struct
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import duckdb

from wsj_embeddings.canonical_markdown import (
    CanonicalMarkdownError,
    read_canonical_markdown,
)
from wsj_embeddings.catalog import EmbeddingCatalog, EmbeddingCatalogError
from wsj_embeddings.catalog import configuration_id as profile_configuration_id
from wsj_embeddings.config import EmbeddingPipelineConfig
from wsj_embeddings.image_rendition import (
    MAX_HOSTED_IMAGE_BYTES,
    MAX_HOSTED_IMAGE_PIXELS,
    MAX_SAFE_DECODE_PIXELS,
    PRODUCTION_IMAGE_INPUT_RULES,
    PRODUCTION_IMAGE_TRANSFORM_ID,
    SOURCE_BYTES_TRANSFORM_ID,
    FixturePassthroughImageCodec,
    ImageCodec,
    ImageCodecError,
    PillowImageCodec,
    scaled_image_dimensions,
)
from wsj_embeddings.models import (
    CanonicalArticle,
    EmbeddingModality,
    EmbeddingProfile,
    SupersessionReason,
    WorkState,
)
from wsj_embeddings.pipeline import (
    _MULTIMODAL_FORMULA,
    _VECTOR_DIMENSIONS,
    EmbeddingPipelineError,
    _multimodal_input_sha256,
    _multimodal_vector,
    _vector_hash,
    _verified_source_vector,
)
from wsj_embeddings.run_metrics import safe_usage_is_valid
from wsj_embeddings.source_image import (
    SourceImageError,
    is_safe_source_relative_path,
    read_source_image,
)
from wsj_pipeline.catalog import Catalog, CatalogError

_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)


class _Digest(Protocol):
    def update(self, value: bytes) -> None: ...


@dataclass(frozen=True, slots=True)
class EmbeddingValidationIssue:
    """One stable validation failure without text, vectors, or credentials."""

    code: str
    message: str


@dataclass(frozen=True, slots=True)
class EmbeddingValidationResult:
    """All deterministic embedding-output validation failures."""

    issues: tuple[EmbeddingValidationIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues


@dataclass(frozen=True, slots=True)
class EmbeddingCoverage:
    """Content-free accounting for one immutable embedding configuration."""

    canonical_articles: int
    text_success: int
    text_failure: int
    header_present: int
    header_absent: int
    header_success: int
    header_failure: int
    multimodal_success: int
    multimodal_unavailable: int
    multimodal_failure: int
    stale_work: int
    unresolved_retryable_work: int
    orphan_artifacts: int


@dataclass(frozen=True, slots=True)
class EmbeddingCorpusValidationResult:
    """Integrity issues and coverage for one selected configuration."""

    configuration_id: str | None
    coverage: EmbeddingCoverage | None
    issues: tuple[EmbeddingValidationIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues


@dataclass(frozen=True, slots=True)
class EmbeddingValidationFailpoints:
    """Generated-fixture seam while independent catalog snapshots are held."""

    after_state_snapshot: Callable[[], None] | None = None


def validate_embedding_corpus(
    config: EmbeddingPipelineConfig,
    *,
    configuration_id: str | None = None,
    image_codec: ImageCodec | None = None,
    failpoints: EmbeddingValidationFailpoints | None = None,
) -> EmbeddingCorpusValidationResult:
    """Return read-only integrity and content-free coverage for one config."""

    issues, selected_configuration_id, coverage = _validate_embedding_state(
        config,
        configuration_id=configuration_id,
        image_codec=image_codec,
        include_coverage=True,
        failpoints=failpoints or EmbeddingValidationFailpoints(),
    )
    return EmbeddingCorpusValidationResult(
        configuration_id=selected_configuration_id,
        coverage=coverage,
        issues=issues,
    )


def validate_embedding_outputs(
    config: EmbeddingPipelineConfig,
    *,
    configuration_id: str | None = None,
    image_codec: ImageCodec | None = None,
) -> EmbeddingValidationResult:
    """Check embedding output against canonical generated preprocessing state."""

    issues, _selected_configuration_id, _coverage = _validate_embedding_state(
        config,
        configuration_id=configuration_id,
        image_codec=image_codec,
        include_coverage=False,
        failpoints=EmbeddingValidationFailpoints(),
    )
    return EmbeddingValidationResult(issues)


def _validate_embedding_state(
    config: EmbeddingPipelineConfig,
    *,
    configuration_id: str | None,
    image_codec: ImageCodec | None,
    include_coverage: bool,
    failpoints: EmbeddingValidationFailpoints,
) -> tuple[
    tuple[EmbeddingValidationIssue, ...],
    str | None,
    EmbeddingCoverage | None,
]:
    """Validate against independently stable read-only catalog snapshots."""

    config.validate()
    issues: list[EmbeddingValidationIssue] = []
    selected_configuration_id: str | None = None
    coverage: EmbeddingCoverage | None = None
    try:
        with EmbeddingCatalog.read_only(config.embedding_catalog) as catalog:
            preprocessing_phase = True
            try:
                with Catalog.read_only(config.preprocessing_catalog) as preprocessing:
                    preprocessing.execute("BEGIN TRANSACTION")
                    try:
                        Catalog.validate_schema(preprocessing)
                        articles = _canonical_articles(preprocessing)
                        preprocessing_phase = False
                        initial_artifact_state = _artifact_state_token(
                            config,
                            catalog.connection,
                            articles,
                        )
                        if failpoints.after_state_snapshot is not None:
                            failpoints.after_state_snapshot()
                        selected_configuration_id, coverage = _validate_stable_state(
                            embedding=catalog.connection,
                            config=config,
                            articles=articles,
                            requested_configuration_id=configuration_id,
                            image_codec=image_codec,
                            include_coverage=include_coverage,
                            issues=issues,
                        )
                        if _artifact_state_token(
                            config,
                            catalog.connection,
                            articles,
                        ) != initial_artifact_state:
                            _append(
                                issues,
                                "concurrent_validation_state_change",
                                "generated artifact state changed during validation",
                            )
                    finally:
                        preprocessing.execute("ROLLBACK")
            except (CatalogError, OSError):
                _append(
                    issues,
                    "invalid_preprocessing_catalog",
                    "canonical preprocessing catalog could not be queried",
                )
            except duckdb.Error:
                if not preprocessing_phase:
                    raise
                _append(
                    issues,
                    "invalid_preprocessing_catalog",
                    "canonical preprocessing catalog could not be queried",
                )
    except EmbeddingCatalogError:
        issues.append(
            EmbeddingValidationIssue(
                "unsupported_embedding_catalog",
                "embedding catalog does not match the supported contract",
            )
        )
    except duckdb.Error:
        issues.append(
            EmbeddingValidationIssue(
                "invalid_embedding_catalog",
                "embedding catalog could not be queried with the supported schema",
            )
        )
    return (
        tuple(sorted(set(issues), key=lambda issue: (issue.code, issue.message))),
        selected_configuration_id,
        coverage,
    )


def _validate_stable_state(
    *,
    embedding: duckdb.DuckDBPyConnection,
    config: EmbeddingPipelineConfig,
    articles: dict[str, CanonicalArticle],
    requested_configuration_id: str | None,
    image_codec: ImageCodec | None,
    include_coverage: bool,
    issues: list[EmbeddingValidationIssue],
) -> tuple[str | None, EmbeddingCoverage | None]:
    """Validate pinned catalog states and their observed generated artifacts."""

    _validate_configurations(embedding, issues)
    _validate_run_metrics(embedding, issues)
    _validate_run_lifecycle(embedding, issues)
    _validate_reconciliation_actions(embedding, issues)
    _validate_global_public_embedding_checkpoints(embedding, issues)
    _validate_global_generation_history_references(embedding, issues)
    _validate_global_image_provenance_references(embedding, issues)
    orphan_artifacts = _scan_rendition_artifacts(embedding, config, issues)
    selected_configuration_id = _select_configuration_id(
        embedding,
        requested_configuration_id,
        issues,
    )
    coverage: EmbeddingCoverage | None = None
    if selected_configuration_id is not None:
        resolved_image_codec = _resolve_validation_image_codec(
            embedding,
            selected_configuration_id,
            image_codec,
            issues,
        )
        _validate_run_coverage(embedding, selected_configuration_id, issues)
        _validate_work_items(
            embedding,
            articles,
            selected_configuration_id,
            issues,
        )
        _validate_header_disposition_coverage(
            embedding,
            articles,
            selected_configuration_id,
            issues,
        )
        _validate_embeddings(
            embedding,
            config,
            articles,
            selected_configuration_id,
            issues,
        )
        _validate_long_text_parts(
            embedding,
            config,
            articles,
            selected_configuration_id,
            issues,
        )
        _validate_image_input_provenance(
            embedding,
            config,
            articles,
            selected_configuration_id,
            issues,
            resolved_image_codec,
        )
        _validate_multimodal_provenance(
            embedding,
            selected_configuration_id,
            issues,
        )
        _validate_generation_history(embedding, selected_configuration_id, issues)
        if include_coverage:
            coverage = _coverage_for_configuration(
                embedding,
                articles,
                selected_configuration_id,
                orphan_artifacts=orphan_artifacts,
            )
    elif (
        include_coverage
        and articles
        and not _connection_has_configurations(embedding)
    ):
        _append(
            issues,
            "missing_embedding_configuration",
            "canonical articles lack an embedding configuration",
        )
    return selected_configuration_id, coverage


def _connection_has_configurations(connection: duckdb.DuckDBPyConnection) -> bool:
    """Return whether an exact catalog contains an embedding configuration."""

    return bool(
        connection.execute("SELECT count(*) FROM embedding_configurations").fetchone()[
            0
        ]
    )


def _canonical_articles(
    connection: duckdb.DuckDBPyConnection,
) -> dict[str, CanonicalArticle]:
    rows = connection.execute(
        """
        SELECT article_id, cleaned_markdown_path, published_at_utc,
               publication_date_new_york, cleaned_markdown_sha256,
               header_image_path
        FROM articles
        """
    ).fetchall()
    return {row[0]: CanonicalArticle(*row) for row in rows}


def _artifact_state_token(
    config: EmbeddingPipelineConfig,
    embedding: duckdb.DuckDBPyConnection,
    articles: dict[str, CanonicalArticle],
) -> str:
    """Fingerprint referenced inputs and the complete no-follow rendition tree."""

    digest = hashlib.sha256()
    for article_id, article in sorted(articles.items()):
        try:
            markdown = read_canonical_markdown(
                config.preprocessing_output_root,
                article.cleaned_markdown_path,
            )
        except CanonicalMarkdownError as error:
            _update_artifact_token(
                digest,
                "markdown",
                article_id,
                article.cleaned_markdown_path,
                error.status,
            )
        else:
            _update_artifact_token(
                digest,
                "markdown",
                article_id,
                article.cleaned_markdown_path,
                hashlib.sha256(markdown).hexdigest(),
            )
        if article.header_image_path is not None:
            try:
                header = read_source_image(
                    config.source_root,
                    article.header_image_path,
                )
            except SourceImageError as error:
                _update_artifact_token(
                    digest,
                    "header",
                    article_id,
                    article.header_image_path,
                    error.status,
                )
            else:
                _update_artifact_token(
                    digest,
                    "header",
                    article_id,
                    article.header_image_path,
                    hashlib.sha256(header).hexdigest(),
                )
    rendition_paths = tuple(
        str(row[0])
        for row in embedding.execute(
            """
            SELECT rendition_relative_path FROM image_input_provenance
            WHERE rendition_relative_path IS NOT NULL
            ORDER BY rendition_relative_path
            """
        ).fetchall()
    )
    for relative_path in rendition_paths:
        try:
            rendition = read_source_image(
                config.embedding_output_root,
                relative_path,
            )
        except SourceImageError as error:
            _update_artifact_token(digest, "rendition", relative_path, error.status)
        else:
            _update_artifact_token(
                digest,
                "rendition",
                relative_path,
                hashlib.sha256(rendition).hexdigest(),
            )
    try:
        _update_rendition_tree_token(digest, config.embedding_output_root)
    except OSError:
        _update_artifact_token(digest, "rendition-tree", "unstable")
    return digest.hexdigest()


def _update_artifact_token(digest: _Digest, *values: object) -> None:
    encoded = json.dumps(values, separators=(",", ":"), ensure_ascii=True).encode()
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _update_rendition_tree_token(
    digest: _Digest,
    output_root: os.PathLike[str] | str,
) -> None:
    """Hash entry identities/bytes below renditions without following links."""

    try:
        output_descriptor = os.open(output_root, _DIRECTORY_FLAGS)
    except OSError:
        _update_artifact_token(digest, "rendition-tree", "output-unavailable")
        return
    try:
        try:
            rendition_descriptor = os.open(
                "renditions",
                _DIRECTORY_FLAGS,
                dir_fd=output_descriptor,
            )
        except FileNotFoundError:
            _update_artifact_token(digest, "rendition-tree", "absent")
            return
        except OSError:
            _update_artifact_token(digest, "rendition-tree", "unsafe")
            return
        try:
            for namespace in sorted(
                os.scandir(rendition_descriptor),
                key=lambda entry: entry.name,
            ):
                try:
                    namespace_stat = namespace.stat(follow_symlinks=False)
                except OSError:
                    _update_artifact_token(
                        digest,
                        "rendition-namespace",
                        namespace.name,
                        "unreadable",
                    )
                    continue
                _update_artifact_token(
                    digest,
                    "rendition-namespace",
                    namespace.name,
                    _stat_identity(namespace_stat),
                )
                if not stat.S_ISDIR(namespace_stat.st_mode):
                    continue
                try:
                    namespace_descriptor = os.open(
                        namespace.name,
                        _DIRECTORY_FLAGS,
                        dir_fd=rendition_descriptor,
                    )
                except OSError:
                    _update_artifact_token(
                        digest,
                        "rendition-namespace",
                        namespace.name,
                        "replaced",
                    )
                    continue
                try:
                    opened_stat = os.fstat(namespace_descriptor)
                    if _stat_identity(opened_stat) != _stat_identity(namespace_stat):
                        _update_artifact_token(
                            digest,
                            "rendition-namespace",
                            namespace.name,
                            "replaced",
                        )
                        continue
                    for entry in sorted(
                        os.scandir(namespace_descriptor),
                        key=lambda item: item.name,
                    ):
                        _update_rendition_leaf_token(
                            digest,
                            namespace.name,
                            namespace_descriptor,
                            entry,
                        )
                finally:
                    os.close(namespace_descriptor)
        finally:
            os.close(rendition_descriptor)
    finally:
        os.close(output_descriptor)


def _update_rendition_leaf_token(
    digest: _Digest,
    namespace: str,
    directory_descriptor: int,
    entry: os.DirEntry[str],
) -> None:
    try:
        entry_stat = entry.stat(follow_symlinks=False)
    except OSError:
        _update_artifact_token(
            digest,
            "rendition-leaf",
            namespace,
            entry.name,
            "unreadable",
        )
        return
    identity = _stat_identity(entry_stat)
    if not stat.S_ISREG(entry_stat.st_mode):
        _update_artifact_token(
            digest,
            "rendition-leaf",
            namespace,
            entry.name,
            identity,
        )
        return
    descriptor: int | None = None
    try:
        descriptor = os.open(
            entry.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_descriptor,
        )
        opened_stat = os.fstat(descriptor)
        if _stat_identity(opened_stat) != identity:
            raise OSError("rendition leaf replaced")
        leaf_digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            leaf_digest.update(chunk)
    except OSError:
        _update_artifact_token(
            digest,
            "rendition-leaf",
            namespace,
            entry.name,
            "replaced",
        )
    else:
        _update_artifact_token(
            digest,
            "rendition-leaf",
            namespace,
            entry.name,
            identity,
            leaf_digest.hexdigest(),
        )
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_mode,
        value.st_dev,
        value.st_ino,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
    )


def _validate_configurations(
    connection: duckdb.DuckDBPyConnection,
    issues: list[EmbeddingValidationIssue],
) -> None:
    invalid = False
    rows = connection.execute(
        """
        SELECT configuration_id, model, observed_model, observed_api_version,
               task, dimensions, output_type, normalization,
               tokenizer_revision, tokenizer_engine, context_token_limit,
               context_rules, long_text_aggregation,
               long_text_part_attempt_limit, batch_max_items,
               batch_max_estimated_tokens, batch_max_encoded_bytes,
               batch_max_concurrency, batch_max_attempts,
               batch_initial_backoff_seconds, batch_max_backoff_seconds,
               image_input_rules,
               image_transform, multimodal_formula,
               client_configuration_version
        FROM embedding_configurations
        """
    ).fetchall()
    configuration_ids = {row[0] for row in rows}
    for row in rows:
        profile = EmbeddingProfile(
            model=row[1],
            observed_model=row[2],
            observed_api_version=row[3],
            task=row[4],
            dimensions=row[5],
            output_type=row[6],
            normalization=row[7],
            tokenizer_revision=row[8],
            tokenizer_engine=row[9],
            context_token_limit=row[10],
            context_rules=row[11],
            long_text_aggregation=row[12],
            long_text_part_attempt_limit=row[13],
            batch_max_items=row[14],
            batch_max_estimated_tokens=row[15],
            batch_max_encoded_bytes=row[16],
            batch_max_concurrency=row[17],
            batch_max_attempts=row[18],
            batch_initial_backoff_seconds=row[19],
            batch_max_backoff_seconds=row[20],
            image_input_rules=row[21],
            image_transform=row[22],
            multimodal_formula=row[23],
            client_configuration_version=row[24],
        )
        if row[0] != profile_configuration_id(profile):
            invalid = True
    if invalid:
        issues.append(
            EmbeddingValidationIssue(
                "invalid_configuration_id",
                "embedding configurations do not match their profile identity",
            )
        )
    if any(
        row[0] not in configuration_ids
        for row in connection.execute("SELECT configuration_id FROM runs").fetchall()
    ):
        issues.append(
            EmbeddingValidationIssue(
                "missing_run_configuration_reference",
                "embedding runs lack a configuration reference",
            )
        )


def _validate_run_metrics(
    connection: duckdb.DuckDBPyConnection,
    issues: list[EmbeddingValidationIssue],
) -> None:
    invalid = False
    rows = connection.execute(
        """
        SELECT hosted_requests, hosted_retries, usage_json, throttles,
               elapsed_seconds
        FROM runs
        """
    ).fetchall()
    for hosted_requests, hosted_retries, usage_json, throttles, elapsed in rows:
        if any(value < 0 for value in (hosted_requests, hosted_retries, throttles)):
            invalid = True
        if not math.isfinite(float(elapsed)) or elapsed < 0:
            invalid = True
        try:
            usage = json.loads(usage_json)
        except (TypeError, json.JSONDecodeError):
            invalid = True
            continue
        if not safe_usage_is_valid(usage):
            invalid = True
    if invalid:
        _append(
            issues,
            "invalid_run_metrics",
            "embedding run observations are not finite content-free metrics",
        )


def _validate_run_lifecycle(
    connection: duckdb.DuckDBPyConnection,
    issues: list[EmbeddingValidationIssue],
) -> None:
    """Reject contradictory scope, terminal, and full-inventory state."""

    inventory_counts = dict(
        connection.execute(
            """
            SELECT run_id, count(*) FROM full_run_articles GROUP BY run_id
            """
        ).fetchall()
    )
    invalid = bool(
        connection.execute(
            """
            SELECT count(*) FROM full_run_articles AS inventory
            LEFT JOIN runs USING (run_id)
            WHERE runs.run_id IS NULL OR runs.scope <> 'full'
            """
        ).fetchone()[0]
    )
    rows = connection.execute(
        """
        SELECT run_id, scope, status, discovery_complete,
               reconciliation_complete, articles, finished_at
        FROM runs
        """
    ).fetchall()
    for (
        run_id,
        scope,
        status,
        discovery_complete,
        reconciliation_complete,
        articles,
        finished_at,
    ) in rows:
        inventory_count = int(inventory_counts.get(run_id, 0))
        if scope not in {"limited", "full"}:
            invalid = True
            continue
        if status not in {"running", "succeeded", "failed", "interrupted"}:
            invalid = True
            continue
        if status == "running":
            invalid |= finished_at is not None or reconciliation_complete
        else:
            invalid |= finished_at is None
        if status != "succeeded":
            invalid |= reconciliation_complete
        if scope == "limited":
            invalid |= bool(
                discovery_complete or reconciliation_complete or inventory_count
            )
        else:
            invalid |= inventory_count != articles
            if status == "succeeded":
                invalid |= not discovery_complete or not reconciliation_complete
    if invalid:
        _append(
            issues,
            "invalid_run_lifecycle",
            "embedding run scope, terminal state, or full inventory is contradictory",
        )


def _validate_reconciliation_actions(
    connection: duckdb.DuckDBPyConnection,
    issues: list[EmbeddingValidationIssue],
) -> None:
    """Prove pending actions still name their exact physical generation."""

    invalid = connection.execute(
        """
        SELECT count(*)
        FROM reconciliation_actions AS action
        LEFT JOIN runs ON runs.run_id = action.run_id
        LEFT JOIN embedding_work_storage AS work
          ON work.article_id = action.article_id
         AND work.modality = action.modality
         AND work.configuration_id = action.configuration_id
        LEFT JOIN embedding_storage AS vector
          ON vector.article_id = action.article_id
         AND vector.modality = action.modality
         AND vector.configuration_id = action.configuration_id
        WHERE runs.run_id IS NULL OR runs.scope <> 'full'
           OR (runs.reconciliation_complete AND runs.status <> 'succeeded')
           OR (runs.status = 'succeeded' AND NOT runs.reconciliation_complete)
           OR action.target_state NOT IN ('stale_input', 'not_applicable')
           OR work.article_id IS NULL
           OR work.source_relative_path IS DISTINCT FROM
                action.expected_source_relative_path
           OR work.input_sha256 IS DISTINCT FROM action.expected_input_sha256
           OR work.state IS DISTINCT FROM action.expected_state
           OR work.generation_run_id IS DISTINCT FROM
                action.expected_generation_run_id
           OR vector.source_relative_path IS DISTINCT FROM
                action.expected_vector_source_relative_path
           OR vector.input_sha256 IS DISTINCT FROM
                action.expected_vector_input_sha256
           OR vector.stored_vector_sha256 IS DISTINCT FROM
                action.expected_stored_vector_sha256
        """
    ).fetchone()[0]
    if invalid:
        _append(
            issues,
            "invalid_reconciliation_action",
            "reconciliation actions do not match exact physical generation state",
        )


def _select_configuration_id(
    connection: duckdb.DuckDBPyConnection,
    requested: str | None,
    issues: list[EmbeddingValidationIssue],
) -> str | None:
    configuration_ids = tuple(
        row[0]
        for row in connection.execute(
            "SELECT configuration_id FROM embedding_configurations "
            "ORDER BY configuration_id"
        ).fetchall()
    )
    if requested is not None:
        if requested not in configuration_ids:
            _append(
                issues,
                "unknown_configuration_id",
                "validation configuration identity is not present",
            )
            return None
        return requested
    if len(configuration_ids) > 1:
        _append(
            issues,
            "configuration_id_required",
            "validation requires a configuration identity when generations coexist",
        )
        return None
    return configuration_ids[0] if configuration_ids else None


def _coverage_for_configuration(
    connection: duckdb.DuckDBPyConnection,
    articles: dict[str, CanonicalArticle],
    configuration_id: str,
    *,
    orphan_artifacts: int,
) -> EmbeddingCoverage:
    """Account for canonical eligibility and current durable work states."""

    work = {
        (str(article_id), str(modality)): str(state)
        for article_id, modality, state in connection.execute(
            """
            SELECT article_id, modality, state
            FROM embedding_work_items
            WHERE configuration_id = ?
            """,
            [configuration_id],
        ).fetchall()
    }
    text_success = sum(
        work.get((article_id, EmbeddingModality.ARTICLE_TEXT.value))
        == WorkState.SUCCEEDED.value
        for article_id in articles
    )
    header_present_ids = {
        article_id
        for article_id, article in articles.items()
        if article.header_image_path is not None
    }
    header_success = sum(
        work.get((article_id, EmbeddingModality.HEADER_IMAGE.value))
        == WorkState.SUCCEEDED.value
        for article_id in header_present_ids
    )
    multimodal_success = sum(
        work.get((article_id, EmbeddingModality.MULTIMODAL_ARTICLE.value))
        == WorkState.SUCCEEDED.value
        for article_id in header_present_ids
    )
    return EmbeddingCoverage(
        canonical_articles=len(articles),
        text_success=text_success,
        text_failure=len(articles) - text_success,
        header_present=len(header_present_ids),
        header_absent=len(articles) - len(header_present_ids),
        header_success=header_success,
        header_failure=len(header_present_ids) - header_success,
        multimodal_success=multimodal_success,
        multimodal_unavailable=len(articles) - len(header_present_ids),
        multimodal_failure=len(header_present_ids) - multimodal_success,
        stale_work=sum(state == WorkState.STALE_INPUT.value for state in work.values()),
        unresolved_retryable_work=sum(
            state == WorkState.RETRYABLE.value for state in work.values()
        ),
        orphan_artifacts=orphan_artifacts,
    )


def _validate_run_coverage(
    connection: duckdb.DuckDBPyConnection,
    configuration_id: str,
    issues: list[EmbeddingValidationIssue],
) -> None:
    missing = connection.execute(
        """
        WITH run_claims AS (
            SELECT runs.run_id, runs.configuration_id,
                   runs.embeddings AS expected_embeddings
            FROM runs
            JOIN embedding_configurations USING (configuration_id)
            WHERE runs.configuration_id = ?
        ), all_generations AS (
            SELECT e.configuration_id, w.generation_run_id AS run_id
            FROM embeddings AS e
            JOIN embedding_work_items AS w
              USING (article_id, modality, configuration_id)
            WHERE e.configuration_id = ?
            UNION ALL
            SELECT configuration_id, generation_run_id AS run_id
            FROM embedding_generation_history
            WHERE configuration_id = ?
            UNION ALL
            SELECT action.configuration_id,
                   action.expected_generation_run_id AS run_id
            FROM reconciliation_actions AS action
            JOIN runs ON runs.run_id = action.run_id
            WHERE action.configuration_id = ?
              AND runs.reconciliation_complete
              AND action.expected_generation_run_id IS NOT NULL
        ), generation_counts AS (
            SELECT run_id, configuration_id, count(*) AS published_embeddings
            FROM all_generations
            GROUP BY run_id, configuration_id
        )
        SELECT count(*)
        FROM run_claims
        LEFT JOIN generation_counts USING (run_id, configuration_id)
        WHERE coalesce(published_embeddings, 0) != expected_embeddings
        """,
        [configuration_id, configuration_id, configuration_id, configuration_id],
    ).fetchone()[0]
    if missing:
        _append(
            issues,
            "missing_published_embedding",
            "successful embedding runs lack their published vectors",
        )


def _validate_global_generation_history_references(
    connection: duckdb.DuckDBPyConnection,
    issues: list[EmbeddingValidationIssue],
) -> None:
    """Require history to retain its configuration and durable work identity."""

    unsupported_modality = connection.execute(
        """
        SELECT count(*) FROM embedding_generation_history
        WHERE modality NOT IN ('article_text', 'header_image', 'multimodal_article')
        """
    ).fetchone()[0]
    if unsupported_modality:
        _append(
            issues,
            "invalid_generation_modality",
            "embedding generation history uses an unsupported modality",
        )
    invalid = connection.execute(
        """
        SELECT count(*)
        FROM embedding_generation_history AS history
        LEFT JOIN embedding_configurations AS configuration
          ON configuration.configuration_id = history.configuration_id
        LEFT JOIN runs AS generation_run
          ON generation_run.run_id = history.generation_run_id
         AND generation_run.configuration_id = history.configuration_id
        LEFT JOIN embedding_work_storage AS work
          ON work.article_id = history.article_id
         AND work.modality = history.modality
         AND work.configuration_id = history.configuration_id
        WHERE configuration.configuration_id IS NULL
           OR generation_run.run_id IS NULL
           OR work.article_id IS NULL
        """
    ).fetchone()[0]
    if invalid:
        _append(
            issues,
            "invalid_generation_identity_reference",
            "embedding generation history lacks its work or configuration identity",
        )


def _validate_global_public_embedding_checkpoints(
    connection: duckdb.DuckDBPyConnection,
    issues: list[EmbeddingValidationIssue],
) -> None:
    """Require every public vector to have exact reusable-success work."""

    invalid = connection.execute(
        """
        SELECT count(*)
        FROM embeddings AS vector
        LEFT JOIN embedding_work_items AS work
          USING (article_id, modality, configuration_id)
        LEFT JOIN runs AS generation_run
          ON generation_run.run_id = work.generation_run_id
         AND generation_run.configuration_id = work.configuration_id
        WHERE work.article_id IS NULL
           OR work.state <> 'succeeded'
           OR work.source_relative_path IS DISTINCT FROM vector.source_relative_path
           OR work.input_sha256 IS DISTINCT FROM vector.input_sha256
           OR work.generation_run_id IS NULL
           OR generation_run.run_id IS NULL
        """
    ).fetchone()[0]
    if invalid:
        _append(
            issues,
            "invalid_public_embedding_checkpoint",
            "public embeddings lack exact reusable-success work provenance",
        )


def _validate_work_items(
    connection: duckdb.DuckDBPyConnection,
    articles: dict[str, CanonicalArticle],
    configuration_id: str,
    issues: list[EmbeddingValidationIssue],
) -> None:
    embeddings = set(
        connection.execute(
            """
            SELECT article_id, modality, configuration_id,
                   source_relative_path, input_sha256
            FROM embeddings
            WHERE configuration_id = ?
            """,
            [configuration_id],
        ).fetchall()
    )
    embedding_keys = {
        row[:3]
        for row in connection.execute(
            """
            SELECT article_id, modality, configuration_id
            FROM embeddings
            WHERE configuration_id = ?
            """,
            [configuration_id],
        ).fetchall()
    }
    run_ids = {
        row[0]
        for row in connection.execute(
            "SELECT run_id FROM runs WHERE configuration_id = ?",
            [configuration_id],
        ).fetchall()
    }
    rows = connection.execute(
        """
        SELECT article_id, modality, configuration_id, source_relative_path,
               input_sha256, state, attempt_count, error_code, status_code,
               retry_after_seconds, generation_run_id
        FROM embedding_work_items
        WHERE configuration_id = ?
        """,
        [configuration_id],
    ).fetchall()
    for row in rows:
        (
            article_id,
            modality,
            configuration_identifier,
            source_relative_path,
            input_sha256,
            state,
            attempt_count,
            error_code,
            status_code,
            retry_after_seconds,
            generation_run_id,
        ) = row
        try:
            work_state = WorkState(state)
        except ValueError:
            _append(
                issues,
                "invalid_work_state",
                "embedding work items use an unsupported lifecycle state",
            )
            continue
        has_matching_embedding = (
            article_id,
            modality,
            configuration_identifier,
            source_relative_path,
            input_sha256,
        ) in embeddings
        if work_state.is_reusable_success and not has_matching_embedding:
            _append(
                issues,
                "missing_published_embedding",
                "successful embedding runs lack their published vectors",
            )
        elif not work_state.is_reusable_success and has_matching_embedding:
            _append(
                issues,
                "embedding_without_success_checkpoint",
                "embedding rows lack a matching successful work checkpoint",
            )
        failure_metadata_present = any(
            value is not None
            for value in (error_code, status_code, retry_after_seconds)
        )
        is_missing_header = (
            modality == EmbeddingModality.HEADER_IMAGE.value
            and work_state is WorkState.RETRYABLE
            and error_code == "missing_header_image"
            and input_sha256 is None
        )
        if work_state is WorkState.NOT_APPLICABLE:
            if (
                modality != EmbeddingModality.HEADER_IMAGE.value
                or source_relative_path is not None
                or input_sha256 is not None
                or attempt_count != 0
                or failure_metadata_present
                or generation_run_id is not None
            ):
                _append(
                    issues,
                    "invalid_not_applicable_checkpoint",
                    "not-applicable image work retains generation metadata",
                )
            article = articles.get(article_id)
            if article is not None and article.header_image_path is not None:
                _append(
                    issues,
                    "invalid_not_applicable_checkpoint",
                    "not-applicable image work retains generation metadata",
                )
            if (article_id, modality, configuration_identifier) in embedding_keys:
                _append(
                    issues,
                    "embedding_without_success_checkpoint",
                    "embedding rows lack a matching successful work checkpoint",
                )
        elif work_state is WorkState.STALE_INPUT:
            article = articles.get(article_id)
            if (
                input_sha256 is not None
                or attempt_count != 0
                or failure_metadata_present
                or generation_run_id is not None
                or (
                    modality == EmbeddingModality.HEADER_IMAGE.value
                    and (
                        (
                            source_relative_path is not None
                            and not is_safe_source_relative_path(
                                source_relative_path
                            )
                        )
                        or (
                            article is None
                            and source_relative_path is not None
                        )
                        or (
                            article is not None
                            and source_relative_path != article.header_image_path
                        )
                    )
                )
                or (
                    modality != EmbeddingModality.HEADER_IMAGE.value
                    and source_relative_path is not None
                )
            ):
                _append(
                    issues,
                    "invalid_stale_input_checkpoint",
                    "stale input work retains generation or unsafe path metadata",
                )
        elif is_missing_header:
            article = articles.get(article_id)
            if (
                not is_safe_source_relative_path(source_relative_path)
                or article is None
                or source_relative_path != article.header_image_path
                or attempt_count != 0
                or status_code is not None
                or retry_after_seconds is not None
                or generation_run_id is not None
                or (article_id, modality, configuration_identifier)
                in embedding_keys
            ):
                _append(
                    issues,
                    "invalid_missing_header_checkpoint",
                    "missing-header work retains invalid generation metadata",
                )
            _append(
                issues,
                "missing_header_image",
                "canonical header image is unavailable",
            )
        else:
            if not _is_sha256(input_sha256):
                _append(
                    issues,
                    "invalid_work_input_hash",
                    "applicable embedding work lacks a valid input hash",
                )
            if modality == EmbeddingModality.HEADER_IMAGE.value:
                if not is_safe_source_relative_path(source_relative_path):
                    _append(
                        issues,
                        "invalid_source_relative_path",
                        "embedding work uses an unsafe source-relative path",
                    )
            elif source_relative_path is not None:
                _append(
                    issues,
                    "invalid_source_relative_path",
                    "only header-image work may retain a source-relative path",
                )
            if work_state.is_reusable_success:
                if generation_run_id not in run_ids:
                    _append(
                        issues,
                        "invalid_generation_checkpoint",
                        "successful work lacks its generating run identity",
                    )
            elif generation_run_id is not None:
                _append(
                    issues,
                    "invalid_generation_checkpoint",
                    "non-successful work retains a generating run identity",
                )
        if work_state.is_failure:
            local_image_terminal = (
                modality == EmbeddingModality.HEADER_IMAGE.value
                and work_state is WorkState.TERMINAL
                and error_code
                in {
                    "corrupt_image",
                    "derived_image_oversized",
                    "image_encode_failure",
                    "unsafe_image",
                    "unsupported_image",
                }
                and attempt_count == 0
            )
            if not is_missing_header and not local_image_terminal and (
                error_code is None or attempt_count < 1
            ):
                _append(
                    issues,
                    "invalid_failure_checkpoint",
                    "failed embedding work items lack stable attempt metadata",
                )
        elif failure_metadata_present:
            _append(
                issues,
                "invalid_failure_checkpoint",
                "non-failed embedding work items retain failure metadata",
            )
        if modality == EmbeddingModality.HEADER_IMAGE.value and not is_missing_header:
            issue_by_state = {
                WorkState.QUEUED: (
                    "header_image_queued",
                    "header-image work remains queued",
                ),
                WorkState.IN_PROGRESS: (
                    "header_image_in_progress",
                    "header-image work remains in progress",
                ),
                WorkState.INTERRUPTED: (
                    "header_image_interrupted",
                    "header-image work remains interrupted",
                ),
                WorkState.RETRYABLE: (
                    "header_image_retryable_failure",
                    "header-image work has a retryable failure",
                ),
                WorkState.TERMINAL: (
                    "header_image_terminal_failure",
                    "header-image work has a terminal failure",
                ),
            }
            issue = issue_by_state.get(work_state)
            if issue is not None:
                _append(issues, *issue)


def _validate_header_disposition_coverage(
    connection: duckdb.DuckDBPyConnection,
    articles: dict[str, CanonicalArticle],
    configuration_id: str,
    issues: list[EmbeddingValidationIssue],
) -> None:
    """Require one header disposition per selected article and honest run claims."""

    selected_article_ids = {
        row[0]
        for row in connection.execute(
            """
            SELECT article_id
            FROM embedding_work_items
            WHERE configuration_id = ? AND modality = 'article_text'
            """,
            [configuration_id],
        ).fetchall()
        if row[0] in articles
    }
    header_article_ids = {
        row[0]
        for row in connection.execute(
            """
            SELECT article_id
            FROM embedding_work_items
            WHERE configuration_id = ? AND modality = 'header_image'
            """,
            [configuration_id],
        ).fetchall()
    }
    if selected_article_ids - header_article_ids:
        _append(
            issues,
            "missing_header_disposition",
            "selected articles lack an explicit header-image disposition",
        )

    run = connection.execute(
        """
        SELECT run_id, header_absent, header_failed
        FROM runs
        WHERE configuration_id = ?
        ORDER BY started_at DESC, run_id DESC
        LIMIT 1
        """,
        [configuration_id],
    ).fetchone()
    if run is None:
        return
    run_id, claimed_absent, claimed_failed = run
    raw_states = connection.execute(
        """
        SELECT state
        FROM embedding_work_items
        WHERE configuration_id = ? AND modality = 'header_image'
          AND last_run_id = ?
        """,
        [configuration_id, run_id],
    ).fetchall()
    try:
        states = [WorkState(row[0]) for row in raw_states]
    except ValueError:
        return
    observed_absent = sum(state is WorkState.NOT_APPLICABLE for state in states)
    observed_failed = sum(state.is_failure for state in states)
    if (claimed_absent, claimed_failed) != (observed_absent, observed_failed):
        _append(
            issues,
            "invalid_header_disposition_counts",
            "run header-disposition counts differ from durable work",
        )


def _validate_embeddings(
    connection: duckdb.DuckDBPyConnection,
    config: EmbeddingPipelineConfig,
    articles: dict[str, CanonicalArticle],
    selected_configuration_id: str,
    issues: list[EmbeddingValidationIssue],
) -> None:
    rows = connection.execute(
        """
        SELECT article_id, modality, configuration_id, published_at_utc,
               source_relative_path, publication_date_new_york, dimensions,
               input_sha256, stored_vector_sha256, vector
        FROM embeddings
        WHERE configuration_id = ?
        ORDER BY article_id, modality, configuration_id
        """,
        [selected_configuration_id],
    ).fetchall()
    configuration_ids = {
        row[0]
        for row in connection.execute(
            "SELECT configuration_id FROM embedding_configurations"
        ).fetchall()
    }
    for row in rows:
        (
            article_id,
            modality,
            configuration_identifier,
            published_at_utc,
            source_relative_path,
            publication_date,
            dimensions,
            input_sha256,
            stored_vector_sha256,
            vector,
        ) = row
        article = articles.get(article_id)
        if configuration_identifier not in configuration_ids:
            _append(
                issues,
                "missing_configuration_reference",
                "embedding rows lack a configuration reference",
            )
        if modality not in {
            EmbeddingModality.ARTICLE_TEXT.value,
            EmbeddingModality.HEADER_IMAGE.value,
            EmbeddingModality.MULTIMODAL_ARTICLE.value,
        }:
            _append(
                issues,
                "invalid_modality",
                "embedding rows use an unsupported modality",
            )
        if modality == EmbeddingModality.HEADER_IMAGE.value:
            if not is_safe_source_relative_path(source_relative_path):
                _append(
                    issues,
                    "invalid_source_relative_path",
                    "header-image embeddings use an unsafe source-relative path",
                )
        elif source_relative_path is not None:
            _append(
                issues,
                "invalid_source_relative_path",
                "only header-image embeddings may retain a source-relative path",
            )
        if dimensions != _VECTOR_DIMENSIONS:
            _append(
                issues,
                "invalid_dimensions",
                "embedding rows use an unsupported dimension",
            )
        _validate_vector(vector, stored_vector_sha256, issues)
        if article is None:
            _append(
                issues,
                "missing_canonical_article",
                "embedding rows lack a canonical article",
            )
            continue
        if (
            published_at_utc != article.published_at_utc
            or publication_date != article.publication_date_new_york
        ):
            _append(
                issues,
                "publication_metadata_mismatch",
                "embedding publication metadata differs from the canonical article",
            )
        if modality == EmbeddingModality.ARTICLE_TEXT.value:
            _validate_markdown_hash(config, article, input_sha256, issues)
        elif modality == EmbeddingModality.HEADER_IMAGE.value:
            _validate_header_image_hash(
                config,
                article,
                source_relative_path,
                input_sha256,
                issues,
            )


def _validate_image_input_provenance(
    connection: duckdb.DuckDBPyConnection,
    config: EmbeddingPipelineConfig,
    articles: dict[str, CanonicalArticle],
    configuration_id: str,
    issues: list[EmbeddingValidationIssue],
    image_codec: ImageCodec | None,
) -> None:
    """Resolve every image generation to the exact bytes supplied to Jina."""

    generation_rows = connection.execute(
        """
        SELECT w.article_id, w.generation_run_id, e.input_sha256
        FROM embedding_work_items AS w
        JOIN embeddings AS e USING (article_id, modality, configuration_id)
        WHERE w.configuration_id = ? AND w.modality = 'header_image'
          AND w.state = 'succeeded' AND w.generation_run_id IS NOT NULL
        UNION ALL
        SELECT article_id, generation_run_id, input_sha256
        FROM embedding_generation_history
        WHERE configuration_id = ? AND modality = 'header_image'
        UNION ALL
        SELECT action.article_id, action.expected_generation_run_id,
               action.expected_vector_input_sha256
        FROM reconciliation_actions AS action
        JOIN runs ON runs.run_id = action.run_id
        WHERE action.configuration_id = ?
          AND action.modality = 'header_image'
          AND runs.reconciliation_complete
          AND action.expected_generation_run_id IS NOT NULL
        """,
        [configuration_id, configuration_id, configuration_id],
    ).fetchall()
    expected_generations = {
        (str(article_id), str(generation_run_id), str(source_sha256))
        for article_id, generation_run_id, source_sha256 in generation_rows
    }
    active_generations = {
        (str(article_id), str(generation_run_id))
        for article_id, generation_run_id in connection.execute(
            """
            SELECT article_id, generation_run_id
            FROM embedding_work_items
            WHERE configuration_id = ? AND modality = 'header_image'
              AND state = 'succeeded' AND generation_run_id IS NOT NULL
            """,
            [configuration_id],
        ).fetchall()
    }
    configured_input_rules, configured_transform = connection.execute(
        """
        SELECT image_input_rules, image_transform FROM embedding_configurations
        WHERE configuration_id = ?
        """,
        [configuration_id],
    ).fetchone()
    if image_codec is not None and (
        image_codec.input_rules != configured_input_rules
        or image_codec.transform_id != configured_transform
    ):
        _append(
            issues,
            "invalid_image_input_provenance",
            "image decoder differs from configuration identity",
        )
        image_codec = None
    rows = connection.execute(
        """
        SELECT article_id, generation_run_id, source_sha256, source_format,
               source_bytes, source_width, source_height, source_frames,
               embedded_input_sha256, embedded_format, embedded_bytes,
               embedded_width, embedded_height, embedded_frames, transform_id,
               rendition_relative_path
        FROM image_input_provenance
        WHERE configuration_id = ?
        """,
        [configuration_id],
    ).fetchall()
    observed_generations: set[tuple[str, str, str]] = set()
    for row in rows:
        (
            article_id,
            generation_run_id,
            source_sha256,
            source_format,
            source_bytes,
            source_width,
            source_height,
            source_frames,
            embedded_sha256,
            embedded_format,
            embedded_bytes,
            embedded_width,
            embedded_height,
            embedded_frames,
            transform_id,
            rendition_relative_path,
        ) = row
        generation = (
            str(article_id),
            str(generation_run_id),
            str(source_sha256),
        )
        observed_generations.add(generation)
        if generation not in expected_generations:
            _append(
                issues,
                "invalid_image_input_provenance",
                "image input provenance lacks a matching vector generation",
            )
        if (
            not _is_sha256(source_sha256)
            or not _is_sha256(embedded_sha256)
            or source_format not in {"JPEG", "PNG", "WEBP"}
            or embedded_format not in {"JPEG", "PNG", "WEBP"}
            or source_bytes < 1
            or embedded_bytes < 1
            or source_width < 1
            or source_height < 1
            or source_frames != 1
            or source_width * source_height > MAX_SAFE_DECODE_PIXELS
            or embedded_width < 1
            or embedded_height < 1
            or embedded_frames != 1
            or embedded_bytes > 5_000_000
            or embedded_width * embedded_height > 20_000_000
        ):
            _append(
                issues,
                "invalid_image_input_provenance",
                "image input provenance contains invalid content-free facts",
            )
        if source_width * source_height > MAX_SAFE_DECODE_PIXELS:
            _append(
                issues,
                "invalid_image_input_provenance",
                "source image exceeds the safe decode pixel ceiling",
            )
        source_data: bytes | None = None
        decoded_source = None
        if (
            image_codec is not None
            and (str(article_id), str(generation_run_id)) in active_generations
        ):
            article = articles.get(str(article_id))
            try:
                if article is None or article.header_image_path is None:
                    raise SourceImageError("missing")
                source_data = read_source_image(
                    config.source_root,
                    article.header_image_path,
                )
                decoded_source = image_codec.inspect(
                    source_data,
                    max_decode_pixels=MAX_SAFE_DECODE_PIXELS,
                )
            except (ImageCodecError, SourceImageError):
                _append(
                    issues,
                    "invalid_image_input_provenance",
                    "source image cannot prove declared provenance facts",
                )
            else:
                if (
                    hashlib.sha256(source_data).hexdigest() != source_sha256
                    or len(source_data) != source_bytes
                    or decoded_source.format != source_format
                    or decoded_source.width != source_width
                    or decoded_source.height != source_height
                    or decoded_source.frames != source_frames
                ):
                    _append(
                        issues,
                        "invalid_image_input_provenance",
                        "decoded source image differs from declared provenance facts",
                    )
        if rendition_relative_path is None:
            if (
                transform_id != SOURCE_BYTES_TRANSFORM_ID
                or source_sha256 != embedded_sha256
                or source_format != embedded_format
                or source_bytes != embedded_bytes
                or source_width != embedded_width
                or source_height != embedded_height
            ):
                _append(
                    issues,
                    "invalid_image_input_provenance",
                    "source-byte image provenance does not identify exact input",
                )
            if source_data is not None and (
                hashlib.sha256(source_data).hexdigest() != embedded_sha256
                or len(source_data) != embedded_bytes
            ):
                _append(
                    issues,
                    "invalid_image_input_provenance",
                    "source-byte image does not match exact embedded input",
                )
            continue
        if embedded_format != "JPEG":
            _append(
                issues,
                "invalid_image_input_provenance",
                "derived image input does not declare the JPEG contract",
            )
        if (
            source_bytes <= MAX_HOSTED_IMAGE_BYTES
            and source_width * source_height <= MAX_HOSTED_IMAGE_PIXELS
        ):
            _append(
                issues,
                "invalid_image_input_provenance",
                "within-limit supported source image was not passed through",
            )
        expected_dimensions: tuple[int, int] | None = None
        try:
            expected_dimensions = scaled_image_dimensions(
                source_width,
                source_height,
            )
        except ImageCodecError:
            _append(
                issues,
                "invalid_image_input_provenance",
                "derived image dimensions cannot be scaled from malformed source facts",
            )
        if expected_dimensions is not None and (
            embedded_width,
            embedded_height,
        ) != expected_dimensions:
            _append(
                issues,
                "invalid_image_input_provenance",
                "derived image dimensions differ from deterministic aspect scale",
            )
        transform_namespace = hashlib.sha256(str(transform_id).encode()).hexdigest()
        if transform_id != configured_transform:
            _append(
                issues,
                "invalid_image_input_provenance",
                "image rendition transform differs from configuration identity",
            )
        expected_path = (
            f"renditions/{transform_namespace}/{embedded_sha256}.jpg"
        )
        if rendition_relative_path != expected_path:
            _append(
                issues,
                "invalid_rendition_path",
                "image rendition path does not match its versioned identity",
            )
            continue
        try:
            rendition = read_source_image(
                config.embedding_output_root,
                rendition_relative_path,
            )
        except SourceImageError as error:
            _append(
                issues,
                (
                    "missing_image_rendition"
                    if error.status == "missing"
                    else "unsafe_image_rendition"
                ),
                (
                    "image rendition is unavailable"
                    if error.status == "missing"
                    else "image rendition path is unsafe"
                ),
            )
            continue
        if (
            len(rendition) != embedded_bytes
            or hashlib.sha256(rendition).hexdigest() != embedded_sha256
        ):
            _append(
                issues,
                "rendition_hash_mismatch",
                "image rendition bytes differ from embedded input provenance",
            )
        if image_codec is not None:
            try:
                decoded_embedded = image_codec.inspect(
                    rendition,
                    max_decode_pixels=MAX_SAFE_DECODE_PIXELS,
                )
            except ImageCodecError:
                _append(
                    issues,
                    "invalid_image_input_provenance",
                    "rendition cannot prove declared embedded-input facts",
                )
            else:
                if (
                    embedded_format != "JPEG"
                    or decoded_embedded.format != "JPEG"
                    or decoded_embedded.width != embedded_width
                    or decoded_embedded.height != embedded_height
                    or decoded_embedded.frames != embedded_frames
                    or decoded_embedded.pixels > MAX_HOSTED_IMAGE_PIXELS
                    or len(rendition) > MAX_HOSTED_IMAGE_BYTES
                ):
                    _append(
                        issues,
                        "invalid_image_input_provenance",
                        "decoded rendition differs from the hosted JPEG contract",
                    )
        if (
            decoded_source is not None
            and source_data is not None
            and len(source_data) <= MAX_HOSTED_IMAGE_BYTES
            and decoded_source.pixels <= MAX_HOSTED_IMAGE_PIXELS
        ):
            _append(
                issues,
                "invalid_image_input_provenance",
                "within-limit supported source image was not passed through",
            )
    if expected_generations - observed_generations:
        _append(
            issues,
            "missing_image_input_provenance",
            "header-image generations lack exact input provenance",
        )


def _resolve_validation_image_codec(
    connection: duckdb.DuckDBPyConnection,
    configuration_id: str,
    supplied: ImageCodec | None,
    issues: list[EmbeddingValidationIssue],
) -> ImageCodec | None:
    if supplied is not None:
        return supplied
    provenance_count = connection.execute(
        """
        SELECT count(*) FROM image_input_provenance
        WHERE configuration_id = ?
        """,
        [configuration_id],
    ).fetchone()[0]
    if not provenance_count:
        return None
    input_rules, transform_id = connection.execute(
        """
        SELECT image_input_rules, image_transform
        FROM embedding_configurations
        WHERE configuration_id = ?
        """,
        [configuration_id],
    ).fetchone()
    if (
        input_rules == PRODUCTION_IMAGE_INPUT_RULES
        and transform_id == PRODUCTION_IMAGE_TRANSFORM_ID
    ):
        return FixturePassthroughImageCodec()
    if (
        input_rules == PRODUCTION_IMAGE_INPUT_RULES
        and str(transform_id).startswith(f"{PRODUCTION_IMAGE_TRANSFORM_ID}-build-")
    ):
        try:
            codec = PillowImageCodec()
        except ImageCodecError:
            codec = None
        if codec is not None and codec.transform_id == transform_id:
            return codec
    _append(
        issues,
        "image_codec_unavailable",
        "configured image provenance cannot be decoded locally",
    )
    return None


def _validate_global_image_provenance_references(
    connection: duckdb.DuckDBPyConnection,
    issues: list[EmbeddingValidationIssue],
) -> None:
    unknown = connection.execute(
        """
        SELECT count(*)
        FROM image_input_provenance AS p
        LEFT JOIN embedding_configurations AS c USING (configuration_id)
        WHERE c.configuration_id IS NULL
        """
    ).fetchone()[0]
    if unknown:
        _append(
            issues,
            "unknown_image_provenance_configuration",
            "image input provenance references an unknown configuration",
        )


def _scan_rendition_artifacts(
    connection: duckdb.DuckDBPyConnection,
    config: EmbeddingPipelineConfig,
    issues: list[EmbeddingValidationIssue],
) -> int:
    orphan_count = 0

    def unsafe_entry() -> None:
        _append(
            issues,
            "unsafe_image_rendition_entry",
            "rendition namespace contains an unsafe entry",
        )

    try:
        output_descriptor = os.open(config.embedding_output_root, _DIRECTORY_FLAGS)
    except OSError:
        return 0
    try:
        try:
            rendition_descriptor = os.open(
                "renditions",
                _DIRECTORY_FLAGS,
                dir_fd=output_descriptor,
            )
        except FileNotFoundError:
            return 0
        except OSError:
            unsafe_entry()
            return 0
        try:
            with os.scandir(rendition_descriptor) as namespaces:
                for namespace in namespaces:
                    if namespace.is_symlink() or not namespace.is_dir(
                        follow_symlinks=False
                    ):
                        unsafe_entry()
                        continue
                    namespace_stat = namespace.stat(follow_symlinks=False)
                    try:
                        namespace_descriptor = os.open(
                            namespace.name,
                            _DIRECTORY_FLAGS,
                            dir_fd=rendition_descriptor,
                        )
                    except OSError:
                        unsafe_entry()
                        continue
                    try:
                        opened_stat = os.fstat(namespace_descriptor)
                        if (
                            not stat.S_ISDIR(opened_stat.st_mode)
                            or (namespace_stat.st_dev, namespace_stat.st_ino)
                            != (opened_stat.st_dev, opened_stat.st_ino)
                        ):
                            unsafe_entry()
                            continue
                        with os.scandir(namespace_descriptor) as entries:
                            for entry in entries:
                                if entry.is_symlink() or not entry.is_file(
                                    follow_symlinks=False
                                ):
                                    unsafe_entry()
                                    continue
                                relative_path = (
                                    f"renditions/{namespace.name}/{entry.name}"
                                )
                                if connection.execute(
                                    """
                                    SELECT 1 FROM image_input_provenance
                                    WHERE rendition_relative_path = ? LIMIT 1
                                    """,
                                    [relative_path],
                                ).fetchone() is not None:
                                    continue
                                orphan_count += 1
                                _append(
                                    issues,
                                    (
                                        "orphan_image_rendition_temp"
                                        if entry.name.endswith(".tmp")
                                        else "orphan_image_rendition"
                                    ),
                                    (
                                        "unreferenced rendition staging "
                                        "file remains"
                                        if entry.name.endswith(".tmp")
                                        else "unreferenced rendition file remains"
                                    ),
                                )
                    finally:
                        os.close(namespace_descriptor)
        finally:
            os.close(rendition_descriptor)
    finally:
        os.close(output_descriptor)
    return orphan_count


def _validate_multimodal_provenance(
    connection: duckdb.DuckDBPyConnection,
    configuration_id: str,
    issues: list[EmbeddingValidationIssue],
) -> None:
    """Recompute active composites and verify immutable generation linkage."""

    formula_row = connection.execute(
        """
        SELECT multimodal_formula
        FROM embedding_configurations
        WHERE configuration_id = ?
        """,
        [configuration_id],
    ).fetchone()
    if formula_row is None:
        return
    configuration_formula = str(formula_row[0])
    provenance_rows = connection.execute(
        """
        SELECT article_id, generation_run_id, text_generation_run_id,
               text_stored_vector_sha256, header_image_generation_run_id,
               header_image_stored_vector_sha256, formula_version
        FROM multimodal_embedding_provenance
        WHERE configuration_id = ?
        """,
        [configuration_id],
    ).fetchall()
    provenance_by_generation = {
        (str(row[0]), str(row[1])): tuple(row[2:]) for row in provenance_rows
    }
    composite_generations = connection.execute(
        """
        SELECT e.article_id, w.generation_run_id, e.input_sha256
        FROM embeddings AS e
        JOIN embedding_work_items AS w
          USING (article_id, modality, configuration_id)
        WHERE e.configuration_id = ? AND e.modality = 'multimodal_article'
        UNION ALL
        SELECT article_id, generation_run_id, input_sha256
        FROM embedding_generation_history
        WHERE configuration_id = ? AND modality = 'multimodal_article'
        UNION ALL
        SELECT action.article_id, action.expected_generation_run_id,
               action.expected_vector_input_sha256
        FROM reconciliation_actions AS action
        JOIN runs ON runs.run_id = action.run_id
        WHERE action.configuration_id = ?
          AND action.modality = 'multimodal_article'
          AND runs.reconciliation_complete
          AND action.expected_generation_run_id IS NOT NULL
        """,
        [configuration_id, configuration_id, configuration_id],
    ).fetchall()
    for article_id, generation_run_id, input_sha256 in composite_generations:
        provenance = provenance_by_generation.get(
            (str(article_id), str(generation_run_id))
        )
        if provenance is None:
            _append(
                issues,
                "missing_multimodal_provenance",
                "multimodal generations lack immutable source provenance",
            )
            continue
        (
            text_generation_run_id,
            text_vector_sha256,
            image_generation_run_id,
            image_vector_sha256,
            formula_version,
        ) = provenance
        expected_input_sha256 = _multimodal_input_sha256(
            formula_version=str(formula_version),
            text_generation_run_id=str(text_generation_run_id),
            text_stored_vector_sha256=str(text_vector_sha256),
            header_image_generation_run_id=str(image_generation_run_id),
            header_image_stored_vector_sha256=str(image_vector_sha256),
        )
        if input_sha256 != expected_input_sha256:
            _append(
                issues,
                "multimodal_input_identity_mismatch",
                "multimodal input identity differs from immutable provenance",
            )
    for row in provenance_rows:
        (
            article_id,
            composite_generation_run_id,
            text_generation_run_id,
            text_vector_sha256,
            image_generation_run_id,
            image_vector_sha256,
            formula_version,
        ) = row
        provenance_valid = (
            formula_version == configuration_formula == _MULTIMODAL_FORMULA
            and _is_sha256(text_vector_sha256)
            and _is_sha256(image_vector_sha256)
            and _generation_reference_exists(
                connection,
                article_id=str(article_id),
                modality=EmbeddingModality.MULTIMODAL_ARTICLE.value,
                configuration_id=configuration_id,
                generation_run_id=str(composite_generation_run_id),
                stored_vector_sha256=None,
            )
        )
        if not provenance_valid:
            _append(
                issues,
                "invalid_multimodal_provenance",
                "multimodal provenance is incomplete or inconsistent",
            )
        if not all(
            (
                _generation_reference_exists(
                    connection,
                    article_id=str(article_id),
                    modality=modality,
                    configuration_id=configuration_id,
                    generation_run_id=str(generation_run_id),
                    stored_vector_sha256=str(vector_sha256),
                )
                for modality, generation_run_id, vector_sha256 in (
                    (
                        EmbeddingModality.ARTICLE_TEXT.value,
                        text_generation_run_id,
                        text_vector_sha256,
                    ),
                    (
                        EmbeddingModality.HEADER_IMAGE.value,
                        image_generation_run_id,
                        image_vector_sha256,
                    ),
                )
            )
        ):
            _append(
                issues,
                "invalid_multimodal_source_linkage",
                "multimodal provenance does not identify its source generations",
            )

    active_rows = connection.execute(
        """
        SELECT composite.article_id, composite.vector,
               composite.stored_vector_sha256, composite_work.generation_run_id,
               text_work.generation_run_id, text.stored_vector_sha256, text.vector,
               image_work.generation_run_id, image.stored_vector_sha256, image.vector
        FROM embeddings AS composite
        JOIN embedding_work_items AS composite_work
          USING (article_id, modality, configuration_id)
        LEFT JOIN embedding_work_items AS text_work
          ON text_work.article_id = composite.article_id
         AND text_work.configuration_id = composite.configuration_id
         AND text_work.modality = 'article_text'
         AND text_work.state = 'succeeded'
        LEFT JOIN embeddings AS text
          ON text.article_id = text_work.article_id
         AND text.configuration_id = text_work.configuration_id
         AND text.modality = text_work.modality
         AND text.input_sha256 = text_work.input_sha256
        LEFT JOIN embedding_work_items AS image_work
          ON image_work.article_id = composite.article_id
         AND image_work.configuration_id = composite.configuration_id
         AND image_work.modality = 'header_image'
         AND image_work.state = 'succeeded'
        LEFT JOIN embeddings AS image
          ON image.article_id = image_work.article_id
         AND image.configuration_id = image_work.configuration_id
         AND image.modality = image_work.modality
         AND image.input_sha256 = image_work.input_sha256
        WHERE composite.configuration_id = ?
          AND composite.modality = 'multimodal_article'
        """,
        [configuration_id],
    ).fetchall()
    for row in active_rows:
        (
            article_id,
            composite_values,
            composite_vector_sha256,
            composite_generation_run_id,
            text_generation_run_id,
            text_vector_sha256,
            text_values,
            image_generation_run_id,
            image_vector_sha256,
            image_values,
        ) = row
        provenance = provenance_by_generation.get(
            (str(article_id), str(composite_generation_run_id))
        )
        expected_linkage = (
            str(text_generation_run_id),
            str(text_vector_sha256),
            str(image_generation_run_id),
            str(image_vector_sha256),
            configuration_formula,
        )
        if (
            provenance is None
            or None
            in (
                text_generation_run_id,
                text_vector_sha256,
                text_values,
                image_generation_run_id,
                image_vector_sha256,
                image_values,
            )
            or provenance != expected_linkage
        ):
            _append(
                issues,
                "invalid_multimodal_source_linkage",
                "multimodal provenance does not identify its source generations",
            )
            continue
        try:
            text_vector = _verified_source_vector(
                text_values,
                str(text_vector_sha256),
                str(article_id),
            )
            image_vector = _verified_source_vector(
                image_values,
                str(image_vector_sha256),
                str(article_id),
            )
            expected_vector = _multimodal_vector(
                text_vector,
                image_vector,
                str(article_id),
            )
        except EmbeddingPipelineError:
            _append(
                issues,
                "multimodal_recomputation_failed",
                "multimodal vector cannot be recomputed from its sources",
            )
            continue
        if (
            tuple(composite_values) != expected_vector
            or composite_vector_sha256 != _vector_hash(expected_vector)
        ):
            _append(
                issues,
                "multimodal_vector_mismatch",
                "multimodal vector differs from its normalized source midpoint",
            )


def _validate_long_text_parts(
    connection: duckdb.DuckDBPyConnection,
    config: EmbeddingPipelineConfig,
    articles: dict[str, CanonicalArticle],
    configuration_id: str,
    issues: list[EmbeddingValidationIssue],
) -> None:
    """Verify current checkpoints and every active or archived aggregate."""

    configuration = connection.execute(
        """
        SELECT context_token_limit, long_text_aggregation,
               long_text_part_attempt_limit
        FROM embedding_configurations
        WHERE configuration_id = ?
        """,
        [configuration_id],
    ).fetchone()
    if configuration is None:
        return
    token_limit, configured_aggregation, attempt_limit = configuration
    run_ids = {
        row[0]
        for row in connection.execute(
            "SELECT run_id FROM runs WHERE configuration_id = ?",
            [configuration_id],
        ).fetchall()
    }
    part_rows = connection.execute(
        """
        SELECT article_id, article_input_sha256, part_index, part_count,
               char_start, char_end, byte_start, byte_end, part_input_sha256,
               token_count, state, attempt_count, error_code, status_code,
               retry_after_seconds, last_run_id, generation_run_id,
               stored_vector_sha256, vector
        FROM long_text_parts
        WHERE configuration_id = ?
        ORDER BY article_id, article_input_sha256, part_index
        """,
        [configuration_id],
    ).fetchall()
    current_parts: dict[tuple[str, str, int], tuple[object, ...]] = {}
    for row in part_rows:
        (
            article_id,
            article_input_sha256,
            part_index,
            part_count,
            char_start,
            char_end,
            byte_start,
            byte_end,
            part_input_sha256,
            part_token_count,
            raw_state,
            attempt_count,
            error_code,
            status_code,
            retry_after_seconds,
            last_run_id,
            generation_run_id,
            vector_sha256,
            vector,
        ) = row
        try:
            state = WorkState(str(raw_state))
        except ValueError:
            state = None
        failure_metadata = (error_code, status_code, retry_after_seconds)
        identity_valid = (
            _is_sha256(article_input_sha256)
            and _is_sha256(part_input_sha256)
            and isinstance(part_index, int)
            and isinstance(part_count, int)
            and 0 <= part_index < part_count
            and part_count > 1
            and _valid_part_span(char_start, char_end, byte_start, byte_end)
            and isinstance(part_token_count, int)
            and 0 < part_token_count <= token_limit
            and isinstance(attempt_count, int)
            and 0 <= attempt_count <= attempt_limit
            and last_run_id in run_ids
            and state is not None
            and state is not WorkState.NOT_APPLICABLE
        )
        if state is WorkState.SUCCEEDED:
            state_valid = (
                attempt_count >= 1
                and not any(value is not None for value in failure_metadata)
                and generation_run_id in run_ids
                and _is_sha256(vector_sha256)
                and vector is not None
            )
            if state_valid:
                issue_count = len(issues)
                _validate_vector(vector, vector_sha256, issues)
                state_valid = len(issues) == issue_count
        else:
            state_valid = (
                generation_run_id is None
                and vector_sha256 is None
                and vector is None
                and (
                    state is None
                    or not state.is_failure
                    or (attempt_count >= 1 and error_code is not None)
                )
                and (
                    state is not WorkState.RETRYABLE
                    or attempt_count < attempt_limit
                )
                and (
                    state is None
                    or state.is_failure
                    or not any(value is not None for value in failure_metadata)
                )
            )
        if not identity_valid or not state_valid:
            _append(
                issues,
                "invalid_long_text_part",
                "long-text part checkpoints are incomplete or inconsistent",
            )
        current_parts[(str(article_id), str(article_input_sha256), int(part_index))] = (
            part_count,
            char_start,
            char_end,
            byte_start,
            byte_end,
            part_input_sha256,
            part_token_count,
            state,
            generation_run_id,
            vector_sha256,
            vector,
        )

    generation_rows = connection.execute(
        """
        SELECT article_id, article_input_sha256, part_index, generation_run_id,
               part_count, char_start, char_end, byte_start, byte_end,
               part_input_sha256, token_count, stored_vector_sha256, vector
        FROM long_text_part_generations
        WHERE configuration_id = ?
        ORDER BY article_id, article_input_sha256, part_index, generation_run_id
        """,
        [configuration_id],
    ).fetchall()
    generations: dict[tuple[str, str, int, str], tuple[object, ...]] = {}
    for row in generation_rows:
        (
            article_id,
            article_input_sha256,
            part_index,
            generation_run_id,
            part_count,
            char_start,
            char_end,
            byte_start,
            byte_end,
            part_input_sha256,
            part_token_count,
            vector_sha256,
            vector,
        ) = row
        valid = (
            _is_sha256(article_input_sha256)
            and _is_sha256(part_input_sha256)
            and isinstance(part_index, int)
            and isinstance(part_count, int)
            and 0 <= part_index < part_count
            and part_count > 1
            and _valid_part_span(char_start, char_end, byte_start, byte_end)
            and isinstance(part_token_count, int)
            and 0 < part_token_count <= token_limit
            and generation_run_id in run_ids
            and _is_sha256(vector_sha256)
            and vector is not None
        )
        if valid:
            issue_count = len(issues)
            _validate_vector(vector, vector_sha256, issues)
            valid = len(issues) == issue_count
        if not valid:
            _append(
                issues,
                "invalid_long_text_part_generation",
                "long-text part generation history is incomplete or inconsistent",
            )
        generations[
            (
                str(article_id),
                str(article_input_sha256),
                int(part_index),
                str(generation_run_id),
            )
        ] = (
            part_count,
            char_start,
            char_end,
            byte_start,
            byte_end,
            part_input_sha256,
            part_token_count,
            vector_sha256,
            vector,
        )

    for key, current in current_parts.items():
        (
            part_count,
            char_start,
            char_end,
            byte_start,
            byte_end,
            part_input_sha256,
            part_token_count,
            state,
            generation_run_id,
            vector_sha256,
            vector,
        ) = current
        if state is not WorkState.SUCCEEDED:
            continue
        generation = generations.get((*key, str(generation_run_id)))
        if generation != (
            part_count,
            char_start,
            char_end,
            byte_start,
            byte_end,
            part_input_sha256,
            part_token_count,
            vector_sha256,
            vector,
        ):
            _append(
                issues,
                "invalid_long_text_part",
                "long-text part checkpoints are incomplete or inconsistent",
            )

    aggregates = connection.execute(
        """
        SELECT e.article_id, e.input_sha256, e.vector,
               e.stored_vector_sha256, w.generation_run_id, true AS active
        FROM embeddings AS e
        JOIN embedding_work_items AS w
          USING (article_id, modality, configuration_id)
        WHERE e.configuration_id = ? AND e.modality = 'article_text'
          AND w.state = 'succeeded'
        UNION ALL
        SELECT article_id, input_sha256, NULL AS vector,
               stored_vector_sha256, generation_run_id, false AS active
        FROM embedding_generation_history
        WHERE configuration_id = ? AND modality = 'article_text'
        UNION ALL
        SELECT action.article_id, action.expected_vector_input_sha256,
               NULL AS vector, action.expected_stored_vector_sha256,
               action.expected_generation_run_id, false AS active
        FROM reconciliation_actions AS action
        JOIN runs ON runs.run_id = action.run_id
        WHERE action.configuration_id = ?
          AND action.modality = 'article_text'
          AND runs.reconciliation_complete
          AND action.expected_generation_run_id IS NOT NULL
        """,
        [configuration_id, configuration_id, configuration_id],
    ).fetchall()
    canonical_cache: dict[str, tuple[str, bytes] | None] = {}
    for (
        article_id,
        article_input_sha256,
        aggregate_vector,
        aggregate_vector_sha256,
        generation_run_id,
        active,
    ) in aggregates:
        has_part_generations = any(
            key[:2] == (str(article_id), str(article_input_sha256))
            for key in generations
        )
        provenance = connection.execute(
            """
            SELECT part_index, article_input_sha256, part_count,
                   part_generation_run_id, part_input_sha256, token_count,
                   part_stored_vector_sha256, aggregation_version
            FROM article_text_aggregation_provenance
            WHERE article_id = ? AND configuration_id = ?
              AND generation_run_id = ?
            ORDER BY part_index
            """,
            [article_id, configuration_id, generation_run_id],
        ).fetchall()
        if not provenance:
            if has_part_generations:
                _append(
                    issues,
                    "missing_long_text_provenance",
                    "long-text aggregates lack complete part provenance",
                )
            continue
        expected_count = provenance[0][2]
        if (
            not isinstance(expected_count, int)
            or len(provenance) != expected_count
            or [row[0] for row in provenance] != list(range(expected_count))
        ):
            _append(
                issues,
                "missing_long_text_provenance",
                "long-text aggregates lack complete part provenance",
            )
            continue
        weighted_parts: list[tuple[int, tuple[float, ...]]] = []
        linked_parts: list[tuple[object, ...]] = []
        linkage_valid = True
        for row in provenance:
            (
                part_index,
                provenance_article_sha256,
                part_count,
                part_generation_run_id,
                part_input_sha256,
                part_token_count,
                part_vector_sha256,
                aggregation_version,
            ) = row
            part = generations.get(
                (
                    str(article_id),
                    str(article_input_sha256),
                    int(part_index),
                    str(part_generation_run_id),
                )
            )
            if part is None:
                linkage_valid = False
                continue
            (
                current_count,
                char_start,
                char_end,
                byte_start,
                byte_end,
                current_input_sha256,
                current_token_count,
                current_vector_sha256,
                current_vector,
            ) = part
            current_link_valid = (
                provenance_article_sha256 == article_input_sha256
                and part_count == expected_count == current_count
                and part_input_sha256 == current_input_sha256
                and part_token_count == current_token_count
                and part_vector_sha256 == current_vector_sha256
                and aggregation_version
                == configured_aggregation
                == "l2-token-count-weighted-mean-float32-v1"
                and current_vector is not None
            )
            if not current_link_valid:
                linkage_valid = False
                continue
            weighted_parts.append(
                (
                    int(part_token_count),
                    tuple(float(value) for value in current_vector),
                )
            )
            linked_parts.append(
                (
                    char_start,
                    char_end,
                    byte_start,
                    byte_end,
                    current_input_sha256,
                )
            )
        if not linkage_valid or len(weighted_parts) != expected_count:
            _append(
                issues,
                "invalid_long_text_provenance",
                "long-text provenance does not identify its part generations",
            )
            continue
        try:
            expected_vector = _weighted_long_text_vector(weighted_parts)
        except (OverflowError, TypeError, ValueError, struct.error):
            _append(
                issues,
                "invalid_long_text_provenance",
                "long-text provenance does not identify its part generations",
            )
            continue
        if (
            aggregate_vector_sha256 != _vector_hash(expected_vector)
            or (active and tuple(aggregate_vector) != expected_vector)
        ):
            _append(
                issues,
                "long_text_vector_mismatch",
                "long-text vector differs from its token-weighted parts",
            )
        article = articles.get(str(article_id))
        if article is None or article.cleaned_markdown_sha256 != article_input_sha256:
            if not _spans_are_ordered(linked_parts):
                _append(
                    issues,
                    "invalid_long_text_source_coverage",
                    "long-text part spans are not ordered and gapless",
                )
            continue
        canonical = canonical_cache.get(str(article_id))
        if str(article_id) not in canonical_cache:
            try:
                markdown_bytes = read_canonical_markdown(
                    config.preprocessing_output_root,
                    article.cleaned_markdown_path,
                )
                markdown = markdown_bytes.decode("utf-8", errors="strict")
            except (CanonicalMarkdownError, UnicodeDecodeError):
                canonical = None
            else:
                canonical = (markdown, markdown_bytes)
            canonical_cache[str(article_id)] = canonical
        if canonical is not None and not _spans_cover_canonical(
            linked_parts,
            canonical[0],
            canonical[1],
        ):
            _append(
                issues,
                "invalid_long_text_source_coverage",
                "long-text part spans do not cover canonical Markdown exactly",
            )


def _valid_part_span(
    char_start: object,
    char_end: object,
    byte_start: object,
    byte_end: object,
) -> bool:
    return (
        isinstance(char_start, int)
        and isinstance(char_end, int)
        and isinstance(byte_start, int)
        and isinstance(byte_end, int)
        and 0 <= char_start < char_end
        and 0 <= byte_start < byte_end
    )


def _spans_are_ordered(parts: list[tuple[object, ...]]) -> bool:
    char_position = 0
    byte_position = 0
    for char_start, char_end, byte_start, byte_end, _part_sha256 in parts:
        if char_start != char_position or byte_start != byte_position:
            return False
        char_position = int(char_end)
        byte_position = int(byte_end)
    return bool(parts)


def _spans_cover_canonical(
    parts: list[tuple[object, ...]],
    markdown: str,
    markdown_bytes: bytes,
) -> bool:
    if not _spans_are_ordered(parts):
        return False
    for char_start, char_end, byte_start, byte_end, part_sha256 in parts:
        char_slice = markdown[int(char_start) : int(char_end)].encode("utf-8")
        byte_slice = markdown_bytes[int(byte_start) : int(byte_end)]
        if (
            char_slice != byte_slice
            or hashlib.sha256(byte_slice).hexdigest() != part_sha256
        ):
            return False
    return parts[-1][1] == len(markdown) and parts[-1][3] == len(markdown_bytes)


def _weighted_long_text_vector(
    parts: list[tuple[int, tuple[float, ...]]],
) -> tuple[float, ...]:
    total_tokens = sum(part_token_count for part_token_count, _vector in parts)
    values = tuple(
        sum(part_token_count * vector[index] for part_token_count, vector in parts)
        / total_tokens
        for index in range(_VECTOR_DIMENSIONS)
    )
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("invalid weighted long-text vector")
    return tuple(
        struct.unpack("<f", struct.pack("<f", value / norm))[0]
        for value in values
    )


def _generation_reference_exists(
    connection: duckdb.DuckDBPyConnection,
    *,
    article_id: str,
    modality: str,
    configuration_id: str,
    generation_run_id: str,
    stored_vector_sha256: str | None,
) -> bool:
    """Resolve one generation against active or superseded content-free state."""

    active = connection.execute(
        """
        SELECT e.stored_vector_sha256
        FROM embedding_work_items AS w
        JOIN embeddings AS e USING (article_id, modality, configuration_id)
        WHERE w.article_id = ? AND w.modality = ? AND w.configuration_id = ?
          AND w.generation_run_id = ?
        """,
        [article_id, modality, configuration_id, generation_run_id],
    ).fetchone()
    history = connection.execute(
        """
        SELECT stored_vector_sha256
        FROM embedding_generation_history
        WHERE article_id = ? AND modality = ? AND configuration_id = ?
          AND generation_run_id = ?
        """,
        [article_id, modality, configuration_id, generation_run_id],
    ).fetchone()
    pending = connection.execute(
        """
        SELECT action.expected_stored_vector_sha256
        FROM reconciliation_actions AS action
        JOIN runs ON runs.run_id = action.run_id
        WHERE action.article_id = ? AND action.modality = ?
          AND action.configuration_id = ?
          AND action.expected_generation_run_id = ?
          AND runs.reconciliation_complete
        """,
        [article_id, modality, configuration_id, generation_run_id],
    ).fetchone()
    references = tuple(row for row in (active, history, pending) if row is not None)
    if stored_vector_sha256 is None:
        return bool(references)
    return any(row[0] == stored_vector_sha256 for row in references)


def _validate_generation_history(
    connection: duckdb.DuckDBPyConnection,
    configuration_id: str,
    issues: list[EmbeddingValidationIssue],
) -> None:
    generation_run_ids = {
        row[0]
        for row in connection.execute(
            "SELECT run_id FROM runs WHERE configuration_id = ?",
            [configuration_id],
        ).fetchall()
    }
    all_run_ids = {
        row[0] for row in connection.execute("SELECT run_id FROM runs").fetchall()
    }
    reasons = {reason.value for reason in SupersessionReason}
    rows = connection.execute(
        """
        SELECT modality, source_relative_path, generation_run_id, input_sha256,
               stored_vector_sha256, superseded_run_id, superseded_reason
        FROM embedding_generation_history
        WHERE configuration_id = ?
        """,
        [configuration_id],
    ).fetchall()
    for (
        modality,
        source_relative_path,
        generation_run_id,
        input_sha256,
        vector_sha256,
        superseded_run_id,
        reason,
    ) in rows:
        if modality not in {item.value for item in EmbeddingModality}:
            _append(
                issues,
                "invalid_generation_modality",
                "embedding generation history uses an unsupported modality",
            )
        if modality == EmbeddingModality.HEADER_IMAGE.value:
            if not is_safe_source_relative_path(source_relative_path):
                _append(
                    issues,
                    "invalid_generation_source_path",
                    "header-image generation history uses an unsafe source path",
                )
        elif source_relative_path is not None:
            _append(
                issues,
                "invalid_generation_source_path",
                "non-image generation history retains a source path",
            )
        if (
            generation_run_id not in generation_run_ids
            or superseded_run_id not in all_run_ids
        ):
            _append(
                issues,
                "invalid_generation_run_reference",
                "embedding generation history lacks a matching run reference",
            )
        if reason not in reasons:
            _append(
                issues,
                "invalid_supersession_reason",
                "embedding generation history uses an unsupported reason",
            )
        if not _is_sha256(input_sha256) or not _is_sha256(vector_sha256):
            _append(
                issues,
                "invalid_generation_hash",
                "embedding generation history contains an invalid hash",
            )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_markdown_hash(
    config: EmbeddingPipelineConfig,
    article: CanonicalArticle,
    input_sha256: str,
    issues: list[EmbeddingValidationIssue],
) -> None:
    try:
        markdown_bytes = read_canonical_markdown(
            config.preprocessing_output_root,
            article.cleaned_markdown_path,
        )
    except CanonicalMarkdownError as error:
        if error.status == "missing":
            _append(
                issues,
                "missing_canonical_markdown",
                "canonical Markdown is unavailable",
            )
        else:
            _append(
                issues,
                "unsafe_markdown_path",
                "canonical Markdown path is unsafe",
            )
        return
    if hashlib.sha256(markdown_bytes).hexdigest() != input_sha256:
        _append(
            issues,
            "input_hash_mismatch",
            "embedding input hash differs from canonical Markdown",
        )


def _validate_header_image_hash(
    config: EmbeddingPipelineConfig,
    article: CanonicalArticle,
    source_relative_path: str | None,
    input_sha256: str,
    issues: list[EmbeddingValidationIssue],
) -> None:
    if (
        source_relative_path is None
        or source_relative_path != article.header_image_path
    ):
        _append(
            issues,
            "header_image_path_mismatch",
            "embedding header image path differs from the canonical article",
        )
        return
    try:
        image_bytes = read_source_image(config.source_root, source_relative_path)
    except SourceImageError as error:
        code = (
            "missing_header_image"
            if error.status == "missing"
            else "unsafe_header_image_path"
        )
        message = (
            "canonical header image is unavailable"
            if error.status == "missing"
            else "canonical header image path is unsafe"
        )
        _append(
            issues,
            code,
            message,
        )
        return
    if hashlib.sha256(image_bytes).hexdigest() != input_sha256:
        _append(
            issues,
            "input_hash_mismatch",
            "embedding input hash differs from the canonical header image",
        )


def _validate_vector(
    vector: tuple[float, ...],
    stored_vector_sha256: str,
    issues: list[EmbeddingValidationIssue],
) -> None:
    try:
        values = tuple(float(value) for value in vector)
    except (TypeError, ValueError):
        _append(issues, "invalid_vector", "embedding rows contain an invalid vector")
        return
    if len(values) != _VECTOR_DIMENSIONS:
        _append(
            issues,
            "invalid_vector_dimensions",
            "embedding rows contain an invalid vector dimension",
        )
        return
    if not all(math.isfinite(value) for value in values):
        _append(issues, "nonfinite_vector", "embedding rows contain non-finite vectors")
        return
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        _append(issues, "zero_vector", "embedding rows contain zero vectors")
    elif not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-5):
        _append(issues, "nonunit_vector", "embedding rows contain non-unit vectors")
    packed = b"".join(struct.pack("<f", value) for value in values)
    if hashlib.sha256(packed).hexdigest() != stored_vector_sha256:
        _append(
            issues,
            "vector_hash_mismatch",
            "embedding vector hashes do not match float32 values",
        )


def _append(
    issues: list[EmbeddingValidationIssue],
    code: str,
    message: str,
) -> None:
    issues.append(EmbeddingValidationIssue(code, message))
