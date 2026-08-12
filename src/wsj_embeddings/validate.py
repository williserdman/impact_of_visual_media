"""Credential-free integrity validation for published article embeddings."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import stat
import struct
from collections.abc import Callable
from dataclasses import dataclass
from tempfile import TemporaryDirectory
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
    VectorDerivationKind,
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
from wsj_embeddings.validation_stream import (
    OrderIndependentDigest,
    RowKind,
    stream_rows,
)
from wsj_pipeline.catalog import Catalog, CatalogError

_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)


class _Digest(Protocol):
    def update(self, value: bytes) -> None: ...


class _CanonicalArticles:
    """Bounded read interface over the pinned preprocessing snapshot."""

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self._connection = connection

    def __len__(self) -> int:
        row = self._connection.execute("SELECT count(*) FROM articles").fetchone()
        return int(row[0])

    def __iter__(self):
        for (article_id,) in stream_rows(
            self._connection,
            "SELECT article_id FROM articles ORDER BY article_id",
        ):
            yield str(article_id)

    def items(self):
        for row in stream_rows(
            self._connection,
            """
            SELECT article_id, cleaned_markdown_path, published_at_utc,
                   publication_date_new_york, cleaned_markdown_sha256,
                   header_image_path
            FROM articles ORDER BY article_id
            """,
        ):
            yield str(row[0]), CanonicalArticle(*row)

    def get(self, article_id: object) -> CanonicalArticle | None:
        row = self._connection.execute(
            """
            SELECT article_id, cleaned_markdown_path, published_at_utc,
                   publication_date_new_york, cleaned_markdown_sha256,
                   header_image_path
            FROM articles WHERE article_id = ?
            """,
            [article_id],
        ).fetchone()
        return None if row is None else CanonicalArticle(*row)

    def __contains__(self, article_id: object) -> bool:
        return self.get(article_id) is not None


@dataclass(frozen=True, slots=True)
class EmbeddingValidationIssue:
    """One stable validation failure without text, vectors, or credentials."""

    code: str
    message: str


class _IssueCollector(list[EmbeddingValidationIssue]):
    """Deduplicate the finite public issue vocabulary as validation proceeds."""

    def __init__(self) -> None:
        super().__init__()
        self._seen: set[EmbeddingValidationIssue] = set()

    def append(self, issue: EmbeddingValidationIssue) -> None:
        if issue not in self._seen:
            self._seen.add(issue)
            super().append(issue)


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
    issues: list[EmbeddingValidationIssue] = _IssueCollector()
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
    articles: _CanonicalArticles,
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
) -> _CanonicalArticles:
    return _CanonicalArticles(connection)


def _artifact_state_token(
    config: EmbeddingPipelineConfig,
    embedding: duckdb.DuckDBPyConnection,
    articles: _CanonicalArticles,
) -> str:
    """Fingerprint referenced inputs and the complete no-follow rendition tree."""

    digest = hashlib.sha256()
    for article_id, article in articles.items():
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
    for (raw_relative_path,) in stream_rows(
        embedding,
        """
        SELECT rendition_relative_path FROM image_input_provenance
        WHERE rendition_relative_path IS NOT NULL
        ORDER BY rendition_relative_path
        """,
    ):
        relative_path = str(raw_relative_path)
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

    tree_digest = OrderIndependentDigest()
    try:
        _accumulate_rendition_tree_token(tree_digest, output_root)
    finally:
        _update_artifact_token(
            digest,
            "rendition-tree-token",
            tree_digest.hexdigest(),
        )


def _accumulate_rendition_tree_token(
    digest: _Digest,
    output_root: os.PathLike[str] | str,
) -> None:
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
            for namespace in os.scandir(rendition_descriptor):
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
                    for entry in os.scandir(namespace_descriptor):
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
    rows = stream_rows(
        connection,
        """
        SELECT configuration_id, model, client_api_contract_version, task,
               dimensions, output_type, normalization,
               tokenizer_revision, tokenizer_engine, context_token_limit,
               context_rules, long_text_aggregation,
               long_text_part_attempt_limit, batch_max_items,
               batch_max_estimated_tokens, batch_max_encoded_bytes,
               batch_max_response_bytes,
               batch_max_concurrency, batch_max_attempts,
               batch_initial_backoff_seconds, batch_max_backoff_seconds,
               image_input_rules,
               image_transform, multimodal_formula,
               client_configuration_version
        FROM embedding_configurations
        """,
    )
    for row in rows:
        profile = EmbeddingProfile(
            model=row[1],
            client_api_contract_version=row[2],
            task=row[3],
            dimensions=row[4],
            output_type=row[5],
            normalization=row[6],
            tokenizer_revision=row[7],
            tokenizer_engine=row[8],
            context_token_limit=row[9],
            context_rules=row[10],
            long_text_aggregation=row[11],
            long_text_part_attempt_limit=row[12],
            batch_max_items=row[13],
            batch_max_estimated_tokens=row[14],
            batch_max_encoded_bytes=row[15],
            batch_max_response_bytes=row[16],
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
    if connection.execute(
        """
        SELECT count(*) FROM runs
        LEFT JOIN embedding_configurations USING (configuration_id)
        WHERE embedding_configurations.configuration_id IS NULL
        """
    ).fetchone()[0]:
        issues.append(
            EmbeddingValidationIssue(
                "missing_run_configuration_reference",
                "embedding runs lack a configuration reference",
            )
        )
    if connection.execute(
        """
        SELECT count(*) FROM embedding_storage AS vector
        LEFT JOIN embedding_configurations USING (configuration_id)
        WHERE embedding_configurations.configuration_id IS NULL
        """
    ).fetchone()[0]:
        _append(
            issues,
            "missing_configuration_reference",
            "embedding rows lack a configuration reference",
        )


def _validate_run_metrics(
    connection: duckdb.DuckDBPyConnection,
    issues: list[EmbeddingValidationIssue],
) -> None:
    invalid = False
    rows = stream_rows(
        connection,
        """
        SELECT hosted_requests, hosted_retries, usage_json, throttles,
               elapsed_seconds
        FROM runs
        """,
    )
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

    invalid = bool(
        connection.execute(
            """
            SELECT count(*) FROM full_run_articles AS inventory
            LEFT JOIN runs USING (run_id)
            WHERE runs.run_id IS NULL OR runs.scope <> 'full'
            """
        ).fetchone()[0]
    )
    rows = stream_rows(
        connection,
        """
        SELECT run_id, scope, status, discovery_complete,
               reconciliation_complete, articles, finished_at,
               (SELECT count(*) FROM full_run_articles AS inventory
                WHERE inventory.run_id = runs.run_id) AS inventory_count
        FROM runs ORDER BY run_id
        """,
    )
    for (
        _run_id,
        scope,
        status,
        discovery_complete,
        reconciliation_complete,
        articles,
        finished_at,
        inventory_count,
    ) in rows:
        inventory_count = int(inventory_count)
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
           OR vector.derivation_kind IS DISTINCT FROM
                action.expected_derivation_kind
           OR vector.raw_response_sha256 IS DISTINCT FROM
                action.expected_raw_response_sha256
           OR vector.response_model IS DISTINCT FROM
                action.expected_response_model
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
    hidden_vectors = stream_rows(
        connection,
        """
        SELECT vector.modality, vector.configuration_id, vector.dimensions,
               vector.derivation_kind, vector.raw_response_sha256,
               vector.response_model, vector.stored_vector_sha256, vector.vector,
               configuration.client_api_contract_version,
               configuration.tokenizer_revision, configuration.tokenizer_engine,
               EXISTS (
                   SELECT 1 FROM article_text_aggregation_provenance AS p
                   WHERE p.article_id = vector.article_id
                     AND p.configuration_id = vector.configuration_id
                     AND p.generation_run_id = action.expected_generation_run_id
               ) AS has_aggregation_provenance
        FROM reconciliation_actions AS action
        JOIN runs ON runs.run_id = action.run_id
        JOIN embedding_storage AS vector
          ON vector.article_id = action.article_id
         AND vector.modality = action.modality
         AND vector.configuration_id = action.configuration_id
        JOIN embedding_configurations AS configuration
          ON configuration.configuration_id = vector.configuration_id
        WHERE runs.reconciliation_complete
        ORDER BY vector.article_id, vector.modality, vector.configuration_id
        """,
        kind=RowKind.SINGLE_VECTOR,
    )
    for (
        modality,
        _configuration_id,
        dimensions,
        derivation_kind,
        raw_response_sha256,
        response_model,
        stored_vector_sha256,
        vector,
        client_api_contract_version,
        tokenizer_revision,
        tokenizer_engine,
        has_aggregation_provenance,
    ) in hidden_vectors:
        synthetic = _is_synthetic_fixture_profile(
            client_api_contract_version,
            tokenizer_revision,
            tokenizer_engine,
        )
        vector_valid = _validate_vector(vector, stored_vector_sha256, issues)
        if (
            dimensions != _VECTOR_DIMENSIONS
            or not _valid_vector_provenance(
                modality,
                derivation_kind,
                raw_response_sha256,
                response_model,
                synthetic_configuration=synthetic,
                has_aggregation_provenance=bool(has_aggregation_provenance),
            )
            or not vector_valid
        ):
            _append(
                issues,
                "invalid_reconciliation_vector",
                "hidden reconciliation vectors have invalid shape or provenance",
            )


def _select_configuration_id(
    connection: duckdb.DuckDBPyConnection,
    requested: str | None,
    issues: list[EmbeddingValidationIssue],
) -> str | None:
    if requested is not None:
        if connection.execute(
            "SELECT 1 FROM embedding_configurations WHERE configuration_id = ?",
            [requested],
        ).fetchone() is None:
            _append(
                issues,
                "unknown_configuration_id",
                "validation configuration identity is not present",
            )
            return None
        return requested
    count, sole = connection.execute(
        "SELECT count(*), min(configuration_id) FROM embedding_configurations"
    ).fetchone()
    if count > 1:
        _append(
            issues,
            "configuration_id_required",
            "validation requires a configuration identity when generations coexist",
        )
        return None
    return str(sole) if count else None


def _coverage_for_configuration(
    connection: duckdb.DuckDBPyConnection,
    articles: _CanonicalArticles,
    configuration_id: str,
    *,
    orphan_artifacts: int,
) -> EmbeddingCoverage:
    """Account for canonical eligibility and current durable work states."""

    canonical_articles = len(articles)
    header_present = sum(
        article.header_image_path is not None
        for _article_id, article in articles.items()
    )
    text_success = header_success = multimodal_success = 0
    stale_work = unresolved_retryable_work = 0
    for article_id, modality, state in stream_rows(
        connection,
        """
        SELECT article_id, modality, state FROM embedding_work_items
        WHERE configuration_id = ? ORDER BY article_id, modality
        """,
        [configuration_id],
    ):
        state = str(state)
        stale_work += state == WorkState.STALE_INPUT.value
        unresolved_retryable_work += state == WorkState.RETRYABLE.value
        article = articles.get(article_id)
        if article is None or state != WorkState.SUCCEEDED.value:
            continue
        text_success += modality == EmbeddingModality.ARTICLE_TEXT.value
        if article.header_image_path is not None:
            header_success += modality == EmbeddingModality.HEADER_IMAGE.value
            multimodal_success += (
                modality == EmbeddingModality.MULTIMODAL_ARTICLE.value
            )
    return EmbeddingCoverage(
        canonical_articles=canonical_articles,
        text_success=text_success,
        text_failure=canonical_articles - text_success,
        header_present=header_present,
        header_absent=canonical_articles - header_present,
        header_success=header_success,
        header_failure=header_present - header_success,
        multimodal_success=multimodal_success,
        multimodal_unavailable=canonical_articles - header_present,
        multimodal_failure=header_present - multimodal_success,
        stale_work=stale_work,
        unresolved_retryable_work=unresolved_retryable_work,
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
    articles: _CanonicalArticles,
    configuration_id: str,
    issues: list[EmbeddingValidationIssue],
) -> None:
    rows = stream_rows(
        connection,
        """
        SELECT work.article_id, work.modality, work.configuration_id,
               work.source_relative_path, work.input_sha256, work.state,
               work.attempt_count, work.error_code, work.status_code,
               work.retry_after_seconds, work.generation_run_id,
               matching.article_id IS NOT NULL AS has_matching_embedding,
               keyed.article_id IS NOT NULL AS has_embedding_key,
               generation.run_id IS NOT NULL AS has_generation_run
        FROM embedding_work_items AS work
        LEFT JOIN embeddings AS matching
          ON matching.article_id = work.article_id
         AND matching.modality = work.modality
         AND matching.configuration_id = work.configuration_id
         AND matching.source_relative_path IS NOT DISTINCT FROM
             work.source_relative_path
         AND matching.input_sha256 IS NOT DISTINCT FROM work.input_sha256
        LEFT JOIN embeddings AS keyed
          ON keyed.article_id = work.article_id
         AND keyed.modality = work.modality
         AND keyed.configuration_id = work.configuration_id
        LEFT JOIN runs AS generation
          ON generation.run_id = work.generation_run_id
         AND generation.configuration_id = work.configuration_id
        WHERE work.configuration_id = ?
        ORDER BY work.article_id, work.modality
        """,
        [configuration_id],
    )
    for row in rows:
        (
            article_id,
            modality,
            _configuration_identifier,
            source_relative_path,
            input_sha256,
            state,
            attempt_count,
            error_code,
            status_code,
            retry_after_seconds,
            generation_run_id,
            has_matching_embedding,
            has_embedding_key,
            has_generation_run,
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
            if has_embedding_key:
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
                or has_embedding_key
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
                if not has_generation_run:
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
    articles: _CanonicalArticles,
    configuration_id: str,
    issues: list[EmbeddingValidationIssue],
) -> None:
    """Require one header disposition per selected article and honest run claims."""

    missing_header_disposition = False
    for (article_id,) in stream_rows(
        connection,
        """
        SELECT text.article_id FROM embedding_work_items AS text
        LEFT JOIN embedding_work_items AS header
          ON header.article_id = text.article_id
         AND header.configuration_id = text.configuration_id
         AND header.modality = 'header_image'
        WHERE text.configuration_id = ? AND text.modality = 'article_text'
          AND header.article_id IS NULL
        ORDER BY text.article_id
        """,
        [configuration_id],
    ):
        if articles.get(article_id) is not None:
            missing_header_disposition = True
            break
    if missing_header_disposition:
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
    raw_states = stream_rows(
        connection,
        """
        SELECT state
        FROM embedding_work_items
        WHERE configuration_id = ? AND modality = 'header_image'
          AND last_run_id = ?
        """,
        [configuration_id, run_id],
    )
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
    articles: _CanonicalArticles,
    selected_configuration_id: str,
    issues: list[EmbeddingValidationIssue],
) -> None:
    synthetic_configuration = _is_synthetic_fixture_configuration(
        connection, selected_configuration_id
    )
    rows = stream_rows(
        connection,
        """
        SELECT e.article_id, e.modality, e.configuration_id, e.published_at_utc,
               e.source_relative_path, e.publication_date_new_york, e.dimensions,
               e.input_sha256, e.derivation_kind, e.raw_response_sha256,
               e.response_model, e.stored_vector_sha256, e.vector,
               EXISTS (
                   SELECT 1 FROM article_text_aggregation_provenance AS p
                   WHERE p.article_id = e.article_id
                     AND p.configuration_id = e.configuration_id
                     AND p.generation_run_id = w.generation_run_id
               ) AS has_aggregation_provenance
        FROM embeddings AS e
        JOIN embedding_work_items AS w
          USING (article_id, modality, configuration_id)
        WHERE e.configuration_id = ?
        ORDER BY e.article_id, e.modality, e.configuration_id
        """,
        [selected_configuration_id],
        kind=RowKind.SINGLE_VECTOR,
    )
    for row in rows:
        (
            article_id,
            modality,
            _configuration_identifier,
            published_at_utc,
            source_relative_path,
            publication_date,
            dimensions,
            input_sha256,
            derivation_kind,
            raw_response_sha256,
            response_model,
            stored_vector_sha256,
            vector,
            has_aggregation_provenance,
        ) = row
        article = articles.get(article_id)
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
        if not _valid_vector_provenance(
            modality,
            derivation_kind,
            raw_response_sha256,
            response_model,
            synthetic_configuration=synthetic_configuration,
            has_aggregation_provenance=bool(has_aggregation_provenance),
        ):
            _append(
                issues,
                "invalid_vector_provenance",
                "embedding rows have inconsistent hosted or derived provenance",
            )
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
    articles: _CanonicalArticles,
    configuration_id: str,
    issues: list[EmbeddingValidationIssue],
    image_codec: ImageCodec | None,
) -> None:
    """Resolve every image generation to the exact bytes supplied to Jina."""

    generation_relation = """
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
    """
    missing = connection.execute(
        f"""
        SELECT count(*) FROM ({generation_relation}) AS expected
        LEFT JOIN image_input_provenance AS provenance
          ON provenance.article_id = expected.article_id
         AND provenance.configuration_id = ?
         AND provenance.generation_run_id = expected.generation_run_id
         AND provenance.source_sha256 = expected.input_sha256
        WHERE provenance.article_id IS NULL
        """,
        [configuration_id, configuration_id, configuration_id, configuration_id],
    ).fetchone()[0]
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
    rows = stream_rows(
        connection,
        """
        SELECT provenance.article_id, provenance.generation_run_id,
               provenance.source_sha256, provenance.source_format,
               provenance.source_bytes, provenance.source_width,
               provenance.source_height, provenance.source_frames,
               provenance.embedded_input_sha256, provenance.embedded_format,
               provenance.embedded_bytes, provenance.embedded_width,
               provenance.embedded_height, provenance.embedded_frames,
               provenance.transform_id, provenance.rendition_relative_path,
               active.article_id IS NOT NULL AS is_active,
               (SELECT count(*) FROM (
                   SELECT w.article_id, w.generation_run_id, e.input_sha256
                   FROM embedding_work_items AS w JOIN embeddings AS e
                     USING (article_id, modality, configuration_id)
                   WHERE w.configuration_id = ? AND w.modality = 'header_image'
                     AND w.state = 'succeeded'
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
               ) AS expected
                WHERE expected.article_id = provenance.article_id
                  AND expected.generation_run_id = provenance.generation_run_id
                  AND expected.input_sha256 = provenance.source_sha256) > 0
                  AS is_expected
        FROM image_input_provenance AS provenance
        LEFT JOIN embedding_work_items AS active
          ON active.article_id = provenance.article_id
         AND active.configuration_id = provenance.configuration_id
         AND active.modality = 'header_image' AND active.state = 'succeeded'
         AND active.generation_run_id = provenance.generation_run_id
        WHERE provenance.configuration_id = ?
        ORDER BY provenance.article_id, provenance.generation_run_id
        """,
        [configuration_id, configuration_id, configuration_id, configuration_id],
    )
    for row in rows:
        (
            article_id,
            _generation_run_id,
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
            is_active,
            is_expected,
        ) = row
        if not is_expected:
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
            and is_active
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
    if missing:
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
    def unsafe_entry() -> None:
        _append(
            issues,
            "unsafe_image_rendition_entry",
            "rendition namespace contains an unsafe entry",
        )

    with TemporaryDirectory(prefix="wsj-embedding-validation-") as scratch:
        reference_index = sqlite3.connect(os.path.join(scratch, "references.sqlite3"))
        try:
            reference_index.execute(
                "CREATE TABLE referenced_paths (path TEXT PRIMARY KEY)"
            )
            for (relative_path,) in stream_rows(
                connection,
                """
                SELECT DISTINCT rendition_relative_path
                FROM image_input_provenance
                WHERE rendition_relative_path IS NOT NULL
                ORDER BY rendition_relative_path
                """,
            ):
                reference_index.execute(
                    "INSERT INTO referenced_paths VALUES (?)",
                    [str(relative_path)],
                )
            reference_index.commit()
            return _scan_rendition_artifacts_with_index(
                reference_index,
                config,
                issues,
                unsafe_entry,
            )
        finally:
            reference_index.close()


def _scan_rendition_artifacts_with_index(
    reference_index: sqlite3.Connection,
    config: EmbeddingPipelineConfig,
    issues: list[EmbeddingValidationIssue],
    unsafe_entry: Callable[[], None],
) -> int:
    orphan_count = 0
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
                                if reference_index.execute(
                                    "SELECT 1 FROM referenced_paths WHERE path = ?",
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
    composite_relation = """
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
    """
    missing = connection.execute(
        f"""
        SELECT count(*) FROM ({composite_relation}) AS generation
        LEFT JOIN multimodal_embedding_provenance AS provenance
          ON provenance.article_id = generation.article_id
         AND provenance.configuration_id = ?
         AND provenance.generation_run_id = generation.generation_run_id
        WHERE provenance.article_id IS NULL
        """,
        [configuration_id, configuration_id, configuration_id, configuration_id],
    ).fetchone()[0]
    if missing:
        _append(
            issues,
            "missing_multimodal_provenance",
            "multimodal generations lack immutable source provenance",
        )
    provenance_rows = stream_rows(
        connection,
        f"""
        SELECT provenance.article_id, provenance.generation_run_id,
               provenance.text_generation_run_id,
               provenance.text_stored_vector_sha256,
               provenance.header_image_generation_run_id,
               provenance.header_image_stored_vector_sha256,
               provenance.formula_version, generation.input_sha256,
               EXISTS (
                   SELECT 1 FROM embedding_generation_history AS history
                   WHERE history.article_id = provenance.article_id
                     AND history.configuration_id = provenance.configuration_id
                     AND history.modality = 'article_text'
                     AND history.generation_run_id =
                         provenance.text_generation_run_id
                     AND history.stored_vector_sha256 =
                         provenance.text_stored_vector_sha256
                   UNION ALL
                   SELECT 1 FROM embedding_work_items AS work
                   JOIN embeddings AS vector
                     USING (article_id, modality, configuration_id)
                   WHERE work.article_id = provenance.article_id
                     AND work.configuration_id = provenance.configuration_id
                     AND work.modality = 'article_text'
                     AND work.generation_run_id =
                         provenance.text_generation_run_id
                     AND vector.stored_vector_sha256 =
                         provenance.text_stored_vector_sha256
                   UNION ALL
                   SELECT 1 FROM reconciliation_actions AS action
                   JOIN runs ON runs.run_id = action.run_id
                   WHERE action.article_id = provenance.article_id
                     AND action.configuration_id = provenance.configuration_id
                     AND action.modality = 'article_text'
                     AND action.expected_generation_run_id =
                         provenance.text_generation_run_id
                     AND action.expected_stored_vector_sha256 =
                         provenance.text_stored_vector_sha256
                     AND runs.reconciliation_complete
               ) AS text_exists,
               EXISTS (
                   SELECT 1 FROM embedding_generation_history AS history
                   WHERE history.article_id = provenance.article_id
                     AND history.configuration_id = provenance.configuration_id
                     AND history.modality = 'header_image'
                     AND history.generation_run_id =
                         provenance.header_image_generation_run_id
                     AND history.stored_vector_sha256 =
                         provenance.header_image_stored_vector_sha256
                   UNION ALL
                   SELECT 1 FROM embedding_work_items AS work
                   JOIN embeddings AS vector
                     USING (article_id, modality, configuration_id)
                   WHERE work.article_id = provenance.article_id
                     AND work.configuration_id = provenance.configuration_id
                     AND work.modality = 'header_image'
                     AND work.generation_run_id =
                         provenance.header_image_generation_run_id
                     AND vector.stored_vector_sha256 =
                         provenance.header_image_stored_vector_sha256
                   UNION ALL
                   SELECT 1 FROM reconciliation_actions AS action
                   JOIN runs ON runs.run_id = action.run_id
                   WHERE action.article_id = provenance.article_id
                     AND action.configuration_id = provenance.configuration_id
                     AND action.modality = 'header_image'
                     AND action.expected_generation_run_id =
                         provenance.header_image_generation_run_id
                     AND action.expected_stored_vector_sha256 =
                         provenance.header_image_stored_vector_sha256
                     AND runs.reconciliation_complete
               ) AS image_exists
        FROM multimodal_embedding_provenance AS provenance
        LEFT JOIN ({composite_relation}) AS generation
          ON generation.article_id = provenance.article_id
         AND generation.generation_run_id = provenance.generation_run_id
        WHERE provenance.configuration_id = ?
        ORDER BY provenance.article_id, provenance.generation_run_id
        """,
        [configuration_id, configuration_id, configuration_id, configuration_id],
    )
    for row in provenance_rows:
        (
            article_id,
            _composite_generation_run_id,
            text_generation_run_id,
            text_vector_sha256,
            image_generation_run_id,
            image_vector_sha256,
            formula_version,
            input_sha256,
            text_exists,
            image_exists,
        ) = row
        provenance_valid = (
            formula_version == configuration_formula == _MULTIMODAL_FORMULA
            and _is_sha256(text_vector_sha256)
            and _is_sha256(image_vector_sha256)
            and input_sha256 is not None
        )
        if not provenance_valid:
            _append(
                issues,
                "invalid_multimodal_provenance",
                "multimodal provenance is incomplete or inconsistent",
            )
        if input_sha256 is not None and input_sha256 != _multimodal_input_sha256(
            formula_version=str(formula_version),
            text_generation_run_id=str(text_generation_run_id),
            text_stored_vector_sha256=str(text_vector_sha256),
            header_image_generation_run_id=str(image_generation_run_id),
            header_image_stored_vector_sha256=str(image_vector_sha256),
        ):
            _append(
                issues,
                "multimodal_input_identity_mismatch",
                "multimodal input identity differs from immutable provenance",
            )
        if not text_exists or not image_exists:
            _append(
                issues,
                "invalid_multimodal_source_linkage",
                "multimodal provenance does not identify its source generations",
            )

    active_rows = stream_rows(
        connection,
        """
        SELECT composite.article_id, composite.vector,
               composite.stored_vector_sha256, composite_work.generation_run_id,
               text_work.generation_run_id, text.stored_vector_sha256, text.vector,
               image_work.generation_run_id, image.stored_vector_sha256, image.vector,
               provenance.text_generation_run_id,
               provenance.text_stored_vector_sha256,
               provenance.header_image_generation_run_id,
               provenance.header_image_stored_vector_sha256,
               provenance.formula_version
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
        LEFT JOIN multimodal_embedding_provenance AS provenance
          ON provenance.article_id = composite.article_id
         AND provenance.configuration_id = composite.configuration_id
         AND provenance.generation_run_id = composite_work.generation_run_id
        WHERE composite.configuration_id = ?
          AND composite.modality = 'multimodal_article'
        """,
        [configuration_id],
        kind=RowKind.TRIPLE_VECTOR,
    )
    for row in active_rows:
        (
            article_id,
            composite_values,
            composite_vector_sha256,
            _composite_generation_run_id,
            text_generation_run_id,
            text_vector_sha256,
            text_values,
            image_generation_run_id,
            image_vector_sha256,
            image_values,
            provenance_text_run,
            provenance_text_hash,
            provenance_image_run,
            provenance_image_hash,
            provenance_formula,
        ) = row
        provenance = (
            provenance_text_run,
            provenance_text_hash,
            provenance_image_run,
            provenance_image_hash,
            provenance_formula,
        )
        expected_linkage = (
            str(text_generation_run_id),
            str(text_vector_sha256),
            str(image_generation_run_id),
            str(image_vector_sha256),
            configuration_formula,
        )
        if (
            None
            in (
                *provenance,
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
    articles: _CanonicalArticles,
    configuration_id: str,
    issues: list[EmbeddingValidationIssue],
) -> None:
    """Verify current checkpoints and every active or archived aggregate."""

    configuration = connection.execute(
        """
        SELECT context_token_limit, long_text_aggregation,
               long_text_part_attempt_limit, client_api_contract_version,
               tokenizer_revision, tokenizer_engine
        FROM embedding_configurations
        WHERE configuration_id = ?
        """,
        [configuration_id],
    ).fetchone()
    if configuration is None:
        return
    (
        token_limit,
        configured_aggregation,
        attempt_limit,
        client_api_contract_version,
        tokenizer_revision,
        tokenizer_engine,
    ) = configuration
    synthetic_configuration = _is_synthetic_fixture_profile(
        client_api_contract_version, tokenizer_revision, tokenizer_engine
    )
    part_rows = stream_rows(
        connection,
        """
        SELECT part.article_id, part.article_input_sha256, part.part_index,
               part.part_count, part.char_start, part.char_end, part.byte_start,
               part.byte_end, part.part_input_sha256, part.token_count,
               part.state, part.attempt_count, part.error_code, part.status_code,
               part.retry_after_seconds, part.last_run_id,
               part.generation_run_id, part.derivation_kind,
               part.raw_response_sha256, part.response_model,
               part.stored_vector_sha256, part.vector,
               last_run.run_id IS NOT NULL, generation_run.run_id IS NOT NULL
        FROM long_text_parts AS part
        LEFT JOIN runs AS last_run ON last_run.run_id = part.last_run_id
          AND last_run.configuration_id = part.configuration_id
        LEFT JOIN runs AS generation_run
          ON generation_run.run_id = part.generation_run_id
         AND generation_run.configuration_id = part.configuration_id
        WHERE part.configuration_id = ?
        ORDER BY part.article_id, part.article_input_sha256, part.part_index
        """,
        [configuration_id],
        kind=RowKind.SINGLE_VECTOR,
    )
    for row in part_rows:
        (
            _article_id,
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
            _last_run_id,
            generation_run_id,
            derivation_kind,
            raw_response_sha256,
            response_model,
            vector_sha256,
            vector,
            last_run_valid,
            generation_run_valid,
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
            and last_run_valid
            and state is not None
            and state is not WorkState.NOT_APPLICABLE
        )
        if state is WorkState.SUCCEEDED:
            state_valid = (
                attempt_count >= 1
                and not any(value is not None for value in failure_metadata)
                and generation_run_valid
                and _valid_part_provenance(
                    derivation_kind,
                    raw_response_sha256,
                    response_model,
                    synthetic_configuration=synthetic_configuration,
                )
                and _is_sha256(vector_sha256)
                and vector is not None
            )
            if state_valid:
                state_valid = _validate_vector(vector, vector_sha256, issues)
        else:
            state_valid = (
                generation_run_id is None
                and derivation_kind is None
                and raw_response_sha256 is None
                and response_model is None
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
    generation_rows = stream_rows(
        connection,
        """
        SELECT part.article_id, part.article_input_sha256, part.part_index,
               part.generation_run_id, part.part_count, part.char_start,
               part.char_end, part.byte_start, part.byte_end,
               part.part_input_sha256, part.token_count, part.derivation_kind,
               part.raw_response_sha256, part.response_model,
               part.stored_vector_sha256, part.vector, run.run_id IS NOT NULL
        FROM long_text_part_generations AS part
        LEFT JOIN runs AS run ON run.run_id = part.generation_run_id
          AND run.configuration_id = part.configuration_id
        WHERE part.configuration_id = ?
        ORDER BY part.article_id, part.article_input_sha256, part.part_index,
                 part.generation_run_id
        """,
        [configuration_id],
        kind=RowKind.SINGLE_VECTOR,
    )
    for row in generation_rows:
        (
            _article_id,
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
            derivation_kind,
            raw_response_sha256,
            response_model,
            vector_sha256,
            vector,
            generation_run_valid,
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
            and generation_run_valid
            and _valid_part_provenance(
                derivation_kind,
                raw_response_sha256,
                response_model,
                synthetic_configuration=synthetic_configuration,
            )
            and _is_sha256(vector_sha256)
            and vector is not None
        )
        if valid:
            valid = _validate_vector(vector, vector_sha256, issues)
        if not valid:
            _append(
                issues,
                "invalid_long_text_part_generation",
                "long-text part generation history is incomplete or inconsistent",
            )
    if connection.execute(
        """
        SELECT count(*) FROM long_text_parts AS current
        LEFT JOIN long_text_part_generations AS generation
          ON generation.article_id = current.article_id
         AND generation.configuration_id = current.configuration_id
         AND generation.article_input_sha256 = current.article_input_sha256
         AND generation.part_index = current.part_index
         AND generation.generation_run_id = current.generation_run_id
        WHERE current.configuration_id = ? AND current.state = 'succeeded'
          AND (generation.article_id IS NULL
            OR generation.part_count IS DISTINCT FROM current.part_count
            OR generation.char_start IS DISTINCT FROM current.char_start
            OR generation.char_end IS DISTINCT FROM current.char_end
            OR generation.byte_start IS DISTINCT FROM current.byte_start
            OR generation.byte_end IS DISTINCT FROM current.byte_end
            OR generation.part_input_sha256 IS DISTINCT FROM current.part_input_sha256
            OR generation.token_count IS DISTINCT FROM current.token_count
            OR generation.derivation_kind IS DISTINCT FROM current.derivation_kind
            OR generation.raw_response_sha256 IS DISTINCT FROM
               current.raw_response_sha256
            OR generation.response_model IS DISTINCT FROM current.response_model
            OR generation.stored_vector_sha256 IS DISTINCT FROM
               current.stored_vector_sha256
            OR generation.vector IS DISTINCT FROM current.vector)
        """,
        [configuration_id],
    ).fetchone()[0]:
        _append(
            issues,
            "invalid_long_text_part",
            "long-text part checkpoints are incomplete or inconsistent",
        )

    aggregate_key: tuple[str, str, str, int] | None = None
    while aggregate := _next_long_text_aggregate(
        connection,
        configuration_id,
        aggregate_key,
    ):
        aggregate_key = (
            str(aggregate[0]),
            str(aggregate[1]),
            str(aggregate[4]),
            int(aggregate[5]),
        )
        _validate_long_text_aggregate(
            connection,
            config,
            articles,
            configuration_id,
            configured_aggregation,
            aggregate,
            issues,
        )


def _next_long_text_aggregate(
    connection: duckdb.DuckDBPyConnection,
    configuration_id: str,
    key: tuple[str, str, str, int] | None,
) -> tuple[object, ...] | None:
    predicate = ""
    parameters: list[object] = [configuration_id] * 3
    if key is not None:
        predicate = (
            "WHERE (article_id, input_sha256, generation_run_id, active) "
            "> (?, ?, ?, ?)"
        )
        parameters.extend(key)
    return connection.execute(
        f"""
        WITH aggregates AS (
            SELECT e.article_id, e.input_sha256, e.vector,
                   e.stored_vector_sha256, w.generation_run_id, 1 AS active
            FROM embeddings AS e JOIN embedding_work_items AS w
              USING (article_id, modality, configuration_id)
            WHERE e.configuration_id = ? AND e.modality = 'article_text'
              AND w.state = 'succeeded'
            UNION ALL
            SELECT article_id, input_sha256, NULL, stored_vector_sha256,
                   generation_run_id, 0
            FROM embedding_generation_history
            WHERE configuration_id = ? AND modality = 'article_text'
            UNION ALL
            SELECT action.article_id, action.expected_vector_input_sha256,
                   NULL, action.expected_stored_vector_sha256,
                   action.expected_generation_run_id, 0
            FROM reconciliation_actions AS action
            JOIN runs ON runs.run_id = action.run_id
            WHERE action.configuration_id = ?
              AND action.modality = 'article_text'
              AND runs.reconciliation_complete
              AND action.expected_generation_run_id IS NOT NULL
        )
        SELECT * FROM aggregates {predicate}
        ORDER BY article_id, input_sha256, generation_run_id, active LIMIT 1
        """,
        parameters,
    ).fetchone()


def _validate_long_text_aggregate(
    connection: duckdb.DuckDBPyConnection,
    config: EmbeddingPipelineConfig,
    articles: _CanonicalArticles,
    configuration_id: str,
    configured_aggregation: str,
    aggregate: tuple[object, ...],
    issues: list[EmbeddingValidationIssue],
) -> None:
    article_id, article_hash, active_vector, vector_hash, run_id, active = aggregate
    article = articles.get(article_id)
    canonical: tuple[str, bytes] | None = None
    if article is not None and article.cleaned_markdown_sha256 == article_hash:
        try:
            markdown_bytes = read_canonical_markdown(
                config.preprocessing_output_root,
                article.cleaned_markdown_path,
            )
            canonical = (
                markdown_bytes.decode("utf-8", errors="strict"),
                markdown_bytes,
            )
        except (CanonicalMarkdownError, UnicodeDecodeError):
            pass
    rows = stream_rows(
        connection,
        """
        SELECT provenance.part_index, provenance.article_input_sha256,
               provenance.part_count, provenance.part_input_sha256,
               provenance.token_count, provenance.part_stored_vector_sha256,
               provenance.aggregation_version, part.part_count,
               part.char_start, part.char_end, part.byte_start, part.byte_end,
               part.part_input_sha256, part.token_count,
               part.stored_vector_sha256, part.vector
        FROM article_text_aggregation_provenance AS provenance
        LEFT JOIN long_text_part_generations AS part
          ON part.article_id = provenance.article_id
         AND part.configuration_id = provenance.configuration_id
         AND part.article_input_sha256 = provenance.article_input_sha256
         AND part.part_index = provenance.part_index
         AND part.generation_run_id = provenance.part_generation_run_id
        WHERE provenance.article_id = ? AND provenance.configuration_id = ?
          AND provenance.generation_run_id = ?
        ORDER BY provenance.part_index
        """,
        [article_id, configuration_id, run_id],
        kind=RowKind.SINGLE_VECTOR,
    )
    accumulator = [0.0] * _VECTOR_DIMENSIONS
    total_tokens = count = 0
    expected_count: int | None = None
    char_position = byte_position = 0
    linkage_valid = coverage_valid = True
    for row in rows:
        (
            part_index,
            provenance_article_hash,
            provenance_count,
            provenance_part_hash,
            provenance_tokens,
            provenance_vector_hash,
            aggregation_version,
            stored_count,
            char_start,
            char_end,
            byte_start,
            byte_end,
            stored_part_hash,
            stored_tokens,
            stored_vector_hash,
            vector,
        ) = row
        expected_count = provenance_count if expected_count is None else expected_count
        valid = (
            part_index == count
            and provenance_article_hash == article_hash
            and provenance_count == expected_count == stored_count
            and provenance_part_hash == stored_part_hash
            and provenance_tokens == stored_tokens
            and provenance_vector_hash == stored_vector_hash
            and aggregation_version
            == configured_aggregation
            == "l2-token-count-weighted-mean-float32-v1"
            and vector is not None
        )
        linkage_valid &= valid
        if not valid:
            count += 1
            continue
        token_count = int(provenance_tokens)
        total_tokens += token_count
        for index, value in enumerate(vector):
            accumulator[index] += token_count * float(value)
        coverage_valid &= char_start == char_position and byte_start == byte_position
        char_position, byte_position = int(char_end), int(byte_end)
        if canonical is not None:
            char_slice = canonical[0][int(char_start) : int(char_end)].encode("utf-8")
            byte_slice = canonical[1][int(byte_start) : int(byte_end)]
            coverage_valid &= (
                char_slice == byte_slice
                and hashlib.sha256(byte_slice).hexdigest() == stored_part_hash
            )
        count += 1
    if count == 0:
        has_parts = connection.execute(
            """
            SELECT 1 FROM long_text_part_generations
            WHERE article_id = ? AND configuration_id = ?
              AND article_input_sha256 = ? LIMIT 1
            """,
            [article_id, configuration_id, article_hash],
        ).fetchone()
        if has_parts is not None:
            _append(
                issues,
                "missing_long_text_provenance",
                "long-text aggregates lack complete part provenance",
            )
        return
    if expected_count != count:
        _append(
            issues,
            "missing_long_text_provenance",
            "long-text aggregates lack complete part provenance",
        )
        return
    if not linkage_valid or total_tokens < 1:
        _append(
            issues,
            "invalid_long_text_provenance",
            "long-text provenance does not identify its part generations",
        )
        return
    if canonical is not None:
        coverage_valid &= (
            char_position == len(canonical[0]) and byte_position == len(canonical[1])
        )
    if not coverage_valid:
        _append(
            issues,
            "invalid_long_text_source_coverage",
            "long-text part spans are not ordered and gapless",
        )
    try:
        expected_vector = _weighted_long_text_accumulator(accumulator, total_tokens)
    except (OverflowError, TypeError, ValueError, struct.error):
        _append(
            issues,
            "invalid_long_text_provenance",
            "long-text provenance does not identify its part generations",
        )
        return
    if vector_hash != _vector_hash(expected_vector) or (
        active and tuple(active_vector) != expected_vector
    ):
        _append(
            issues,
            "long_text_vector_mismatch",
            "long-text vector differs from its token-weighted parts",
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


def _weighted_long_text_accumulator(
    accumulator: list[float],
    total_tokens: int,
) -> tuple[float, ...]:
    if total_tokens < 1:
        raise ValueError("invalid weighted long-text vector")
    values = tuple(value / total_tokens for value in accumulator)
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("invalid weighted long-text vector")
    return tuple(
        struct.unpack("<f", struct.pack("<f", value / norm))[0]
        for value in values
    )


def _validate_generation_history(
    connection: duckdb.DuckDBPyConnection,
    configuration_id: str,
    issues: list[EmbeddingValidationIssue],
) -> None:
    synthetic_configuration = _is_synthetic_fixture_configuration(
        connection, configuration_id
    )
    reasons = {reason.value for reason in SupersessionReason}
    rows = stream_rows(
        connection,
        """
        SELECT history.article_id, history.modality,
               history.source_relative_path, history.generation_run_id,
               history.input_sha256, history.derivation_kind,
               history.raw_response_sha256, history.response_model,
               history.stored_vector_sha256, history.superseded_run_id,
               history.superseded_reason, generation.run_id IS NOT NULL,
               superseding.run_id IS NOT NULL,
               EXISTS (
                   SELECT 1 FROM article_text_aggregation_provenance AS provenance
                   WHERE provenance.article_id = history.article_id
                     AND provenance.configuration_id = history.configuration_id
                     AND provenance.generation_run_id = history.generation_run_id
               )
        FROM embedding_generation_history AS history
        LEFT JOIN runs AS generation ON generation.run_id = history.generation_run_id
          AND generation.configuration_id = history.configuration_id
        LEFT JOIN runs AS superseding
          ON superseding.run_id = history.superseded_run_id
        WHERE history.configuration_id = ?
        ORDER BY history.article_id, history.modality, history.generation_run_id
        """,
        [configuration_id],
    )
    for (
        _article_id,
        modality,
        source_relative_path,
        _generation_run_id,
        input_sha256,
        derivation_kind,
        raw_response_sha256,
        response_model,
        vector_sha256,
        _superseded_run_id,
        reason,
        generation_run_valid,
        superseded_run_valid,
        has_aggregation_provenance,
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
        if not generation_run_valid or not superseded_run_valid:
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
        if not _valid_vector_provenance(
            modality,
            derivation_kind,
            raw_response_sha256,
            response_model,
            synthetic_configuration=synthetic_configuration,
            has_aggregation_provenance=bool(
                modality == EmbeddingModality.ARTICLE_TEXT.value
                and has_aggregation_provenance
            ),
        ):
            _append(
                issues,
                "invalid_generation_provenance",
                "embedding generation history has inconsistent provenance",
            )


def _valid_part_provenance(
    derivation_kind: object,
    raw_response_sha256: object,
    response_model: object,
    *,
    synthetic_configuration: bool,
) -> bool:
    return _valid_vector_provenance(
        EmbeddingModality.ARTICLE_TEXT.value,
        derivation_kind,
        raw_response_sha256,
        response_model,
        allow_aggregate=False,
        synthetic_configuration=synthetic_configuration,
    )


def _valid_vector_provenance(
    modality: object,
    derivation_kind: object,
    raw_response_sha256: object,
    response_model: object,
    *,
    allow_aggregate: bool = True,
    synthetic_configuration: bool = False,
    has_aggregation_provenance: bool = False,
) -> bool:
    if modality == EmbeddingModality.MULTIMODAL_ARTICLE.value:
        return (
            allow_aggregate
            and derivation_kind
            == VectorDerivationKind.MULTIMODAL_MIDPOINT.value
            and raw_response_sha256 is None
            and response_model is None
        )
    if derivation_kind == VectorDerivationKind.HOSTED_RESPONSE.value:
        return (
            modality
            in {
                EmbeddingModality.ARTICLE_TEXT.value,
                EmbeddingModality.HEADER_IMAGE.value,
            }
            and _is_sha256(raw_response_sha256)
            and _is_safe_response_model(response_model)
        )
    if derivation_kind == VectorDerivationKind.SYNTHETIC_FIXTURE.value:
        return (
            synthetic_configuration
            and modality
            in {
                EmbeddingModality.ARTICLE_TEXT.value,
                EmbeddingModality.HEADER_IMAGE.value,
            }
            and raw_response_sha256 is None
            and response_model is None
        )
    if (
        allow_aggregate
        and modality == EmbeddingModality.ARTICLE_TEXT.value
        and derivation_kind == VectorDerivationKind.LONG_TEXT_AGGREGATE.value
        and has_aggregation_provenance
    ):
        return raw_response_sha256 is None and response_model is None
    return False


def _is_synthetic_fixture_configuration(
    connection: duckdb.DuckDBPyConnection, configuration_id: str
) -> bool:
    row = connection.execute(
        """
        SELECT client_api_contract_version, tokenizer_revision, tokenizer_engine
        FROM embedding_configurations WHERE configuration_id = ?
        """,
        [configuration_id],
    ).fetchone()
    return bool(row and _is_synthetic_fixture_profile(*row))


def _is_synthetic_fixture_profile(
    client_api_contract_version: object,
    tokenizer_revision: object,
    tokenizer_engine: object,
) -> bool:
    return (
        client_api_contract_version == "synthetic-v1"
        and isinstance(tokenizer_revision, str)
        and tokenizer_revision.startswith("synthetic-")
        and isinstance(tokenizer_engine, str)
        and tokenizer_engine.startswith("synthetic-")
    )


def _is_safe_response_model(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and len(value.encode("utf-8")) <= 256
        and all(
            ord(character) >= 0x20 and ord(character) != 0x7F
            for character in value
        )
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
) -> bool:
    try:
        values = tuple(float(value) for value in vector)
    except (TypeError, ValueError):
        _append(issues, "invalid_vector", "embedding rows contain an invalid vector")
        return False
    if len(values) != _VECTOR_DIMENSIONS:
        _append(
            issues,
            "invalid_vector_dimensions",
            "embedding rows contain an invalid vector dimension",
        )
        return False
    if not all(math.isfinite(value) for value in values):
        _append(issues, "nonfinite_vector", "embedding rows contain non-finite vectors")
        return False
    valid = True
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        _append(issues, "zero_vector", "embedding rows contain zero vectors")
        valid = False
    elif not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-5):
        _append(issues, "nonunit_vector", "embedding rows contain non-unit vectors")
        valid = False
    packed = b"".join(struct.pack("<f", value) for value in values)
    if hashlib.sha256(packed).hexdigest() != stored_vector_sha256:
        _append(
            issues,
            "vector_hash_mismatch",
            "embedding vector hashes do not match float32 values",
        )
        valid = False
    return valid


def _append(
    issues: list[EmbeddingValidationIssue],
    code: str,
    message: str,
) -> None:
    issues.append(EmbeddingValidationIssue(code, message))
