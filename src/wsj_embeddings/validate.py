"""Credential-free integrity validation for published article embeddings."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass

import duckdb

from wsj_embeddings.canonical_markdown import (
    CanonicalMarkdownError,
    read_canonical_markdown,
)
from wsj_embeddings.catalog import EmbeddingCatalog, EmbeddingCatalogError
from wsj_embeddings.catalog import configuration_id as profile_configuration_id
from wsj_embeddings.config import EmbeddingPipelineConfig
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
from wsj_embeddings.source_image import (
    SourceImageError,
    is_safe_source_relative_path,
    read_source_image,
)
from wsj_pipeline.catalog import Catalog, CatalogError


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


def validate_embedding_outputs(
    config: EmbeddingPipelineConfig,
    *,
    configuration_id: str | None = None,
) -> EmbeddingValidationResult:
    """Check embedding output against canonical generated preprocessing state."""

    config.validate()
    issues: list[EmbeddingValidationIssue] = []
    try:
        with EmbeddingCatalog.read_only(config.embedding_catalog) as catalog:
            articles = _canonical_articles(config, issues)
            if articles is not None:
                _validate_configurations(catalog.connection, issues)
                selected_configuration_id = _select_configuration_id(
                    catalog.connection,
                    configuration_id,
                    issues,
                )
                if selected_configuration_id is not None:
                    _validate_run_coverage(
                        catalog.connection, selected_configuration_id, issues
                    )
                    _validate_work_items(
                        catalog.connection,
                        articles,
                        selected_configuration_id,
                        issues,
                    )
                    _validate_header_disposition_coverage(
                        catalog.connection,
                        articles,
                        selected_configuration_id,
                        issues,
                    )
                    _validate_embeddings(
                        catalog.connection,
                        config,
                        articles,
                        selected_configuration_id,
                        issues,
                    )
                    _validate_multimodal_provenance(
                        catalog.connection,
                        selected_configuration_id,
                        issues,
                    )
                    _validate_generation_history(
                        catalog.connection, selected_configuration_id, issues
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
    return EmbeddingValidationResult(
        tuple(sorted(set(issues), key=lambda issue: (issue.code, issue.message)))
    )


def _canonical_articles(
    config: EmbeddingPipelineConfig,
    issues: list[EmbeddingValidationIssue],
) -> dict[str, CanonicalArticle] | None:
    try:
        with Catalog.read_only(config.preprocessing_catalog) as db:
            Catalog.validate_schema(db)
            rows = db.execute(
                """
                SELECT article_id, cleaned_markdown_path, published_at_utc,
                       publication_date_new_york, cleaned_markdown_sha256,
                       header_image_path
                FROM articles
                """
            ).fetchall()
    except (CatalogError, duckdb.Error, OSError):
        issues.append(
            EmbeddingValidationIssue(
                "invalid_preprocessing_catalog",
                "canonical preprocessing catalog could not be queried",
            )
        )
        return None
    return {row[0]: CanonicalArticle(*row) for row in rows}


def _validate_configurations(
    connection: duckdb.DuckDBPyConnection,
    issues: list[EmbeddingValidationIssue],
) -> None:
    invalid = False
    rows = connection.execute(
        """
        SELECT configuration_id, model, observed_model, observed_api_version,
               task, dimensions, output_type, normalization,
               tokenizer_revision, context_token_limit, context_rules,
               long_text_aggregation, image_input_rules, image_transform,
               multimodal_formula, client_configuration_version
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
            context_token_limit=row[9],
            context_rules=row[10],
            long_text_aggregation=row[11],
            image_input_rules=row[12],
            image_transform=row[13],
            multimodal_formula=row[14],
            client_configuration_version=row[15],
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
        [configuration_id, configuration_id, configuration_id],
    ).fetchone()[0]
    if missing:
        _append(
            issues,
            "missing_published_embedding",
            "successful embedding runs lack their published vectors",
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
            if not is_missing_header and (error_code is None or attempt_count < 1):
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
        """,
        [configuration_id, configuration_id],
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
    references = tuple(row for row in (active, history) if row is not None)
    if stored_vector_sha256 is None:
        return bool(references)
    return any(row[0] == stored_vector_sha256 for row in references)


def _validate_generation_history(
    connection: duckdb.DuckDBPyConnection,
    configuration_id: str,
    issues: list[EmbeddingValidationIssue],
) -> None:
    run_ids = {
        row[0]
        for row in connection.execute(
            "SELECT run_id FROM runs WHERE configuration_id = ?",
            [configuration_id],
        ).fetchall()
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
        if generation_run_id not in run_ids or superseded_run_id not in run_ids:
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
