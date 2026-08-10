"""Credential-free integrity validation for published article embeddings."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import duckdb

from wsj_embeddings.catalog import (
    EmbeddingCatalog,
    EmbeddingCatalogError,
    configuration_id,
)
from wsj_embeddings.config import EmbeddingPipelineConfig
from wsj_embeddings.models import EmbeddingProfile
from wsj_embeddings.pipeline import _ARTICLE_TEXT_MODALITY, _VECTOR_DIMENSIONS


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
class _CanonicalArticle:
    article_id: str
    cleaned_markdown_path: str
    published_at_utc: datetime
    publication_date_new_york: date


def validate_embedding_outputs(
    config: EmbeddingPipelineConfig,
) -> EmbeddingValidationResult:
    """Check embedding output against canonical generated preprocessing state."""

    config.validate()
    issues: list[EmbeddingValidationIssue] = []
    try:
        with EmbeddingCatalog.read_only(config.embedding_catalog) as catalog:
            articles = _canonical_articles(config, issues)
            if articles is not None:
                _validate_configurations(catalog.connection, issues)
                _validate_embeddings(catalog.connection, config, articles, issues)
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
) -> dict[str, _CanonicalArticle] | None:
    try:
        with duckdb.connect(str(config.preprocessing_catalog), read_only=True) as db:
            rows = db.execute(
                """
                SELECT article_id, cleaned_markdown_path, published_at_utc,
                       publication_date_new_york
                FROM articles
                """
            ).fetchall()
    except duckdb.Error:
        issues.append(
            EmbeddingValidationIssue(
                "invalid_preprocessing_catalog",
                "canonical preprocessing catalog could not be queried",
            )
        )
        return None
    return {row[0]: _CanonicalArticle(*row) for row in rows}


def _validate_configurations(
    connection: duckdb.DuckDBPyConnection,
    issues: list[EmbeddingValidationIssue],
) -> None:
    invalid = False
    rows = connection.execute(
        """
        SELECT configuration_id, model, task, dimensions, output_type, normalization
        FROM embedding_configurations
        """
    ).fetchall()
    configuration_ids = {row[0] for row in rows}
    for row in rows:
        profile = EmbeddingProfile(*row[1:])
        if row[0] != configuration_id(profile):
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


def _validate_embeddings(
    connection: duckdb.DuckDBPyConnection,
    config: EmbeddingPipelineConfig,
    articles: dict[str, _CanonicalArticle],
    issues: list[EmbeddingValidationIssue],
) -> None:
    rows = connection.execute(
        """
        SELECT article_id, modality, configuration_id, published_at_utc,
               publication_date_new_york, dimensions, input_sha256,
               stored_vector_sha256, vector
        FROM embeddings
        ORDER BY article_id, modality, configuration_id
        """
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
            profile_id,
            published_at_utc,
            publication_date,
            dimensions,
            input_sha256,
            stored_vector_sha256,
            vector,
        ) = row
        article = articles.get(article_id)
        if article is None:
            _append(
                issues,
                "missing_canonical_article",
                "embedding rows lack a canonical article",
            )
            continue
        if profile_id not in configuration_ids:
            _append(
                issues,
                "missing_configuration_reference",
                "embedding rows lack a configuration reference",
            )
        if modality != _ARTICLE_TEXT_MODALITY:
            _append(
                issues,
                "invalid_modality",
                "embedding rows use an unsupported modality",
            )
        if dimensions != _VECTOR_DIMENSIONS:
            _append(
                issues,
                "invalid_dimensions",
                "embedding rows use an unsupported dimension",
            )
        if (
            published_at_utc != article.published_at_utc
            or publication_date != article.publication_date_new_york
        ):
            _append(
                issues,
                "publication_metadata_mismatch",
                "embedding publication metadata differs from the canonical article",
            )
        _validate_markdown_hash(config, article, input_sha256, issues)
        _validate_vector(vector, stored_vector_sha256, issues)


def _validate_markdown_hash(
    config: EmbeddingPipelineConfig,
    article: _CanonicalArticle,
    input_sha256: str,
    issues: list[EmbeddingValidationIssue],
) -> None:
    relative_path = Path(article.cleaned_markdown_path)
    if relative_path.is_absolute():
        _append(issues, "unsafe_markdown_path", "canonical Markdown path is unsafe")
        return
    path = (config.preprocessing_output_root / relative_path).resolve()
    if not path.is_relative_to(config.preprocessing_output_root):
        _append(issues, "unsafe_markdown_path", "canonical Markdown path is unsafe")
        return
    try:
        markdown_bytes = path.read_bytes()
    except OSError:
        _append(
            issues,
            "missing_canonical_markdown",
            "canonical Markdown is unavailable",
        )
        return
    if hashlib.sha256(markdown_bytes).hexdigest() != input_sha256:
        _append(
            issues,
            "input_hash_mismatch",
            "embedding input hash differs from canonical Markdown",
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
