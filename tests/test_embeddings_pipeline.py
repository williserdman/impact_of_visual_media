from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import pytest

from wsj_embeddings import (
    EmbeddingCatalogError,
    EmbeddingPipelineConfig,
    EmbeddingRunResult,
    EmbeddingValidationIssue,
    EmbeddingValidationResult,
    FakeEmbeddingAdapter,
    run_embedding_pipeline,
    validate_embedding_outputs,
)


def write_generated_preprocessing_fixture(tmp_path: Path) -> EmbeddingPipelineConfig:
    source_root = tmp_path / "source"
    preprocessing_output_root = tmp_path / "preprocessed"
    embedding_output_root = tmp_path / "embeddings"
    source_root.mkdir()
    markdown_path = (
        preprocessing_output_root / "text" / "2024" / "01" / "02" / "article.md"
    )
    markdown_path.parent.mkdir(parents=True)
    markdown = "#\n"
    markdown_path.write_text(markdown, encoding="utf-8")
    with duckdb.connect(str(preprocessing_output_root / "catalog.duckdb")) as db:
        db.execute(
            """
            CREATE TABLE articles (
                article_id VARCHAR PRIMARY KEY,
                source_html_path VARCHAR NOT NULL,
                cleaned_markdown_path VARCHAR NOT NULL,
                header_image_path VARCHAR,
                inline_image_urls VARCHAR[] NOT NULL,
                published_at_utc TIMESTAMP WITH TIME ZONE NOT NULL,
                publication_date_new_york DATE NOT NULL,
                cleaned_markdown_sha256 VARCHAR NOT NULL
            )
            """
        )
        db.execute(
            """
            INSERT INTO articles VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "wsj:SYNTHETIC-EMBEDDING",
                "synthetic/article.html",
                markdown_path.relative_to(preprocessing_output_root).as_posix(),
                None,
                [],
                datetime(2024, 1, 2, 15, 30, tzinfo=UTC),
                date(2024, 1, 2),
                hashlib.sha256(markdown.encode()).hexdigest(),
            ],
        )
    return EmbeddingPipelineConfig(
        source_root=source_root,
        preprocessing_output_root=preprocessing_output_root,
        embedding_output_root=embedding_output_root,
    )


def snapshot_fixture_inputs(config: EmbeddingPipelineConfig) -> dict[str, str]:
    roots = (config.source_root, config.preprocessing_output_root)
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for root in roots
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_pipeline_publishes_one_normalized_article_text_embedding(tmp_path):
    config = write_generated_preprocessing_fixture(tmp_path)
    before = snapshot_fixture_inputs(config)

    result = run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)

    assert result == EmbeddingRunResult(articles=1, embeddings=1)
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        row = db.execute(
            """
            SELECT article_id, modality, published_at_utc,
                   publication_date_new_york, dimensions,
                   input_sha256, stored_vector_sha256, vector
            FROM embeddings
            """
        ).fetchone()
        vector_type = db.execute("PRAGMA table_info('embeddings')").fetchall()[8][2]
    assert row[:5] == (
        "wsj:SYNTHETIC-EMBEDDING",
        "article_text",
        datetime(2024, 1, 2, 15, 30, tzinfo=UTC),
        date(2024, 1, 2),
        2048,
    )
    assert vector_type == "FLOAT[2048]"
    assert row[7] == (1.0,) + (0.0,) * 2047
    assert snapshot_fixture_inputs(config) == before


def test_validator_accepts_fresh_synthetic_embedding_output(tmp_path):
    config = write_generated_preprocessing_fixture(tmp_path)

    run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)

    assert validate_embedding_outputs(config) == EmbeddingValidationResult(issues=())


def test_pipeline_refuses_unknown_existing_embedding_schema(tmp_path):
    config = write_generated_preprocessing_fixture(tmp_path)
    config.embedding_output_root.mkdir()
    with duckdb.connect(str(config.embedding_catalog)) as db:
        db.execute("CREATE TABLE legacy_vectors(value INTEGER)")

    with pytest.raises(EmbeddingCatalogError, match="unsupported embedding catalog"):
        run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)


def test_validator_reports_changed_canonical_markdown_without_text(tmp_path):
    config = write_generated_preprocessing_fixture(tmp_path)
    run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)
    markdown_path = (
        config.preprocessing_output_root / "text" / "2024" / "01" / "02" / "article.md"
    )
    markdown_path.write_text("# changed\n", encoding="utf-8")

    result = validate_embedding_outputs(config)

    assert result == EmbeddingValidationResult(
        issues=(
            EmbeddingValidationIssue(
                "input_hash_mismatch",
                "embedding input hash differs from canonical Markdown",
            ),
        )
    )


def test_validator_reports_vector_and_publication_metadata_corruption(tmp_path):
    config = write_generated_preprocessing_fixture(tmp_path)
    run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)
    with duckdb.connect(str(config.embedding_catalog)) as db:
        db.execute(
            """
            UPDATE embeddings
            SET published_at_utc = ?, vector = ?
            """,
            [datetime(2024, 1, 3, 15, 30, tzinfo=UTC), [0.0] * 2048],
        )

    result = validate_embedding_outputs(config)

    assert result == EmbeddingValidationResult(
        issues=(
            EmbeddingValidationIssue(
                "publication_metadata_mismatch",
                "embedding publication metadata differs from the canonical article",
            ),
            EmbeddingValidationIssue(
                "vector_hash_mismatch",
                "embedding vector hashes do not match float32 values",
            ),
            EmbeddingValidationIssue(
                "zero_vector",
                "embedding rows contain zero vectors",
            ),
        )
    )


def test_validator_reports_run_without_configuration_reference(tmp_path):
    config = write_generated_preprocessing_fixture(tmp_path)
    run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)
    with duckdb.connect(str(config.embedding_catalog)) as db:
        db.execute("UPDATE runs SET configuration_id = 'missing'")

    result = validate_embedding_outputs(config)

    assert result == EmbeddingValidationResult(
        issues=(
            EmbeddingValidationIssue(
                "missing_run_configuration_reference",
                "embedding runs lack a configuration reference",
            ),
        )
    )


def test_validator_checks_intrinsic_properties_for_orphaned_embedding_rows(tmp_path):
    config = write_generated_preprocessing_fixture(tmp_path)
    run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)
    with duckdb.connect(str(config.embedding_catalog)) as db:
        db.execute(
            "UPDATE embeddings SET article_id = 'orphan', modality = 'unsupported'"
        )

    result = validate_embedding_outputs(config)

    assert result == EmbeddingValidationResult(
        issues=(
            EmbeddingValidationIssue(
                "invalid_modality",
                "embedding rows use an unsupported modality",
            ),
            EmbeddingValidationIssue(
                "missing_canonical_article",
                "embedding rows lack a canonical article",
            ),
        )
    )
