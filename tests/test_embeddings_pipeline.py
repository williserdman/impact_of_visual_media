from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb

from wsj_embeddings import (
    EmbeddingPipelineConfig,
    EmbeddingRunResult,
    FakeEmbeddingAdapter,
    run_embedding_pipeline,
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
    markdown = "# Synthetic embedding article\n\nSynthetic editorial content.\n"
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
    assert row[:5] == (
        "wsj:SYNTHETIC-EMBEDDING",
        "article_text",
        datetime(2024, 1, 2, 15, 30, tzinfo=UTC),
        date(2024, 1, 2),
        2048,
    )
    assert row[7] == [1.0] + [0.0] * 2047
    assert snapshot_fixture_inputs(config) == before
