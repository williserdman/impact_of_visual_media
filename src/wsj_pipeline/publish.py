"""Atomic publication of canonical and audit Parquet datasets."""

from __future__ import annotations

import os
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from wsj_pipeline.canonical import (
    PartitionKey,
    deserialize_article,
)
from wsj_pipeline.config import PipelineConfig
from wsj_pipeline.models import ExtractedArticle
from wsj_pipeline.state import PipelineState

ARTICLE_SCHEMA_VERSION = "1"
UTC_TIMESTAMP = pa.timestamp("us", tz="UTC")
NEW_YORK_TIMESTAMP = pa.timestamp("us", tz="America/New_York")

ARTICLE_SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("article_key", pa.string(), nullable=False),
        pa.field("wsj_article_id", pa.string()),
        pa.field("canonical_url", pa.string()),
        pa.field("source_path", pa.string(), nullable=False),
        pa.field("source_html_sha256", pa.string(), nullable=False),
        pa.field("extractor_version", pa.string(), nullable=False),
        pa.field("headline", pa.string()),
        pa.field("original_headline", pa.string()),
        pa.field("summary", pa.string()),
        pa.field("authors", pa.list_(pa.string()), nullable=False),
        pa.field("keywords", pa.list_(pa.string()), nullable=False),
        pa.field("section", pa.string()),
        pa.field("page", pa.string()),
        pa.field("article_type", pa.string()),
        pa.field("access", pa.string()),
        pa.field("published_at_utc", UTC_TIMESTAMP, nullable=False),
        pa.field("publication_date_utc", pa.date32(), nullable=False),
        pa.field("published_at_new_york", NEW_YORK_TIMESTAMP, nullable=False),
        pa.field("publication_date_new_york", pa.date32(), nullable=False),
        pa.field("publication_year_ny", pa.int32(), nullable=False),
        pa.field("publication_month_ny", pa.int8(), nullable=False),
        pa.field("updated_at_utc", UTC_TIMESTAMP),
        pa.field("created_at_utc", UTC_TIMESTAMP),
        pa.field("last_published_at_utc", UTC_TIMESTAMP),
        pa.field("body_paragraphs", pa.list_(pa.string()), nullable=False),
        pa.field("body_text", pa.string(), nullable=False),
        pa.field("body_word_count", pa.int32(), nullable=False),
        pa.field("image_path", pa.string()),
        pa.field("image_present", pa.bool_(), nullable=False),
        pa.field("image_media_type", pa.string()),
        pa.field("image_size_bytes", pa.int64()),
        pa.field("image_caption", pa.string()),
        pa.field("image_credit", pa.string()),
        pa.field("image_creator", pa.string()),
        pa.field("image_alt", pa.string()),
        pa.field("image_width", pa.int32()),
        pa.field("image_height", pa.int32()),
        pa.field("metadata_completeness", pa.int16(), nullable=False),
        pa.field("warnings", pa.list_(pa.string()), nullable=False),
        pa.field("raw_metadata_json", pa.string(), nullable=False),
    ]
)

DUPLICATE_SCHEMA = pa.schema(
    [
        pa.field("article_key", pa.string(), nullable=False),
        pa.field("source_path", pa.string(), nullable=False),
        pa.field("winner_source_path", pa.string(), nullable=False),
        pa.field("duplicate_rank", pa.int32(), nullable=False),
        pa.field("reason", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
    ]
)

FAILURE_SCHEMA = pa.schema(
    [
        pa.field("failure_id", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("source_path", pa.string(), nullable=False),
        pa.field("stage", pa.string(), nullable=False),
        pa.field("error_code", pa.string(), nullable=False),
        pa.field("message", pa.string(), nullable=False),
        pa.field("extractor_version", pa.string(), nullable=False),
        pa.field("created_at", UTC_TIMESTAMP, nullable=False),
    ]
)

MISSING_IMAGE_SCHEMA = pa.schema(
    [
        pa.field("article_key", pa.string(), nullable=False),
        pa.field("source_path", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
    ]
)

RUN_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("command", pa.string(), nullable=False),
        pa.field("scope", pa.string(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("started_at", UTC_TIMESTAMP, nullable=False),
        pa.field("finished_at", UTC_TIMESTAMP),
        pa.field("summary_json", pa.string()),
    ]
)


@dataclass(frozen=True, slots=True)
class PublishSummary:
    """Counts for canonical partition publication."""

    partitions_written: int
    rows_written: int


def publish_partitions(
    state: PipelineState,
    config: PipelineConfig,
    partitions: list[PartitionKey] | tuple[PartitionKey, ...],
    run_id: str,
) -> PublishSummary:
    """Atomically rewrite only the supplied monthly partitions."""

    rows_written = 0
    unique_partitions = sorted(set(partitions))
    for partition in unique_partitions:
        rows = _canonical_rows(state, partition)
        table = pa.Table.from_pylist(rows, schema=ARTICLE_SCHEMA)
        if table.num_rows:
            table = table.sort_by(
                [
                    ("published_at_utc", "ascending"),
                    ("article_key", "ascending"),
                ]
            )
        _validate_partition_table(table, partition)
        target = (
            config.parquet_root
            / "articles"
            / f"publication_year_ny={partition.year:04d}"
            / f"publication_month_ny={partition.month:02d}"
            / "articles.parquet"
        )
        _write_atomic_table(table, target, config.staging_root, run_id)
        rows_written += table.num_rows

    return PublishSummary(
        partitions_written=len(unique_partitions),
        rows_written=rows_written,
    )


def publish_audits(
    state: PipelineState,
    config: PipelineConfig,
    run_id: str,
) -> None:
    """Publish deterministic, content-free operational audit datasets."""

    duplicate_rows = [
        {
            "article_key": row[0],
            "source_path": row[1],
            "winner_source_path": row[2],
            "duplicate_rank": row[3],
            "reason": row[4],
            "run_id": row[5],
        }
        for row in state.connection.execute(
            """
            SELECT article_key, source_path, winner_source_path,
                   duplicate_rank, reason, run_id
            FROM duplicates
            ORDER BY article_key, duplicate_rank, source_path
            """
        ).fetchall()
    ]
    failure_rows = [
        {
            "failure_id": row[0],
            "run_id": row[1],
            "source_path": row[2],
            "stage": row[3],
            "error_code": row[4],
            "message": row[5],
            "extractor_version": row[6],
            "created_at": row[7],
        }
        for row in state.connection.execute(
            """
            SELECT failure_id, run_id, source_path, stage, error_code,
                   message, extractor_version, created_at
            FROM failures
            ORDER BY created_at, failure_id
            """
        ).fetchall()
    ]
    missing_image_rows = _missing_image_rows(state, run_id)
    run_rows = [
        {
            "run_id": row[0],
            "command": row[1],
            "scope": row[2],
            "status": row[3],
            "started_at": row[4],
            "finished_at": row[5],
            "summary_json": row[6],
        }
        for row in state.connection.execute(
            """
            SELECT run_id, command, scope, status, started_at,
                   finished_at, summary_json
            FROM runs
            ORDER BY started_at, run_id
            """
        ).fetchall()
    ]

    exports = (
        ("duplicates", duplicate_rows, DUPLICATE_SCHEMA),
        ("failures", failure_rows, FAILURE_SCHEMA),
        ("missing_images", missing_image_rows, MISSING_IMAGE_SCHEMA),
        ("runs", run_rows, RUN_SCHEMA),
    )
    for name, rows, schema in exports:
        table = pa.Table.from_pylist(rows, schema=schema)
        target = config.parquet_root / "audit" / f"{name}.parquet"
        _write_atomic_table(table, target, config.staging_root, run_id)


def _canonical_rows(
    state: PipelineState,
    partition: PartitionKey,
) -> list[dict[str, object]]:
    stored = state.connection.execute(
        """
        SELECT c.article_key, e.payload_json
        FROM canonical_sources AS c
        JOIN extracted_sources AS e
          ON e.source_path = c.source_path
        WHERE c.publication_year_ny = ?
          AND c.publication_month_ny = ?
        ORDER BY e.published_at_utc, c.article_key
        """,
        [partition.year, partition.month],
    ).fetchall()
    return [
        _article_row(article_key, deserialize_article(payload_json))
        for article_key, payload_json in stored
    ]


def _article_row(
    article_key: str,
    article: ExtractedArticle,
) -> dict[str, object]:
    return {
        "schema_version": ARTICLE_SCHEMA_VERSION,
        "article_key": article_key,
        "wsj_article_id": article.wsj_article_id,
        "canonical_url": article.canonical_url,
        "source_path": article.relative_html_path,
        "source_html_sha256": article.source_html_sha256,
        "extractor_version": article.extractor_version,
        "headline": article.headline,
        "original_headline": article.original_headline,
        "summary": article.summary,
        "authors": list(article.authors),
        "keywords": list(article.keywords),
        "section": article.section,
        "page": article.page,
        "article_type": article.article_type,
        "access": article.access,
        "published_at_utc": article.published_at_utc,
        "publication_date_utc": article.publication_date_utc,
        "published_at_new_york": article.published_at_new_york,
        "publication_date_new_york": article.publication_date_new_york,
        "publication_year_ny": article.publication_year_ny,
        "publication_month_ny": article.publication_month_ny,
        "updated_at_utc": article.updated_at_utc,
        "created_at_utc": article.created_at_utc,
        "last_published_at_utc": article.last_published_at_utc,
        "body_paragraphs": list(article.body_paragraphs),
        "body_text": article.body_text,
        "body_word_count": article.body_word_count,
        "image_path": article.relative_image_path,
        "image_present": article.image_present,
        "image_media_type": article.image_media_type,
        "image_size_bytes": article.image_size_bytes,
        "image_caption": article.image_caption,
        "image_credit": article.image_credit,
        "image_creator": article.image_creator,
        "image_alt": article.image_alt,
        "image_width": article.image_width,
        "image_height": article.image_height,
        "metadata_completeness": article.metadata_completeness,
        "warnings": list(article.warnings),
        "raw_metadata_json": article.raw_metadata_json,
    }


def _missing_image_rows(
    state: PipelineState,
    run_id: str,
) -> list[dict[str, object]]:
    stored = state.connection.execute(
        """
        SELECT c.article_key, e.source_path, e.payload_json
        FROM canonical_sources AS c
        JOIN extracted_sources AS e
          ON e.source_path = c.source_path
        ORDER BY c.article_key
        """
    ).fetchall()
    return [
        {
            "article_key": article_key,
            "source_path": source_path,
            "run_id": run_id,
        }
        for article_key, source_path, payload_json in stored
        if not deserialize_article(payload_json).image_present
    ]


def _validate_partition_table(
    table: pa.Table,
    partition: PartitionKey,
) -> None:
    if not table.schema.equals(ARTICLE_SCHEMA):
        raise ValueError("article table has an incompatible schema")
    keys = table.column("article_key").to_pylist()
    if len(keys) != len(set(keys)):
        raise ValueError("article partition contains duplicate keys")
    years = set(table.column("publication_year_ny").to_pylist())
    months = set(table.column("publication_month_ny").to_pylist())
    if years - {partition.year} or months - {partition.month}:
        raise ValueError("article date does not match target partition")


def _write_atomic_table(
    table: pa.Table,
    target: Path,
    staging_root: Path,
    run_id: str,
) -> None:
    relative_token = target.as_posix().replace("/", "__")
    run_staging = staging_root / run_id
    staged = run_staging / relative_token
    run_staging.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        pq.write_table(table, staged, compression="zstd")
        reopened = pq.read_table(staged, partitioning=None)
        if reopened.num_rows != table.num_rows or not reopened.schema.equals(
            table.schema
        ):
            raise ValueError("staged Parquet validation failed")
        os.replace(staged, target)
    finally:
        if staged.exists():
            staged.unlink()
        with suppress(OSError):
            run_staging.rmdir()
