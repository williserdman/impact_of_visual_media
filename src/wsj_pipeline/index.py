"""Transactional DuckDB publication-date query index."""

from __future__ import annotations

import os
from contextlib import suppress
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb

from wsj_pipeline.config import PipelineConfig


@dataclass(frozen=True, slots=True)
class IndexSummary:
    """Counts and bounds for a rebuilt query index."""

    article_count: int
    first_publication_date_new_york: date | None
    last_publication_date_new_york: date | None


def build_index(config: PipelineConfig, run_id: str) -> IndexSummary:
    """Build a complete temporary index and atomically replace the prior one."""

    article_glob = (
        config.parquet_root
        / "articles"
        / "publication_year_ny=*"
        / "publication_month_ny=*"
        / "articles.parquet"
    )
    audit_root = config.parquet_root / "audit"
    run_staging = config.staging_root / run_id
    staged = run_staging / "wsj.duckdb"
    run_staging.mkdir(parents=True, exist_ok=True)
    config.index_db.parent.mkdir(parents=True, exist_ok=True)
    if staged.exists():
        staged.unlink()

    connection: duckdb.DuckDBPyConnection | None = None
    try:
        connection = duckdb.connect(str(staged))
        connection.execute(
            f"""
            CREATE VIEW articles AS
            SELECT *
            FROM read_parquet(
                '{_sql_path(article_glob)}',
                hive_partitioning = true,
                union_by_name = true
            )
            """
        )
        for relation in ("duplicates", "failures", "missing_images", "runs"):
            path = audit_root / f"{relation}.parquet"
            connection.execute(
                f"""
                CREATE VIEW {relation} AS
                SELECT * FROM read_parquet('{_sql_path(path)}')
                """
            )
        connection.execute(
            """
            CREATE TABLE publication_index AS
            SELECT article_key,
                   source_path,
                   published_at_utc,
                   publication_date_utc,
                   published_at_new_york,
                   publication_date_new_york
            FROM articles
            ORDER BY publication_date_new_york, published_at_utc, article_key
            """
        )
        connection.execute(
            """
            CREATE UNIQUE INDEX publication_index_article_key
            ON publication_index(article_key)
            """
        )
        connection.execute(
            """
            CREATE INDEX publication_index_date
            ON publication_index(publication_date_new_york)
            """
        )
        connection.execute(
            """
            CREATE VIEW daily_publication_counts AS
            SELECT publication_date_new_york, count(*)::BIGINT AS article_count
            FROM publication_index
            GROUP BY publication_date_new_york
            """
        )
        row = connection.execute(
            """
            SELECT count(*),
                   min(publication_date_new_york),
                   max(publication_date_new_york)
            FROM publication_index
            """
        ).fetchone()
        summary = IndexSummary(
            article_count=row[0],
            first_publication_date_new_york=row[1],
            last_publication_date_new_york=row[2],
        )
        connection.execute("CHECKPOINT")
        connection.close()
        connection = None
        os.replace(staged, config.index_db)
    finally:
        if connection is not None:
            connection.close()
        if staged.exists():
            staged.unlink()
        with suppress(OSError):
            run_staging.rmdir()
    return summary


def _sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")
