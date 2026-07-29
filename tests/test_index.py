from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from tests.test_publish import build_published_state
from wsj_pipeline.index import build_index
from wsj_pipeline.publish import publish_audits, publish_partitions
from wsj_pipeline.state import PipelineState


def build_indexed_outputs(tmp_path: Path):
    config, run_id, canonical = build_published_state(tmp_path)
    with PipelineState.open(config.state_db) as state:
        publish_partitions(
            state,
            config,
            canonical.affected_partitions,
            run_id,
        )
        publish_audits(state, config, run_id)
    summary = build_index(config, run_id)
    return config, run_id, summary


def test_index_exposes_articles_dates_counts_and_audits(tmp_path: Path) -> None:
    config, _run_id, summary = build_indexed_outputs(tmp_path)

    with duckdb.connect(str(config.index_db), read_only=True) as connection:
        relations = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
        exact_date = connection.execute(
            """
            SELECT article_key
            FROM publication_index
            WHERE publication_date_new_york = DATE '2023-12-31'
            ORDER BY published_at_utc, article_key
            """
        ).fetchall()
        date_range = connection.execute(
            """
            SELECT article_key
            FROM publication_index
            WHERE publication_date_new_york
                  BETWEEN DATE '2023-12-31' AND DATE '2024-01-31'
            ORDER BY publication_date_new_york, published_at_utc, article_key
            """
        ).fetchall()
        daily_counts = connection.execute(
            """
            SELECT publication_date_new_york, article_count
            FROM daily_publication_counts
            ORDER BY publication_date_new_york
            """
        ).fetchall()
        index_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info('publication_index')"
            ).fetchall()
        }
        duplicate_count = connection.execute(
            "SELECT count(*) FROM duplicates"
        ).fetchone()[0]

    assert {
        "articles",
        "publication_index",
        "daily_publication_counts",
        "duplicates",
        "failures",
        "missing_images",
        "runs",
    }.issubset(relations)
    assert exact_date == [("wsj:UTC-LATE",)]
    assert date_range == [("wsj:UTC-LATE",), ("wsj:JANUARY",)]
    assert [count for _date, count in daily_counts] == [1, 1]
    assert "body_text" not in index_columns
    assert duplicate_count == 1
    assert summary.article_count == 2


def test_rebuilt_index_replaces_prior_database(tmp_path: Path) -> None:
    config, run_id, _summary = build_indexed_outputs(tmp_path)
    with duckdb.connect(str(config.index_db)) as connection:
        connection.execute("CREATE TABLE stale_table AS SELECT 'old' AS marker")

    build_index(config, f"{run_id}-rebuilt")

    with duckdb.connect(str(config.index_db), read_only=True) as connection:
        tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
    assert "stale_table" not in tables
    assert list(config.staging_root.rglob("*.duckdb")) == []


def test_failed_index_build_cleans_staged_database(tmp_path: Path) -> None:
    config, run_id, _summary = build_indexed_outputs(tmp_path)
    (config.parquet_root / "audit" / "duplicates.parquet").unlink()

    with pytest.raises(duckdb.IOException):
        build_index(config, f"{run_id}-failed")

    assert list(config.staging_root.rglob("*.duckdb")) == []
