from __future__ import annotations

from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from tests.test_index import build_indexed_outputs
from wsj_pipeline.validate import validate_outputs


def article_files(config) -> list[Path]:
    return sorted((config.parquet_root / "articles").rglob("articles.parquet"))


def test_clean_outputs_validate_successfully(tmp_path: Path) -> None:
    config, _run_id, _summary = build_indexed_outputs(tmp_path)

    report = validate_outputs(config)

    assert report.ok
    assert report.issues == ()


def test_validation_detects_duplicate_article_keys(tmp_path: Path) -> None:
    config, _run_id, _summary = build_indexed_outputs(tmp_path)
    january = article_files(config)[1]
    table = pq.read_table(january, partitioning=None)
    pq.write_table(pa.concat_tables([table, table.slice(0, 1)]), january)

    report = validate_outputs(config)

    assert "duplicate_article_key" in {issue.code for issue in report.issues}


def test_validation_detects_partition_date_mismatch(tmp_path: Path) -> None:
    config, _run_id, _summary = build_indexed_outputs(tmp_path)
    january = article_files(config)[1]
    table = pq.read_table(january, partitioning=None)
    wrong_year = pa.array([1999] * table.num_rows, type=pa.int32())
    table = table.set_column(
        table.schema.get_field_index("publication_year_ny"),
        "publication_year_ny",
        wrong_year,
    )
    pq.write_table(table, january)

    report = validate_outputs(config)

    assert "partition_date_mismatch" in {issue.code for issue in report.issues}


def test_validation_detects_broken_duplicate_winner_reference(
    tmp_path: Path,
) -> None:
    config, _run_id, _summary = build_indexed_outputs(tmp_path)
    duplicate_path = config.parquet_root / "audit" / "duplicates.parquet"
    rows = pq.read_table(duplicate_path).to_pylist()
    rows[0]["winner_source_path"] = "missing/winner.html"
    pq.write_table(pa.Table.from_pylist(rows), duplicate_path)

    report = validate_outputs(config)

    assert "invalid_duplicate_winner" in {issue.code for issue in report.issues}


def test_validation_detects_stale_index_keys(tmp_path: Path) -> None:
    config, _run_id, _summary = build_indexed_outputs(tmp_path)
    with duckdb.connect(str(config.index_db)) as connection:
        connection.execute(
            "DELETE FROM publication_index WHERE article_key = 'wsj:JANUARY'"
        )

    report = validate_outputs(config)

    assert "index_key_mismatch" in {issue.code for issue in report.issues}
