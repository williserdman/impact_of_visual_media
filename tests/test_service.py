from __future__ import annotations

import hashlib
import os
from pathlib import Path

import duckdb
import pyarrow.parquet as pq
import pytest

import wsj_pipeline.service as service_module
import wsj_pipeline.state as state_module
from tests.fixtures import write_article_fixture
from wsj_pipeline.config import PipelineConfig
from wsj_pipeline.service import (
    PipelineLockedError,
    index_only,
    pipeline_lock,
    process_only,
    publish_all,
    run_incremental,
)


def parquet_hashes(config: PipelineConfig) -> dict[str, str]:
    return {
        path.relative_to(config.parquet_root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted((config.parquet_root / "articles").rglob("articles.parquet"))
    }


def service_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        source_root=tmp_path / "archive",
        output_root=tmp_path / "processed",
        batch_size=2,
    )


def test_unchanged_end_to_end_rerun_preserves_article_partitions(
    tmp_path: Path,
) -> None:
    config = service_config(tmp_path)
    write_article_fixture(
        config.source_root,
        "2024/2024/a.html",
        article_id="A",
    )
    write_article_fixture(
        config.source_root,
        "2024/2024/b.html",
        article_id="B",
    )

    first = run_incremental(config, limit=None, full=True)
    before_hashes = parquet_hashes(config)
    second = run_incremental(config, limit=None, full=True)

    assert first.processing.succeeded == 2
    assert first.validation_ok
    assert second.processing.succeeded == 0
    assert second.processing.unchanged == 2
    assert second.partitions_written == 0
    assert parquet_hashes(config) == before_hashes


def test_adding_future_year_processes_only_new_source(tmp_path: Path) -> None:
    config = service_config(tmp_path)
    write_article_fixture(
        config.source_root,
        "2024/2024/a.html",
        article_id="A",
    )
    run_incremental(config, limit=None, full=True)
    write_article_fixture(
        config.source_root,
        "2025/2025/new.html",
        article_id="NEW",
        published="2025-04-03T14:00:00Z",
    )

    summary = run_incremental(config, limit=None, full=True)

    assert summary.processing.new == 1
    assert summary.processing.succeeded == 1
    assert summary.processing.unchanged == 1
    assert summary.article_count == 2
    assert (
        config.parquet_root
        / "articles"
        / "publication_year_ny=2025"
        / "publication_month_ny=04"
        / "articles.parquet"
    ).is_file()


def test_existing_lock_prevents_concurrent_mutation(tmp_path: Path) -> None:
    lock_path = tmp_path / "state" / "pipeline.lock"

    with (
        pipeline_lock(lock_path),
        pytest.raises(PipelineLockedError, match="already running"),
        pipeline_lock(lock_path),
    ):
        pass

    assert not lock_path.exists()


def test_manual_index_rebuild_respects_pipeline_lock(tmp_path: Path) -> None:
    config = service_config(tmp_path)
    lock_path = config.output_root / "state" / "pipeline.lock"

    with (
        pipeline_lock(lock_path),
        pytest.raises(PipelineLockedError, match="already running"),
    ):
        index_only(config)


def test_committed_extraction_batches_are_canonicalized_after_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = service_config(tmp_path)
    config = PipelineConfig(
        source_root=config.source_root,
        output_root=config.output_root,
        batch_size=1,
    )
    for article_id in ("A", "B", "C"):
        write_article_fixture(
            config.source_root,
            f"2024/{article_id.lower()}.html",
            article_id=article_id,
        )
    real_extract = state_module.extract_article
    calls = 0

    def fail_second_extraction(candidate, source_root):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated batch interruption")
        return real_extract(candidate, source_root)

    monkeypatch.setattr(state_module, "extract_article", fail_second_extraction)
    with pytest.raises(RuntimeError, match="simulated batch interruption"):
        run_incremental(config, limit=None, full=True)
    monkeypatch.setattr(state_module, "extract_article", real_extract)

    summary = run_incremental(config, limit=None, full=True)

    with duckdb.connect(str(config.state_db), read_only=True) as connection:
        canonical_count = connection.execute(
            "SELECT count(*) FROM canonical_sources"
        ).fetchone()[0]
    assert summary.article_count == 3
    assert canonical_count == 3


def test_failed_partition_publication_is_retried_on_unchanged_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = service_config(tmp_path)
    html_path = write_article_fixture(
        config.source_root,
        "2024/a.html",
        article_id="A",
        paragraphs=("Original body.",),
    )
    run_incremental(config, limit=None, full=True)
    html_path.write_text(
        html_path.read_text(encoding="utf-8").replace(
            "Original body.", "Changed body after interruption."
        ),
        encoding="utf-8",
    )
    real_replace = os.replace

    def fail_article_replace(source, destination):
        if Path(destination).name == "articles.parquet":
            raise OSError("simulated partition interruption")
        real_replace(source, destination)

    monkeypatch.setattr("wsj_pipeline.publish.os.replace", fail_article_replace)
    with pytest.raises(OSError, match="simulated partition interruption"):
        run_incremental(config, limit=None, full=True)
    monkeypatch.setattr("wsj_pipeline.publish.os.replace", real_replace)
    with duckdb.connect(str(config.state_db), read_only=True) as connection:
        assert connection.execute(
            "SELECT count(*) FROM dirty_partitions"
        ).fetchone()[0] == 1

    summary = run_incremental(config, limit=None, full=True)
    [article_path] = sorted(
        (config.parquet_root / "articles").rglob("articles.parquet")
    )
    [row] = pq.read_table(article_path, partitioning=None).to_pylist()

    assert summary.partitions_written == 1
    assert row["body_text"] == "Changed body after interruption."
    with duckdb.connect(str(config.state_db), read_only=True) as connection:
        assert connection.execute(
            "SELECT count(*) FROM dirty_partitions"
        ).fetchone()[0] == 0


def test_full_run_removes_sources_missing_from_archive(tmp_path: Path) -> None:
    config = service_config(tmp_path)
    write_article_fixture(
        config.source_root,
        "2024/a.html",
        article_id="A",
    )
    removed_html = write_article_fixture(
        config.source_root,
        "2024/b.html",
        article_id="B",
    )
    run_incremental(config, limit=None, full=True)
    removed_html.unlink()
    removed_html.with_name("b_main_image.jpg").unlink()

    summary = run_incremental(config, limit=None, full=True)

    [article_path] = sorted(
        (config.parquet_root / "articles").rglob("articles.parquet")
    )
    rows = pq.read_table(article_path, partitioning=None).to_pylist()
    assert summary.processing.removed == 1
    assert summary.article_count == 1
    assert [row["article_key"] for row in rows] == ["wsj:A"]
    with duckdb.connect(str(config.state_db), read_only=True) as connection:
        assert connection.execute(
            """
            SELECT status, article_key
            FROM source_manifest
            WHERE source_path = '2024/b.html'
            """
        ).fetchone() == ("missing", None)


def test_limited_run_does_not_reconcile_unseen_sources(tmp_path: Path) -> None:
    config = service_config(tmp_path)
    write_article_fixture(config.source_root, "2024/a.html", article_id="A")
    write_article_fixture(config.source_root, "2024/b.html", article_id="B")
    run_incremental(config, limit=None, full=True)

    summary = run_incremental(config, limit=1, full=False)

    assert summary.processing.removed == 0
    assert summary.article_count == 2


def test_publish_failure_closes_run_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = service_config(tmp_path)
    with state_module.PipelineState.open(config.state_db):
        pass

    def fail_recompute(*_args, **_kwargs):
        raise RuntimeError("simulated canonical failure")

    monkeypatch.setattr(service_module, "recompute_canonical", fail_recompute)

    with pytest.raises(RuntimeError, match="simulated canonical failure"):
        publish_all(config)

    with duckdb.connect(str(config.state_db), read_only=True) as connection:
        assert connection.execute(
            """
            SELECT status
            FROM runs
            WHERE command = 'publish'
            ORDER BY started_at DESC
            LIMIT 1
            """
        ).fetchone()[0] == "failed"


def test_publish_all_drains_pending_removed_article_keys(
    tmp_path: Path,
) -> None:
    config = service_config(tmp_path)
    write_article_fixture(config.source_root, "2024/a.html", article_id="A")
    removed_html = write_article_fixture(
        config.source_root,
        "2024/b.html",
        article_id="B",
    )
    run_incremental(config, limit=None, full=True)
    removed_html.unlink()
    removed_html.with_name("b_main_image.jpg").unlink()
    process_only(config, limit=None, full=True)

    publish_all(config)

    [article_path] = sorted(
        (config.parquet_root / "articles").rglob("articles.parquet")
    )
    rows = pq.read_table(article_path, partitioning=None).to_pylist()
    assert [row["article_key"] for row in rows] == ["wsj:A"]
    with duckdb.connect(str(config.state_db), read_only=True) as connection:
        assert connection.execute(
            "SELECT count(*) FROM pending_article_keys"
        ).fetchone()[0] == 0
