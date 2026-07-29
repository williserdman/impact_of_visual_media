from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tests.fixtures import write_article_fixture
from wsj_pipeline.config import PipelineConfig
from wsj_pipeline.service import (
    PipelineLockedError,
    pipeline_lock,
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
