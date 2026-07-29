from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from tests.test_publish import build_published_state
from wsj_pipeline.canonical import PartitionKey
from wsj_pipeline.publish import publish_partitions
from wsj_pipeline.state import PipelineState


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_failed_atomic_replace_preserves_existing_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, run_id, canonical = build_published_state(tmp_path)
    january = PartitionKey(2024, 1)
    with PipelineState.open(config.state_db) as state:
        publish_partitions(state, config, [january], run_id)
    target = (
        config.parquet_root
        / "articles"
        / "publication_year_ny=2024"
        / "publication_month_ny=01"
        / "articles.parquet"
    )
    original_hash = file_hash(target)
    real_replace = os.replace

    def fail_replace(source: Path, destination: Path) -> None:
        if Path(destination) == target:
            raise OSError("simulated replace failure")
        real_replace(source, destination)

    monkeypatch.setattr("wsj_pipeline.publish.os.replace", fail_replace)
    with (
        PipelineState.open(config.state_db) as state,
        pytest.raises(OSError, match="simulated replace failure"),
    ):
        publish_partitions(state, config, [january], f"{run_id}-failed")

    assert file_hash(target) == original_hash
    assert list(config.staging_root.rglob("*.parquet")) == []
    assert canonical.affected_partitions == (
        PartitionKey(2023, 12),
        PartitionKey(2024, 1),
    )
