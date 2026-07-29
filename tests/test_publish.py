from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tests.fixtures import write_article_fixture
from wsj_pipeline.canonical import recompute_canonical
from wsj_pipeline.config import PipelineConfig
from wsj_pipeline.discovery import discover_sources
from wsj_pipeline.extract import extract_article
from wsj_pipeline.publish import (
    ARTICLE_SCHEMA,
    ARTICLE_SCHEMA_VERSION,
    publish_audits,
    publish_partitions,
)
from wsj_pipeline.state import PipelineState


def build_published_state(tmp_path: Path):
    archive = tmp_path / "archive"
    write_article_fixture(
        archive,
        "2023/late-utc.html",
        article_id="UTC-LATE",
        published="2024-01-01T02:30:00Z",
        with_image=False,
    )
    write_article_fixture(
        archive,
        "2024/2024/january.html",
        article_id="JANUARY",
        published="2024-01-15T15:00:00Z",
    )
    write_article_fixture(
        archive,
        "2024/2024/january-duplicate.html",
        article_id="JANUARY",
        published="2024-01-15T15:00:00Z",
        paragraphs=("Short.",),
    )
    config = PipelineConfig(
        source_root=archive,
        output_root=tmp_path / "processed",
        batch_size=10,
    )
    with PipelineState.open(config.state_db) as state:
        run_id = state.begin_run("test", "fixture")
        article_keys = []
        for candidate in discover_sources(archive):
            article_keys.append(
                state.store_success(
                    run_id,
                    candidate,
                    extract_article(candidate, archive),
                )
            )
        canonical = recompute_canonical(state, article_keys, run_id)
    return config, run_id, canonical


def test_article_schema_uses_native_temporal_and_list_types() -> None:
    assert ARTICLE_SCHEMA.field("schema_version").type == pa.string()
    assert ARTICLE_SCHEMA.field("published_at_utc").type == pa.timestamp("us", tz="UTC")
    assert ARTICLE_SCHEMA.field("published_at_new_york").type == pa.timestamp(
        "us", tz="America/New_York"
    )
    assert ARTICLE_SCHEMA.field("publication_date_utc").type == pa.date32()
    assert ARTICLE_SCHEMA.field("publication_date_new_york").type == pa.date32()
    assert ARTICLE_SCHEMA.field("authors").type == pa.list_(pa.string())
    assert ARTICLE_SCHEMA.field("keywords").type == pa.list_(pa.string())
    assert ARTICLE_SCHEMA.field("body_paragraphs").type == pa.list_(pa.string())
    assert ARTICLE_SCHEMA.field("warnings").type == pa.list_(pa.string())


def test_publishes_sorted_hive_style_monthly_partitions(tmp_path: Path) -> None:
    config, run_id, canonical = build_published_state(tmp_path)

    with PipelineState.open(config.state_db) as state:
        summary = publish_partitions(
            state,
            config,
            canonical.affected_partitions,
            run_id,
        )

    december_path = (
        config.parquet_root
        / "articles"
        / "publication_year_ny=2023"
        / "publication_month_ny=12"
        / "articles.parquet"
    )
    january_path = (
        config.parquet_root
        / "articles"
        / "publication_year_ny=2024"
        / "publication_month_ny=01"
        / "articles.parquet"
    )
    assert summary.partitions_written == 2
    assert summary.rows_written == 2
    assert december_path.is_file()
    assert january_path.is_file()

    december = pq.read_table(
        december_path,
        partitioning=None,
    ).to_pylist()
    january = pq.read_table(january_path, partitioning=None).to_pylist()
    assert [row["article_key"] for row in december] == ["wsj:UTC-LATE"]
    assert december[0]["publication_date_utc"].isoformat() == "2024-01-01"
    assert december[0]["publication_date_new_york"].isoformat() == "2023-12-31"
    assert [row["article_key"] for row in january] == ["wsj:JANUARY"]
    assert january[0]["schema_version"] == ARTICLE_SCHEMA_VERSION
    assert january[0]["authors"] == ["Alice Reporter", "Bob Editor"]


def test_publishes_content_free_audit_datasets(tmp_path: Path) -> None:
    config, run_id, canonical = build_published_state(tmp_path)

    with PipelineState.open(config.state_db) as state:
        publish_partitions(
            state,
            config,
            canonical.affected_partitions,
            run_id,
        )
        publish_audits(state, config, run_id)

    audit_root = config.parquet_root / "audit"
    duplicates = pq.read_table(audit_root / "duplicates.parquet").to_pylist()
    missing_images = pq.read_table(audit_root / "missing_images.parquet").to_pylist()
    failures_schema = pq.read_schema(audit_root / "failures.parquet")

    assert duplicates[0]["article_key"] == "wsj:JANUARY"
    assert duplicates[0]["reason"] == "shorter_body"
    assert missing_images == [
        {
            "article_key": "wsj:UTC-LATE",
            "source_path": "2023/late-utc.html",
            "run_id": run_id,
        }
    ]
    for forbidden in ("body", "body_text", "payload_json"):
        assert forbidden not in failures_schema.names
