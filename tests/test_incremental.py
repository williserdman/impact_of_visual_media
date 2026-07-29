from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb

from tests.fixtures import write_article_fixture
from wsj_pipeline.config import PipelineConfig
from wsj_pipeline.discovery import discover_sources
from wsj_pipeline.state import PipelineState, process_candidates


def config_for(tmp_path: Path, archive: Path) -> PipelineConfig:
    return PipelineConfig(
        source_root=archive,
        output_root=tmp_path / "processed",
        batch_size=2,
    )


def run_process(config: PipelineConfig):
    with PipelineState.open(config.state_db) as state:
        run_id = state.begin_run("process", "limit:fixture")
    summary = process_candidates(
        config,
        discover_sources(config.source_root),
        run_id,
    )
    with PipelineState.open(config.state_db) as state:
        state.finish_run(run_id, status="succeeded", summary=summary.as_dict())
    return summary


def test_unchanged_second_run_performs_no_extraction(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    write_article_fixture(archive, "2024/2024/one.html")
    write_article_fixture(archive, "2024/2024/two.html")
    config = config_for(tmp_path, archive)

    first = run_process(config)
    second = run_process(config)

    assert first.discovered == 2
    assert first.new == 2
    assert first.succeeded == 2
    assert second.discovered == 2
    assert second.unchanged == 2
    assert second.succeeded == 0
    with duckdb.connect(str(config.state_db), read_only=True) as connection:
        assert (
            connection.execute("SELECT count(*) FROM extracted_sources").fetchone()[0]
            == 2
        )


def test_mtime_only_change_updates_manifest_without_replacing_payload(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    html_path = write_article_fixture(archive, "2016/one.html")
    config = config_for(tmp_path, archive)
    run_process(config)
    with duckdb.connect(str(config.state_db), read_only=True) as connection:
        before = connection.execute(
            """
            SELECT payload_json, html_mtime_ns
            FROM extracted_sources
            JOIN source_manifest USING (source_path)
            """
        ).fetchone()

    stat = html_path.stat()
    os.utime(html_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    summary = run_process(config)

    with duckdb.connect(str(config.state_db), read_only=True) as connection:
        after = connection.execute(
            """
            SELECT payload_json, html_mtime_ns
            FROM extracted_sources
            JOIN source_manifest USING (source_path)
            """
        ).fetchone()
    assert summary.metadata_only == 1
    assert summary.succeeded == 0
    assert json.loads(after[0]) == json.loads(before[0])
    assert after[1] != before[1]


def test_content_change_replaces_one_extracted_payload(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    html_path = write_article_fixture(archive, "2016/one.html")
    write_article_fixture(archive, "2016/two.html")
    config = config_for(tmp_path, archive)
    run_process(config)

    html_path.write_text(
        html_path.read_text(encoding="utf-8").replace(
            "First paragraph.", "A materially changed first paragraph."
        ),
        encoding="utf-8",
    )
    summary = run_process(config)

    assert summary.changed == 1
    assert summary.unchanged == 1
    assert summary.succeeded == 1
    with duckdb.connect(str(config.state_db), read_only=True) as connection:
        payload = connection.execute(
            """
            SELECT payload_json FROM extracted_sources
            WHERE source_path = '2016/one.html'
            """
        ).fetchone()[0]
    assert "materially changed" in json.loads(payload)["body_text"]


def test_malformed_source_does_not_stop_later_candidate(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    write_article_fixture(archive, "2016/a-malformed.html", paragraphs=())
    write_article_fixture(archive, "2016/b-valid.html")
    config = config_for(tmp_path, archive)

    summary = run_process(config)

    assert summary.failed == 1
    assert summary.succeeded == 1
    with duckdb.connect(str(config.state_db), read_only=True) as connection:
        failure = connection.execute(
            "SELECT source_path, error_code, message FROM failures"
        ).fetchone()
        extracted_paths = connection.execute(
            "SELECT source_path FROM extracted_sources"
        ).fetchall()
    assert failure[0:2] == ("2016/a-malformed.html", "missing_body")
    assert "First paragraph" not in failure[2]
    assert extracted_paths == [("2016/b-valid.html",)]


def test_stale_extractor_requires_explicit_reprocess(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    write_article_fixture(archive, "2016/one.html")
    config = config_for(tmp_path, archive)
    run_process(config)
    with duckdb.connect(str(config.state_db)) as connection:
        connection.execute(
            "UPDATE source_manifest SET extractor_version = 'older'"
        )

    with PipelineState.open(config.state_db) as state:
        skipped_run_id = state.begin_run("process", "limit:fixture")
    skipped = process_candidates(
        config,
        discover_sources(config.source_root),
        skipped_run_id,
    )
    with PipelineState.open(config.state_db) as state:
        reprocess_run_id = state.begin_run("process", "limit:fixture,reprocess")
    reprocessed = process_candidates(
        config,
        discover_sources(config.source_root),
        reprocess_run_id,
        reprocess_stale=True,
    )

    assert skipped.stale == 1
    assert skipped.succeeded == 0
    assert reprocessed.reprocessed == 1
    assert reprocessed.succeeded == 1


def test_image_only_change_refreshes_extracted_image_fields(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    html_path = write_article_fixture(
        archive,
        "2016/one.html",
        with_image=False,
    )
    config = config_for(tmp_path, archive)
    run_process(config)
    html_path.with_name(f"{html_path.stem}_main_image.webp").write_bytes(
        b"new-image"
    )

    summary = run_process(config)

    with duckdb.connect(str(config.state_db), read_only=True) as connection:
        payload = json.loads(
            connection.execute(
                "SELECT payload_json FROM extracted_sources"
            ).fetchone()[0]
        )
    assert summary.changed == 1
    assert summary.succeeded == 1
    assert summary.metadata_only == 0
    assert payload["image_present"] is True
    assert payload["relative_image_path"] == "2016/one_main_image.webp"
    assert payload["image_size_bytes"] == len(b"new-image")
