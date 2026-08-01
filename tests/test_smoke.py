from __future__ import annotations

import importlib.util
import json
from datetime import date
from pathlib import Path

import duckdb
import pytest

import wsj_pipeline.pipeline as pipeline_module
from wsj_pipeline.cli import main
from wsj_pipeline.config import PipelineConfig
from wsj_pipeline.validate import validate_outputs


def test_legacy_parquet_modules_are_absent() -> None:
    assert importlib.util.find_spec("wsj_pipeline.publish") is None
    assert importlib.util.find_spec("wsj_pipeline.index") is None


def test_fixture_smoke_proves_the_complete_markdown_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def inspect_then_validate(config: PipelineConfig):
        with duckdb.connect(str(config.catalog_db), read_only=True) as connection:
            observed["article_count"] = connection.execute(
                "SELECT count(*) FROM articles"
            ).fetchone()[0]
            observed["source_count"] = connection.execute(
                "SELECT count(*) FROM source_manifest"
            ).fetchone()[0]
            observed["duplicate_count"] = connection.execute(
                "SELECT count(*) FROM duplicates"
            ).fetchone()[0]
            observed["failure_count"] = connection.execute(
                "SELECT count(*) FROM failures"
            ).fetchone()[0]
            observed["inline_urls"] = connection.execute(
                """
                SELECT article_id, inline_image_urls
                FROM articles
                ORDER BY article_id
                """
            ).fetchall()
        observed["markdown_paths"] = [
            path.relative_to(config.output_root).as_posix()
            for path in sorted(config.text_root.rglob("*.md"))
        ]
        report = validate_outputs(config)
        observed["validation_issues"] = report.issues
        return report

    monkeypatch.setattr(pipeline_module, "validate_outputs", inspect_then_validate)

    summary = pipeline_module.run_smoke()

    assert summary.discovered == 3
    assert summary.articles == 2
    assert summary.duplicate_count == 1
    assert summary.failed == 0
    assert summary.validation_ok is True
    assert observed == {
        "article_count": 2,
        "source_count": 3,
        "duplicate_count": 1,
        "failure_count": 0,
        "inline_urls": [
            ("wsj:SMOKE-A", ["https://images.wsj.net/smoke-a.jpg"]),
            ("wsj:SMOKE-B", ["https://images.wsj.net/smoke-b.jpg"]),
        ],
        "markdown_paths": [
            pipeline_module.article_markdown_path(
                "wsj:SMOKE-A",
                date(2024, 1, 2),
            ).as_posix(),
            pipeline_module.article_markdown_path(
                "wsj:SMOKE-B",
                date(2025, 2, 3),
            ).as_posix(),
        ],
        "validation_issues": (),
    }


def test_cli_smoke_never_loads_default_or_explicit_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def refuse_config_load(cls, path: Path):
        raise AssertionError(f"smoke tried to load configured archive: {path}")

    monkeypatch.setattr(PipelineConfig, "from_toml", classmethod(refuse_config_load))

    exit_code = main(
        ["--config", str(tmp_path / "must-not-open.toml"), "smoke"]
    )
    line = capsys.readouterr().out
    payload = json.loads(line)

    assert exit_code == 0
    assert payload["articles"] == 2
    assert payload["discovered"] == 3
    assert payload["duplicate_count"] == 1
    assert payload["failed"] == 0
    assert payload["validation_ok"] is True
    assert line == f"{json.dumps(payload, sort_keys=True)}\n"
