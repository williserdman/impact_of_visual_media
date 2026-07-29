from __future__ import annotations

import json
from pathlib import Path

from tests.fixtures import write_article_fixture
from wsj_pipeline.cli import main
from wsj_pipeline.service import run_smoke


def write_cli_config(path: Path, source: Path, output: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "[paths]",
                f'source = "{source.as_posix()}"',
                f'output = "{output.as_posix()}"',
                "[pipeline]",
                "batch_size = 2",
            ]
        ),
        encoding="utf-8",
    )


def test_fixture_only_smoke_workflow_succeeds() -> None:
    summary = run_smoke()

    assert summary.validation_ok
    assert summary.article_count == 2
    assert summary.processing.discovered == 3


def test_cli_runs_bounded_fixture_pipeline(
    tmp_path: Path,
    capsys,
) -> None:
    archive = tmp_path / "archive"
    write_article_fixture(archive, "2024/a.html", article_id="A")
    write_article_fixture(archive, "2024/b.html", article_id="B")
    config_path = tmp_path / "pipeline.toml"
    write_cli_config(config_path, archive, tmp_path / "processed")

    exit_code = main(
        [
            "--config",
            str(config_path),
            "run",
            "--limit",
            "2",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["processing"]["discovered"] == 2
    assert output["validation_ok"] is True


def test_cli_smoke_uses_generated_temporary_archive(capsys) -> None:
    exit_code = main(["smoke"])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["validation_ok"] is True
    assert output["processing"]["discovered"] == 3
