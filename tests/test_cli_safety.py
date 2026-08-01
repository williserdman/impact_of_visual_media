from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from tests.fixtures import write_article_fixture
from wsj_pipeline.cli import build_parser, main


def _command_names(parser: argparse.ArgumentParser) -> set[str]:
    [subcommands] = [
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ]
    return set(subcommands.choices)


def _write_config(path: Path, source: Path, output: Path) -> None:
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


def test_parser_exposes_exactly_the_four_safe_commands() -> None:
    assert _command_names(build_parser()) == {
        "inventory",
        "run",
        "smoke",
        "validate",
    }


@pytest.mark.parametrize("command", ["process", "publish", "index"])
def test_removed_commands_are_rejected(command: str) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([command])

    assert exc.value.code == 2


def test_run_requires_limit_or_full() -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["run"])

    assert exc.value.code == 2


def test_run_limit_and_full_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["run", "--limit", "3", "--full"])

    assert exc.value.code == 2


@pytest.mark.parametrize("value", ["0", "-1", "not-a-number"])
def test_run_limit_must_be_a_positive_integer(value: str) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["run", "--limit", value])

    assert exc.value.code == 2


def test_reprocess_requires_a_run_scope() -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["run", "--reprocess"])

    assert exc.value.code == 2


@pytest.mark.parametrize("command", ["inventory", "validate", "smoke"])
def test_reprocess_is_rejected_outside_run(command: str) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([command, "--reprocess"])

    assert exc.value.code == 2


@pytest.mark.parametrize(
    ("scope", "limit", "full"),
    [(["--limit", "3"], 3, False), (["--full"], None, True)],
)
def test_run_accepts_reprocess_with_an_explicit_scope(
    scope: list[str],
    limit: int | None,
    full: bool,
) -> None:
    args = build_parser().parse_args(["run", *scope, "--reprocess"])

    assert args.limit == limit
    assert args.full is full
    assert args.reprocess is True


def test_real_commands_emit_one_deterministically_ordered_json_object(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive = tmp_path / "archive"
    output = tmp_path / "processed"
    write_article_fixture(archive, "2024/a.html", article_id="A")
    config_path = tmp_path / "pipeline.toml"
    _write_config(config_path, archive, output)

    inv_code = main(["--config", str(config_path), "inventory"])
    inventory_line = capsys.readouterr().out
    run_code = main(
        ["--config", str(config_path), "run", "--limit", "1"]
    )
    run_line = capsys.readouterr().out
    validation_code = main(["--config", str(config_path), "validate"])
    validation_line = capsys.readouterr().out

    assert inv_code == run_code == validation_code == 0
    for line in (inventory_line, run_line, validation_line):
        payload = json.loads(line)
        assert line == f"{json.dumps(payload, sort_keys=True)}\n"
    assert json.loads(inventory_line) == {
        "discovered": 1,
        "missing_image": 0,
        "with_image": 1,
    }
    assert json.loads(run_line)["validation_ok"] is True
    assert json.loads(validation_line) == {"issues": []}


def test_run_reports_an_isolated_extraction_failure_without_failing_command(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive = tmp_path / "archive"
    output = tmp_path / "processed"
    write_article_fixture(archive, "2024/bad.html", published=None)
    config_path = tmp_path / "pipeline.toml"
    _write_config(config_path, archive, output)

    exit_code = main(
        ["--config", str(config_path), "run", "--limit", "1"]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["failed"] == 1


def test_validate_returns_one_for_an_invalid_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "pipeline.toml"
    _write_config(config_path, tmp_path / "archive", tmp_path / "processed")

    exit_code = main(["--config", str(config_path), "validate"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload == {
        "issues": [
            {
                "code": "missing_catalog",
                "message": "catalog path is missing",
            }
        ]
    }
