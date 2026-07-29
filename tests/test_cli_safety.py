import pytest

from wsj_pipeline.cli import build_parser


@pytest.mark.parametrize("command", ["process", "run"])
def test_processing_requires_limit_or_full(command: str) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as exc:
        parser.parse_args([command])

    assert exc.value.code == 2


def test_limit_and_full_are_mutually_exclusive() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["run", "--limit", "3", "--full"])

    assert exc.value.code == 2


@pytest.mark.parametrize("value", ["0", "-1", "not-a-number"])
def test_limit_must_be_a_positive_integer(value: str) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["process", "--limit", value])

    assert exc.value.code == 2


@pytest.mark.parametrize(
    "command", ["inventory", "publish", "index", "validate", "smoke"]
)
def test_nonprocessing_commands_do_not_require_scope(command: str) -> None:
    args = build_parser().parse_args([command])

    assert args.command == command
