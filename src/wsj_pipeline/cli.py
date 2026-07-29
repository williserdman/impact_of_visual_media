"""Command-line interface and corpus-scale safety gates."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def positive_int(value: str) -> int:
    """Parse a strictly positive command-line integer."""

    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser without performing any I/O."""

    parser = argparse.ArgumentParser(prog="wsj-pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("inventory", "publish", "index", "validate", "smoke"):
        subparsers.add_parser(command)

    for command in ("process", "run"):
        command_parser = subparsers.add_parser(command)
        scope = command_parser.add_mutually_exclusive_group(required=True)
        scope.add_argument("--limit", type=positive_int)
        scope.add_argument("--full", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse a command, preserving safety gates before handlers are wired."""

    args = build_parser().parse_args(argv)
    print(f"{args.command}: command handler not yet available")
    return 1
