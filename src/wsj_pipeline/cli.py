"""Command-line interface and corpus-scale safety gates."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path

from wsj_pipeline.config import PipelineConfig
from wsj_pipeline.index import build_index
from wsj_pipeline.service import (
    inventory_sources,
    process_only,
    publish_all,
    run_incremental,
    run_smoke,
)
from wsj_pipeline.validate import validate_outputs


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
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/wsj_pipeline.toml"),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--limit", type=positive_int)

    for command in ("publish", "index", "validate", "smoke"):
        subparsers.add_parser(command)

    for command in ("process", "run"):
        command_parser = subparsers.add_parser(command)
        scope = command_parser.add_mutually_exclusive_group(required=True)
        scope.add_argument("--limit", type=positive_int)
        scope.add_argument("--full", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one command and emit a deterministic JSON summary."""

    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        result = run_smoke()
    else:
        config = PipelineConfig.from_toml(args.config)
        if args.command == "inventory":
            result = inventory_sources(config, limit=args.limit)
        elif args.command == "process":
            result = process_only(
                config,
                limit=args.limit,
                full=args.full,
            )
        elif args.command == "publish":
            result = publish_all(config)
        elif args.command == "index":
            result = build_index(config, "manual-index")
        elif args.command == "validate":
            result = validate_outputs(config)
        elif args.command == "run":
            result = run_incremental(
                config,
                limit=args.limit,
                full=args.full,
            )
        else:
            raise AssertionError(f"unhandled command {args.command}")

    payload = asdict(result) if is_dataclass(result) else result
    print(json.dumps(payload, sort_keys=True, default=str))
    if args.command == "validate":
        return 0 if result.ok else 1
    if hasattr(result, "validation_ok"):
        return 0 if result.validation_ok else 1
    return 0
