"""Four-command interface for safe Markdown catalog operations."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

from wsj_pipeline.config import PipelineConfig
from wsj_pipeline.pipeline import (
    inventory_sources,
    run_smoke,
    run_validated_pipeline,
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
    subparsers.add_parser("inventory")
    run_parser = subparsers.add_parser("run")
    scope = run_parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--limit", type=positive_int)
    scope.add_argument("--full", action="store_true")
    run_parser.add_argument("--reprocess", action="store_true")
    run_parser.add_argument("--workers", type=positive_int)
    subparsers.add_parser("validate")
    subparsers.add_parser("smoke")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one safe command and emit one deterministic JSON object."""

    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        result = run_smoke()
    else:
        config = PipelineConfig.from_toml(args.config)
        if args.command == "inventory":
            result = inventory_sources(config)
        elif args.command == "run":
            if args.workers is not None:
                config = replace(config, workers=args.workers)
            result = run_validated_pipeline(
                config,
                limit=args.limit,
                full=args.full,
                reprocess=args.reprocess,
            )
        elif args.command == "validate":
            result = validate_outputs(config)
        else:
            raise AssertionError(f"unhandled command {args.command}")

    print(json.dumps(asdict(result), sort_keys=True))
    if args.command == "validate":
        return 0 if result.ok else 1
    if args.command in {"run", "smoke"}:
        return 0 if result.validation_ok else 1
    return 0
