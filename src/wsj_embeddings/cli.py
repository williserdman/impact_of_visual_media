"""Command-line interface for fixture-safe article-text embedding checks."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict

from wsj_embeddings.smoke import run_embedding_smoke


def build_parser() -> argparse.ArgumentParser:
    """Build the embedding CLI parser without performing any I/O."""

    parser = argparse.ArgumentParser(prog="wsj-embeddings")
    parser.add_subparsers(dest="command", required=True).add_parser("smoke")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one fixture-safe embedding command."""

    args = build_parser().parse_args(argv)
    if args.command != "smoke":
        raise AssertionError(f"unhandled command {args.command}")
    result = run_embedding_smoke()
    print(json.dumps(asdict(result), sort_keys=True))
    return 0 if result.validation_ok else 1
