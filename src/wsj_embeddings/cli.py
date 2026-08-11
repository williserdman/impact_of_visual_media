"""Command-line interface for fixture-safe checks and generated hosted probes."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict

from wsj_embeddings.adapters import (
    JinaEmbeddingAdapter,
    JinaHostedAdapterError,
    JinaTransport,
)
from wsj_embeddings.pilot import run_jina_pilot
from wsj_embeddings.smoke import run_embedding_smoke


def build_parser() -> argparse.ArgumentParser:
    """Build the embedding CLI parser without performing any I/O."""

    parser = argparse.ArgumentParser(prog="wsj-embeddings")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("smoke")
    commands.add_parser("pilot")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    environment: Mapping[str, str] | None = None,
    transport: JinaTransport | None = None,
) -> int:
    """Run one fixture-safe command or the explicit hosted generated pilot."""

    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        result = run_embedding_smoke()
        print(json.dumps(asdict(result), sort_keys=True))
        return 0 if result.validation_ok else 1
    if args.command == "pilot":
        try:
            result = run_jina_pilot(
                JinaEmbeddingAdapter(environment=environment, transport=transport)
            )
        except JinaHostedAdapterError as error:
            print(json.dumps({"error": error.code}, sort_keys=True))
            return 1
        print(json.dumps(result, sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command {args.command}")
