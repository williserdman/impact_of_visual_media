"""Command-line interface for safe embedding checks, inventory, and operation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

from wsj_embeddings.adapters import (
    EmbeddingAdapter,
    JinaEmbeddingAdapter,
    JinaHostedAdapterError,
    JinaTransport,
)
from wsj_embeddings.config import EmbeddingPipelineConfig
from wsj_embeddings.pilot import run_jina_pilot
from wsj_embeddings.pipeline import (
    HostedProcessingAuthorizationError,
    inventory_embedding_articles,
    run_embedding_pipeline,
)
from wsj_embeddings.smoke import run_embedding_smoke

_OPTION_NAME = re.compile(r"(?<!\w)(--?[A-Za-z][A-Za-z0-9-]*)")


class _ContentFreeArgumentParser(argparse.ArgumentParser):
    """Reject invalid CLI input without echoing supplied text or paths."""

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        option = _OPTION_NAME.search(message)
        if option is None:
            self.exit(2, f"{self.prog}: error: invalid arguments\n")
        self.exit(2, f"{self.prog}: error: invalid option {option.group(1)}\n")


def build_parser() -> argparse.ArgumentParser:
    """Build the embedding CLI parser without performing any I/O."""

    parser = _ContentFreeArgumentParser(prog="wsj-embeddings")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("smoke")
    commands.add_parser("pilot")
    inventory = commands.add_parser("inventory")
    _add_root_arguments(inventory)
    run = commands.add_parser("run")
    _add_root_arguments(run)
    run.add_argument("--limit", required=True, type=_positive_int)
    run.add_argument("--authorize-hosted-processing", action="store_true")
    return parser


def _add_root_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--preprocessing-output-root", required=True, type=Path)
    parser.add_argument("--embedding-output-root", required=True, type=Path)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _config_from_args(
    args: argparse.Namespace,
    *,
    hosted_processing_authorized: bool = False,
) -> EmbeddingPipelineConfig:
    return EmbeddingPipelineConfig(
        source_root=args.source_root,
        preprocessing_output_root=args.preprocessing_output_root,
        embedding_output_root=args.embedding_output_root,
        hosted_processing_authorized=hosted_processing_authorized,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    environment: Mapping[str, str] | None = None,
    transport: JinaTransport | None = None,
    adapter_factory: Callable[[], EmbeddingAdapter] | None = None,
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
    if args.command == "inventory":
        result = inventory_embedding_articles(_config_from_args(args))
        print(json.dumps(asdict(result), sort_keys=True))
        return 0
    if args.command == "run":
        config = _config_from_args(
            args,
            hosted_processing_authorized=args.authorize_hosted_processing,
        )
        if not config.hosted_processing_authorized:
            error = HostedProcessingAuthorizationError()
            print(json.dumps({"error": error.code}, sort_keys=True))
            return 1
        try:
            adapter = (
                JinaEmbeddingAdapter(environment=environment, transport=transport)
                if adapter_factory is None
                else adapter_factory()
            )
        except JinaHostedAdapterError as error:
            print(json.dumps({"error": error.code}, sort_keys=True))
            return 1
        result = run_embedding_pipeline(config, adapter, limit=args.limit)
        print(json.dumps(asdict(result), sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command {args.command}")
