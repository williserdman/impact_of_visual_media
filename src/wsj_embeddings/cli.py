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
from wsj_embeddings.catalog import EmbeddingCatalogError, configuration_id
from wsj_embeddings.config import EmbeddingConfigError, EmbeddingPipelineConfig
from wsj_embeddings.long_text import TextOffsetTokenizer
from wsj_embeddings.pilot import run_jina_pilot
from wsj_embeddings.pipeline import (
    EmbeddingOutputRootError,
    EmbeddingPipelineError,
    EmbeddingPipelineLockedError,
    HostedProcessingAuthorizationError,
    inventory_embedding_articles,
    run_embedding_pipeline,
)
from wsj_embeddings.smoke import run_embedding_smoke
from wsj_embeddings.validate import validate_embedding_corpus

_OPTION_NAME = re.compile(r"(?<!\w)(--?[A-Za-z][A-Za-z0-9-]*)")
_OPERATIONAL_ERROR_CODES = {
    EmbeddingConfigError: "invalid_configuration",
    EmbeddingCatalogError: "embedding_catalog",
    EmbeddingOutputRootError: "embedding_output_root",
    EmbeddingPipelineLockedError: "embedding_pipeline_locked",
}
_OPERATIONAL_ERRORS = (
    EmbeddingConfigError,
    EmbeddingCatalogError,
    EmbeddingOutputRootError,
    EmbeddingPipelineLockedError,
    EmbeddingPipelineError,
    HostedProcessingAuthorizationError,
    JinaHostedAdapterError,
)


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
    validate = commands.add_parser("validate")
    _add_root_arguments(validate)
    validate.add_argument("--configuration-id")
    run = commands.add_parser("run")
    _add_root_arguments(run)
    scope = run.add_mutually_exclusive_group(required=True)
    scope.add_argument("--limit", type=_positive_int)
    scope.add_argument("--full", action="store_true")
    run.add_argument("--reprocess", action="store_true")
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


def _report_operational_error(error: Exception) -> int:
    code = getattr(error, "code", None)
    if code is None:
        code = next(
            value
            for error_type, value in _OPERATIONAL_ERROR_CODES.items()
            if isinstance(error, error_type)
        )
    print(json.dumps({"error": code}, sort_keys=True))
    return 1


def main(
    argv: Sequence[str] | None = None,
    *,
    environment: Mapping[str, str] | None = None,
    transport: JinaTransport | None = None,
    adapter_factory: Callable[[], EmbeddingAdapter] | None = None,
    tokenizer: TextOffsetTokenizer | None = None,
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
                JinaEmbeddingAdapter(
                    environment=environment,
                    transport=transport,
                    tokenizer=tokenizer,
                )
            )
        except JinaHostedAdapterError as error:
            print(json.dumps({"error": error.code}, sort_keys=True))
            return 1
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "inventory":
        try:
            result = inventory_embedding_articles(_config_from_args(args))
        except _OPERATIONAL_ERRORS as error:
            return _report_operational_error(error)
        print(json.dumps(asdict(result), sort_keys=True))
        return 0
    if args.command == "validate":
        try:
            result = validate_embedding_corpus(
                _config_from_args(args),
                configuration_id=args.configuration_id,
            )
        except _OPERATIONAL_ERRORS as error:
            return _report_operational_error(error)
        payload = asdict(result)
        payload["validation_ok"] = result.ok
        print(json.dumps(payload, sort_keys=True))
        return 0 if result.ok else 1
    if args.command == "run":
        try:
            config = _config_from_args(
                args,
                hosted_processing_authorized=args.authorize_hosted_processing,
            )
            if not config.hosted_processing_authorized:
                raise HostedProcessingAuthorizationError
            adapter = (
                JinaEmbeddingAdapter(environment=environment, transport=transport)
                if adapter_factory is None
                else adapter_factory()
            )
            result = run_embedding_pipeline(
                config,
                adapter,
                limit=args.limit,
                full=args.full,
                reprocess=args.reprocess,
            )
        except _OPERATIONAL_ERRORS as error:
            return _report_operational_error(error)
        payload = asdict(result)
        payload["configuration_id"] = configuration_id(adapter.profile)
        print(json.dumps(payload, sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command {args.command}")
