"""Pinned local tokenizer acquisition for Jina Embeddings v4 text planning."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

JINA_V4_TOKENIZER_REPOSITORY = "jinaai/jina-embeddings-v4"
JINA_V4_TOKENIZER_FILENAME = "tokenizer.json"
JINA_V4_TOKENIZER_REVISION = "d1e5d70b7b34d927a8cddac458583c4fbe50a914"
JINA_V4_TOKENIZER_SHA256 = (
    "9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa"
)


class _Encoding(Protocol):
    offsets: Sequence[tuple[int, int]]


class _LoadedTokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> _Encoding: ...


class PinnedTokenizerError(RuntimeError):
    """Content-free failure to resolve or verify the required tokenizer."""


TokenizerResolver = Callable[[str, str, str], Path]
TokenizerLoader = Callable[[str], _LoadedTokenizer]


class PinnedJinaV4Tokenizer:
    """Load only checksum-verified tokenizer bytes at one immutable revision."""

    def __init__(
        self,
        *,
        resolver: TokenizerResolver | None = None,
        loader: TokenizerLoader | None = None,
        expected_sha256: str = JINA_V4_TOKENIZER_SHA256,
    ) -> None:
        self._resolver = _resolve_tokenizer if resolver is None else resolver
        self._loader = _load_tokenizer if loader is None else loader
        self._expected_sha256 = expected_sha256
        self._tokenizer: _LoadedTokenizer | None = None

    def token_offsets(self, text: str) -> tuple[tuple[int, int], ...]:
        """Return source offsets with special-token insertion disabled."""

        tokenizer = self._tokenizer
        if tokenizer is None:
            tokenizer = self._load_verified()
            self._tokenizer = tokenizer
        try:
            encoding = tokenizer.encode(text, add_special_tokens=False)
            return tuple((int(start), int(end)) for start, end in encoding.offsets)
        except (AttributeError, TypeError, ValueError):
            raise PinnedTokenizerError(
                "pinned tokenizer returned invalid offsets"
            ) from None

    def _load_verified(self) -> _LoadedTokenizer:
        try:
            artifact_path = Path(
                self._resolver(
                    JINA_V4_TOKENIZER_REPOSITORY,
                    JINA_V4_TOKENIZER_FILENAME,
                    JINA_V4_TOKENIZER_REVISION,
                )
            )
            artifact_bytes = artifact_path.read_bytes()
        except (OSError, RuntimeError, TypeError, ValueError):
            raise PinnedTokenizerError(
                "pinned tokenizer artifact could not be resolved"
            ) from None
        if hashlib.sha256(artifact_bytes).hexdigest() != self._expected_sha256:
            raise PinnedTokenizerError(
                "pinned tokenizer artifact checksum does not match"
            )
        try:
            serialized = artifact_bytes.decode("utf-8", errors="strict")
            return self._loader(serialized)
        except (OSError, RuntimeError, TypeError, UnicodeError, ValueError):
            raise PinnedTokenizerError(
                "pinned tokenizer artifact could not be loaded"
            ) from None


def _resolve_tokenizer(repo_id: str, filename: str, revision: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
        )
    )


def _load_tokenizer(serialized: str) -> _LoadedTokenizer:
    from tokenizers import Tokenizer

    return Tokenizer.from_str(serialized)
