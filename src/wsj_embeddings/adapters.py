"""Injected embedding encoders."""

from __future__ import annotations

from typing import Protocol

from wsj_embeddings.models import EmbeddingProfile


class EmbeddingAdapter(Protocol):
    """Encodes one complete canonical article text without network policy."""

    @property
    def profile(self) -> EmbeddingProfile: ...

    def embed_text(self, text: str) -> tuple[float, ...]: ...


class FakeEmbeddingAdapter:
    """Deterministic generated-fixture encoder for the first smoke slice."""

    profile = EmbeddingProfile(
        model="fake-jina-embeddings-v4",
        task="retrieval.passage",
        dimensions=2048,
    )

    def embed_text(self, text: str) -> tuple[float, ...]:
        if not text:
            raise ValueError("text input must not be empty")
        return (1.0, *(0.0 for _ in range(2047)))
