"""Immutable records at the embedding coordinator boundary."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EmbeddingProfile:
    """Identity and output contract for one embedding configuration."""

    model: str
    task: str
    dimensions: int
    output_type: str = "float"
    normalization: str = "l2-client-v1"


@dataclass(frozen=True, slots=True)
class EmbeddingRunResult:
    """Content-free counts published by one embedding run."""

    articles: int
    embeddings: int
