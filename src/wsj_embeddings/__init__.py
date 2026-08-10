"""Deterministic downstream embeddings for generated preprocessing outputs."""

from wsj_embeddings.adapters import EmbeddingAdapter, FakeEmbeddingAdapter
from wsj_embeddings.config import EmbeddingPipelineConfig
from wsj_embeddings.models import EmbeddingProfile, EmbeddingRunResult
from wsj_embeddings.pipeline import run_embedding_pipeline

__all__ = [
    "EmbeddingAdapter",
    "EmbeddingPipelineConfig",
    "EmbeddingProfile",
    "EmbeddingRunResult",
    "FakeEmbeddingAdapter",
    "run_embedding_pipeline",
]
