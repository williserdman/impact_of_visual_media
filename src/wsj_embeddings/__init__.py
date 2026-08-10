"""Deterministic downstream embeddings for generated preprocessing outputs."""

from wsj_embeddings.adapters import EmbeddingAdapter, FakeEmbeddingAdapter
from wsj_embeddings.catalog import EmbeddingCatalogError
from wsj_embeddings.config import EmbeddingPipelineConfig
from wsj_embeddings.models import EmbeddingProfile, EmbeddingRunResult
from wsj_embeddings.pipeline import run_embedding_pipeline
from wsj_embeddings.validate import (
    EmbeddingValidationIssue,
    EmbeddingValidationResult,
    validate_embedding_outputs,
)

__all__ = [
    "EmbeddingAdapter",
    "EmbeddingCatalogError",
    "EmbeddingPipelineConfig",
    "EmbeddingProfile",
    "EmbeddingRunResult",
    "EmbeddingValidationIssue",
    "EmbeddingValidationResult",
    "FakeEmbeddingAdapter",
    "run_embedding_pipeline",
    "validate_embedding_outputs",
]
