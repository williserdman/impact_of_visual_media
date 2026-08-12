"""Deterministic downstream text/image embeddings for preprocessing outputs."""

from wsj_embeddings.adapters import EmbeddingAdapter, FakeEmbeddingAdapter
from wsj_embeddings.catalog import EmbeddingCatalogError
from wsj_embeddings.config import EmbeddingPipelineConfig
from wsj_embeddings.models import (
    EmbeddingInventoryResult,
    EmbeddingProfile,
    EmbeddingRunResult,
)
from wsj_embeddings.pipeline import (
    EmbeddingPipelineFailpoints,
    HostedProcessingAuthorizationError,
    inventory_embedding_articles,
    run_embedding_pipeline,
)
from wsj_embeddings.smoke import EmbeddingSmokeResult, run_embedding_smoke
from wsj_embeddings.validate import (
    EmbeddingCorpusValidationResult,
    EmbeddingCoverage,
    EmbeddingValidationIssue,
    EmbeddingValidationResult,
    validate_embedding_corpus,
    validate_embedding_outputs,
)

__all__ = [
    "EmbeddingAdapter",
    "EmbeddingCatalogError",
    "EmbeddingCorpusValidationResult",
    "EmbeddingCoverage",
    "EmbeddingInventoryResult",
    "EmbeddingPipelineConfig",
    "EmbeddingPipelineFailpoints",
    "EmbeddingProfile",
    "EmbeddingRunResult",
    "EmbeddingSmokeResult",
    "EmbeddingValidationIssue",
    "EmbeddingValidationResult",
    "FakeEmbeddingAdapter",
    "HostedProcessingAuthorizationError",
    "inventory_embedding_articles",
    "run_embedding_pipeline",
    "run_embedding_smoke",
    "validate_embedding_corpus",
    "validate_embedding_outputs",
]
