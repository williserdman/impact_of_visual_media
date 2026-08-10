"""Generated, fixture-safe smoke workflow for article-text embeddings."""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from wsj_embeddings.adapters import FakeEmbeddingAdapter
from wsj_embeddings.config import EmbeddingPipelineConfig
from wsj_embeddings.pipeline import run_embedding_pipeline
from wsj_embeddings.validate import validate_embedding_outputs
from wsj_pipeline.catalog import ArticleRecord, Catalog
from wsj_pipeline.pipeline import article_markdown_path, write_markdown_atomic


@dataclass(frozen=True, slots=True)
class EmbeddingSmokeResult:
    """Content-free outcome from one generated embedding smoke workflow."""

    articles: int
    embeddings: int
    validation_ok: bool


def run_embedding_smoke() -> EmbeddingSmokeResult:
    """Embed and validate one generated canonical Markdown artifact."""

    with tempfile.TemporaryDirectory(prefix="wsj-embeddings-smoke-") as directory:
        root = Path(directory)
        config = EmbeddingPipelineConfig(
            source_root=root / "source",
            preprocessing_output_root=root / "preprocessed",
            embedding_output_root=root / "embeddings",
        )
        config.source_root.mkdir()
        _write_canonical_preprocessing_fixture(config)
        snapshot = _snapshot_inputs(config)

        run_result = run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)
        validation_result = validate_embedding_outputs(config)

        if _snapshot_inputs(config) != snapshot:
            raise AssertionError("embedding smoke modified preprocessing inputs")
        return EmbeddingSmokeResult(
            articles=run_result.articles,
            embeddings=run_result.embeddings,
            validation_ok=validation_result.ok,
        )


def _write_canonical_preprocessing_fixture(config: EmbeddingPipelineConfig) -> None:
    article_id = "wsj:SYNTHETIC-EMBEDDING"
    publication_date = date(2024, 1, 2)
    markdown_path = article_markdown_path(article_id, publication_date)
    markdown = "#\n"
    markdown_sha256 = write_markdown_atomic(
        markdown,
        config.preprocessing_output_root / markdown_path,
        config.preprocessing_output_root / ".staging",
        "smoke",
    )
    with Catalog.open(config.preprocessing_catalog) as catalog, catalog.transaction():
        catalog.replace_article(
            ArticleRecord(
                article_id=article_id,
                source_html_path="synthetic/article.html",
                cleaned_markdown_path=markdown_path.as_posix(),
                header_image_path=None,
                inline_image_urls=(),
                published_at_utc=datetime(2024, 1, 2, 15, 30, tzinfo=UTC),
                publication_date_new_york=publication_date,
                cleaned_markdown_sha256=markdown_sha256,
            )
        )


def _snapshot_inputs(config: EmbeddingPipelineConfig) -> dict[str, str]:
    return {
        f"{root.name}/{path.relative_to(root).as_posix()}": hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for root in (config.source_root, config.preprocessing_output_root)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
