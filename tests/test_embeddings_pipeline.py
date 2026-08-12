from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import struct
import sys
import types
import warnings
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import pytest

import wsj_embeddings.canonical_markdown as canonical_markdown_module
import wsj_embeddings.catalog as embedding_catalog_module
import wsj_embeddings.image_rendition as image_rendition_module
import wsj_embeddings.pipeline as embedding_pipeline_module
import wsj_embeddings.source_image as source_image_module
from wsj_embeddings import (
    EmbeddingCatalogError,
    EmbeddingPipelineConfig,
    EmbeddingPipelineFailpoints,
    EmbeddingProfile,
    EmbeddingRunResult,
    EmbeddingValidationIssue,
    EmbeddingValidationResult,
    FakeEmbeddingAdapter,
    run_embedding_pipeline,
    validate_embedding_outputs,
)
from wsj_embeddings.adapters import JinaEmbeddingAdapter, JinaHostedAdapterError
from wsj_embeddings.catalog import EmbeddingCatalog, configuration_id
from wsj_embeddings.config import EmbeddingConfigError
from wsj_embeddings.image_rendition import (
    ImageCodecError,
    ImageInfo,
    PillowImageCodec,
)
from wsj_embeddings.pipeline import (
    EmbeddingPipelineError,
    EmbeddingPipelineLockedError,
    HostedProcessingAuthorizationError,
    embedding_pipeline_lock,
)
from wsj_pipeline.catalog import ArticleRecord, Catalog

EXPECTED_EMBEDDING_TABLE_COLUMNS = {
    "metadata": (
        ("key", "VARCHAR", True, None, True),
        ("value", "VARCHAR", True, None, False),
    ),
    "embedding_configurations": (
        ("configuration_id", "VARCHAR", True, None, True),
        ("model", "VARCHAR", True, None, False),
        ("observed_model", "VARCHAR", True, None, False),
        ("observed_api_version", "VARCHAR", True, None, False),
        ("task", "VARCHAR", True, None, False),
        ("dimensions", "INTEGER", True, None, False),
        ("output_type", "VARCHAR", True, None, False),
        ("normalization", "VARCHAR", True, None, False),
        ("tokenizer_revision", "VARCHAR", True, None, False),
        ("tokenizer_engine", "VARCHAR", True, None, False),
        ("context_token_limit", "INTEGER", True, None, False),
        ("context_rules", "VARCHAR", True, None, False),
        ("long_text_aggregation", "VARCHAR", True, None, False),
        ("long_text_part_attempt_limit", "INTEGER", True, None, False),
        ("image_input_rules", "VARCHAR", True, None, False),
        ("image_transform", "VARCHAR", True, None, False),
        ("multimodal_formula", "VARCHAR", True, None, False),
        ("client_configuration_version", "VARCHAR", True, None, False),
    ),
    "runs": (
        ("run_id", "VARCHAR", True, None, True),
        ("configuration_id", "VARCHAR", True, None, False),
        ("articles", "INTEGER", True, None, False),
        ("embeddings", "INTEGER", True, None, False),
        ("reused", "INTEGER", True, None, False),
        ("attempted", "INTEGER", True, None, False),
        ("succeeded", "INTEGER", True, None, False),
        ("retryable", "INTEGER", True, None, False),
        ("terminal", "INTEGER", True, None, False),
        ("interrupted", "INTEGER", True, None, False),
        ("header_absent", "INTEGER", True, None, False),
        ("header_failed", "INTEGER", True, None, False),
        ("started_at", "TIMESTAMP WITH TIME ZONE", True, None, False),
    ),
    "embedding_work_items": (
        ("article_id", "VARCHAR", True, None, True),
        ("modality", "VARCHAR", True, None, True),
        ("configuration_id", "VARCHAR", True, None, True),
        ("source_relative_path", "VARCHAR", False, None, False),
        ("input_sha256", "VARCHAR", False, None, False),
        ("state", "VARCHAR", True, None, False),
        ("attempt_count", "INTEGER", True, None, False),
        ("error_code", "VARCHAR", False, None, False),
        ("status_code", "INTEGER", False, None, False),
        ("retry_after_seconds", "DOUBLE", False, None, False),
        ("last_run_id", "VARCHAR", True, None, False),
        ("generation_run_id", "VARCHAR", False, None, False),
        ("updated_at", "TIMESTAMP WITH TIME ZONE", True, None, False),
    ),
    "embeddings": (
        ("article_id", "VARCHAR", True, None, True),
        ("modality", "VARCHAR", True, None, True),
        ("configuration_id", "VARCHAR", True, None, True),
        ("source_relative_path", "VARCHAR", False, None, False),
        ("published_at_utc", "TIMESTAMP WITH TIME ZONE", True, None, False),
        ("publication_date_new_york", "DATE", True, None, False),
        ("dimensions", "INTEGER", True, None, False),
        ("input_sha256", "VARCHAR", True, None, False),
        ("stored_vector_sha256", "VARCHAR", True, None, False),
        ("vector", "FLOAT[2048]", True, None, False),
    ),
    "embedding_generation_history": (
        ("article_id", "VARCHAR", True, None, True),
        ("modality", "VARCHAR", True, None, True),
        ("configuration_id", "VARCHAR", True, None, True),
        ("generation_run_id", "VARCHAR", True, None, True),
        ("source_relative_path", "VARCHAR", False, None, False),
        ("input_sha256", "VARCHAR", True, None, False),
        ("stored_vector_sha256", "VARCHAR", True, None, False),
        ("superseded_run_id", "VARCHAR", True, None, False),
        ("superseded_reason", "VARCHAR", True, None, False),
        ("superseded_at", "TIMESTAMP WITH TIME ZONE", True, None, False),
    ),
    "multimodal_embedding_provenance": (
        ("article_id", "VARCHAR", True, None, True),
        ("configuration_id", "VARCHAR", True, None, True),
        ("generation_run_id", "VARCHAR", True, None, True),
        ("text_generation_run_id", "VARCHAR", True, None, False),
        ("text_stored_vector_sha256", "VARCHAR", True, None, False),
        ("header_image_generation_run_id", "VARCHAR", True, None, False),
        ("header_image_stored_vector_sha256", "VARCHAR", True, None, False),
        ("formula_version", "VARCHAR", True, None, False),
        ("created_at", "TIMESTAMP WITH TIME ZONE", True, None, False),
    ),
    "image_input_provenance": (
        ("article_id", "VARCHAR", True, None, True),
        ("configuration_id", "VARCHAR", True, None, True),
        ("generation_run_id", "VARCHAR", True, None, True),
        ("source_sha256", "VARCHAR", True, None, False),
        ("source_format", "VARCHAR", True, None, False),
        ("source_bytes", "BIGINT", True, None, False),
        ("source_width", "INTEGER", True, None, False),
        ("source_height", "INTEGER", True, None, False),
        ("source_frames", "INTEGER", True, None, False),
        ("embedded_input_sha256", "VARCHAR", True, None, False),
        ("embedded_format", "VARCHAR", True, None, False),
        ("embedded_bytes", "BIGINT", True, None, False),
        ("embedded_width", "INTEGER", True, None, False),
        ("embedded_height", "INTEGER", True, None, False),
        ("embedded_frames", "INTEGER", True, None, False),
        ("transform_id", "VARCHAR", True, None, False),
        ("rendition_relative_path", "VARCHAR", False, None, False),
        ("created_at", "TIMESTAMP WITH TIME ZONE", True, None, False),
    ),
    "long_text_parts": (
        ("article_id", "VARCHAR", True, None, True),
        ("configuration_id", "VARCHAR", True, None, True),
        ("article_input_sha256", "VARCHAR", True, None, True),
        ("part_index", "INTEGER", True, None, True),
        ("part_count", "INTEGER", True, None, False),
        ("char_start", "BIGINT", True, None, False),
        ("char_end", "BIGINT", True, None, False),
        ("byte_start", "BIGINT", True, None, False),
        ("byte_end", "BIGINT", True, None, False),
        ("part_input_sha256", "VARCHAR", True, None, False),
        ("token_count", "INTEGER", True, None, False),
        ("state", "VARCHAR", True, None, False),
        ("attempt_count", "INTEGER", True, None, False),
        ("error_code", "VARCHAR", False, None, False),
        ("status_code", "INTEGER", False, None, False),
        ("retry_after_seconds", "DOUBLE", False, None, False),
        ("last_run_id", "VARCHAR", True, None, False),
        ("generation_run_id", "VARCHAR", False, None, False),
        ("stored_vector_sha256", "VARCHAR", False, None, False),
        ("vector", "FLOAT[2048]", False, None, False),
        ("updated_at", "TIMESTAMP WITH TIME ZONE", True, None, False),
    ),
    "long_text_part_generations": (
        ("article_id", "VARCHAR", True, None, True),
        ("configuration_id", "VARCHAR", True, None, True),
        ("article_input_sha256", "VARCHAR", True, None, True),
        ("part_index", "INTEGER", True, None, True),
        ("generation_run_id", "VARCHAR", True, None, True),
        ("part_count", "INTEGER", True, None, False),
        ("char_start", "BIGINT", True, None, False),
        ("char_end", "BIGINT", True, None, False),
        ("byte_start", "BIGINT", True, None, False),
        ("byte_end", "BIGINT", True, None, False),
        ("part_input_sha256", "VARCHAR", True, None, False),
        ("token_count", "INTEGER", True, None, False),
        ("stored_vector_sha256", "VARCHAR", True, None, False),
        ("vector", "FLOAT[2048]", True, None, False),
        ("created_at", "TIMESTAMP WITH TIME ZONE", True, None, False),
    ),
    "article_text_aggregation_provenance": (
        ("article_id", "VARCHAR", True, None, True),
        ("configuration_id", "VARCHAR", True, None, True),
        ("generation_run_id", "VARCHAR", True, None, True),
        ("part_index", "INTEGER", True, None, True),
        ("article_input_sha256", "VARCHAR", True, None, False),
        ("part_count", "INTEGER", True, None, False),
        ("part_generation_run_id", "VARCHAR", True, None, False),
        ("part_input_sha256", "VARCHAR", True, None, False),
        ("token_count", "INTEGER", True, None, False),
        ("part_stored_vector_sha256", "VARCHAR", True, None, False),
        ("aggregation_version", "VARCHAR", True, None, False),
        ("created_at", "TIMESTAMP WITH TIME ZONE", True, None, False),
    ),
}
EXPECTED_EMBEDDING_PRIMARY_KEYS = {
    ("metadata", ("key",)),
    ("embedding_configurations", ("configuration_id",)),
    ("runs", ("run_id",)),
    (
        "embedding_work_items",
        ("article_id", "modality", "configuration_id"),
    ),
    ("embeddings", ("article_id", "modality", "configuration_id")),
    (
        "embedding_generation_history",
        ("article_id", "modality", "configuration_id", "generation_run_id"),
    ),
    (
        "multimodal_embedding_provenance",
        ("article_id", "configuration_id", "generation_run_id"),
    ),
    (
        "image_input_provenance",
        ("article_id", "configuration_id", "generation_run_id"),
    ),
    (
        "long_text_parts",
        (
            "article_id",
            "configuration_id",
            "article_input_sha256",
            "part_index",
        ),
    ),
    (
        "long_text_part_generations",
        (
            "article_id",
            "configuration_id",
            "article_input_sha256",
            "part_index",
            "generation_run_id",
        ),
    ),
    (
        "article_text_aggregation_provenance",
        ("article_id", "configuration_id", "generation_run_id", "part_index"),
    ),
}


def write_generated_preprocessing_fixture(tmp_path: Path) -> EmbeddingPipelineConfig:
    source_root = tmp_path / "source"
    preprocessing_output_root = tmp_path / "preprocessed"
    embedding_output_root = tmp_path / "embeddings"
    source_root.mkdir()
    markdown_path = (
        preprocessing_output_root / "text" / "2024" / "01" / "02" / "article.md"
    )
    markdown_path.parent.mkdir(parents=True)
    markdown = "#\n"
    markdown_path.write_text(markdown, encoding="utf-8")
    with Catalog.open(
        preprocessing_output_root / "catalog.duckdb"
    ) as catalog, catalog.transaction():
        catalog.replace_article(
            ArticleRecord(
                article_id="wsj:SYNTHETIC-EMBEDDING",
                source_html_path="synthetic/article.html",
                cleaned_markdown_path=markdown_path.relative_to(
                    preprocessing_output_root
                ).as_posix(),
                header_image_path=None,
                inline_image_urls=(),
                published_at_utc=datetime(2024, 1, 2, 15, 30, tzinfo=UTC),
                publication_date_new_york=date(2024, 1, 2),
                cleaned_markdown_sha256=hashlib.sha256(markdown.encode()).hexdigest(),
            )
        )
    return EmbeddingPipelineConfig(
        source_root=source_root,
        preprocessing_output_root=preprocessing_output_root,
        embedding_output_root=embedding_output_root,
        hosted_processing_authorized=True,
    )


def snapshot_fixture_inputs(config: EmbeddingPipelineConfig) -> dict[str, str]:
    roots = (config.source_root, config.preprocessing_output_root)
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for root in roots
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def add_generated_embedding_article(
    config: EmbeddingPipelineConfig,
    *,
    article_id: str,
    markdown: str,
) -> None:
    markdown_path = config.preprocessing_output_root / "text" / f"{article_id[-1]}.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    with Catalog.open(config.preprocessing_catalog) as catalog, catalog.transaction():
        catalog.replace_article(
            ArticleRecord(
                article_id=article_id,
                source_html_path=f"synthetic/{article_id[-1]}.html",
                cleaned_markdown_path=markdown_path.relative_to(
                    config.preprocessing_output_root
                ).as_posix(),
                header_image_path=None,
                inline_image_urls=(),
                published_at_utc=datetime(2024, 1, 3, 15, 30, tzinfo=UTC),
                publication_date_new_york=date(2024, 1, 3),
                cleaned_markdown_sha256=hashlib.sha256(markdown.encode()).hexdigest(),
            )
        )


def replace_generated_embedding_markdown(
    config: EmbeddingPipelineConfig,
    markdown: str,
) -> str:
    markdown_path = (
        config.preprocessing_output_root
        / "text"
        / "2024"
        / "01"
        / "02"
        / "article.md"
    )
    markdown_path.write_text(markdown, encoding="utf-8")
    input_sha256 = hashlib.sha256(markdown.encode()).hexdigest()
    with (
        Catalog.open(config.preprocessing_catalog) as catalog,
        catalog.transaction(),
    ):
        catalog.connection.execute(
            """
            UPDATE articles
            SET cleaned_markdown_sha256 = ?
            WHERE article_id = 'wsj:SYNTHETIC-EMBEDDING'
            """,
            [input_sha256],
        )
    return input_sha256


def attach_generated_header_image(
    config: EmbeddingPipelineConfig,
    image_bytes: bytes,
    *,
    relative_path: str = "synthetic/article_main_image.png",
) -> str:
    """Attach one generated local image and one forbidden remote reference."""

    image_path = config.source_root / relative_path
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(image_bytes)
    with Catalog.open(config.preprocessing_catalog) as catalog, catalog.transaction():
        catalog.connection.execute(
            """
            UPDATE articles
            SET header_image_path = ?, inline_image_urls = ?
            WHERE article_id = 'wsj:SYNTHETIC-EMBEDDING'
            """,
            [relative_path, ["https://example.invalid/never-fetch.png"]],
        )
    return relative_path


def declare_missing_generated_header_image(
    config: EmbeddingPipelineConfig,
    *,
    relative_path: str = "synthetic/missing-header.png",
) -> str:
    """Declare one generated fixture path without creating the optional file."""

    with Catalog.open(config.preprocessing_catalog) as catalog, catalog.transaction():
        catalog.connection.execute(
            """
            UPDATE articles
            SET header_image_path = ?
            WHERE article_id = 'wsj:SYNTHETIC-EMBEDDING'
            """,
            [relative_path],
        )
    return relative_path


class SyntheticInterruption(BaseException):
    """Simulate abrupt process loss without entering operational error handling."""


class InterruptingAdapter(FakeEmbeddingAdapter):
    def __init__(self, *, after_response: bool) -> None:
        self.after_response = after_response
        self.calls = 0
        self.responded = False

    def embed_text(self, text: str) -> tuple[float, ...]:
        self.calls += 1
        if self.after_response:
            super().embed_text(text)
            self.responded = True
        raise SyntheticInterruption


class RecordingAdapter(FakeEmbeddingAdapter):
    def __init__(self) -> None:
        self.calls = 0
        self.responded = False

    def embed_text(self, text: str) -> tuple[float, ...]:
        self.calls += 1
        vector = super().embed_text(text)
        self.responded = True
        return vector


class CharacterTokenLongTextAdapter(FakeEmbeddingAdapter):
    """Offline tokenizer/encoder whose one-codepoint offsets are hand-checkable."""

    profile = replace(
        FakeEmbeddingAdapter.profile,
        tokenizer_revision="synthetic-character-offset-tokenizer-v1",
        context_token_limit=8,
        context_rules="markdown-block-greedy-no-overlap-v1",
        long_text_aggregation="l2-token-count-weighted-mean-float32-v1",
        client_configuration_version="wsj-embeddings-config-v2",
    )

    def __init__(self) -> None:
        self.embedded_texts: list[str] = []

    def token_offsets(self, text: str) -> tuple[tuple[int, int], ...]:
        return tuple((index, index + 1) for index in range(len(text)))

    def embed_text(self, text: str) -> tuple[float, ...]:
        axis = len(self.embedded_texts)
        self.embedded_texts.append(text)
        return tuple(1.0 if index == axis else 0.0 for index in range(2048))


class RecordingMultimodalAdapter(RecordingAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.image_calls = 0
        self.encoded_images: list[str] = []

    def embed_image(self, image_base64: str) -> tuple[float, ...]:
        self.image_calls += 1
        self.encoded_images.append(image_base64)
        return super().embed_image(image_base64)


class ScenarioImageCodec:
    """Offline image-policy seam with literal, hand-checkable fixture outcomes."""

    input_rules = "synthetic-static-max5000000b-max20000000px-v1"
    transform_id = "synthetic-jpeg-rendition-v1"

    def __init__(self, scenarios: dict[bytes, ImageInfo | ImageCodecError]) -> None:
        self.scenarios = scenarios
        self.rendered: list[tuple[bytes, int, int]] = []

    def inspect(self, data: bytes, *, max_decode_pixels: int) -> ImageInfo:
        del max_decode_pixels
        outcome = self.scenarios[data]
        if isinstance(outcome, ImageCodecError):
            raise outcome
        return outcome

    def render_jpeg(
        self,
        data: bytes,
        *,
        width: int,
        height: int,
    ) -> bytes:
        self.rendered.append((data, width, height))
        return b"generated deterministic jpeg rendition"


class ScenarioImageAdapter(RecordingMultimodalAdapter):
    profile = replace(
        FakeEmbeddingAdapter.profile,
        image_input_rules=ScenarioImageCodec.input_rules,
        image_transform=ScenarioImageCodec.transform_id,
    )


def install_synthetic_pillow(
    monkeypatch,
    *,
    jpeg_version: str,
    turbo_version: str,
    open_image,
) -> None:
    """Install a complete synthetic Pillow import boundary without a wheel."""

    class DecompressionBombError(Exception):
        pass

    class DecompressionBombWarning(RuntimeWarning):
        pass

    class UnidentifiedImageError(Exception):
        pass

    image = types.SimpleNamespace(
        DecompressionBombError=DecompressionBombError,
        DecompressionBombWarning=DecompressionBombWarning,
        open=open_image,
    )
    features = types.SimpleNamespace(
        check=lambda name: name == "webp",
        check_codec=lambda name: name in {"jpg", "zlib"},
        check_feature=lambda name: name == "libjpeg_turbo",
        version_codec=lambda name: {
            "jpg": jpeg_version,
            "zlib": "1.3.1",
        }[name],
        version_feature=lambda name: (
            turbo_version if name == "libjpeg_turbo" else None
        ),
        version_module=lambda name: "1.5.0" if name == "webp" else None,
    )
    pillow = types.ModuleType("PIL")
    pillow.__version__ = "11.3.0"
    pillow.Image = image
    pillow.ImageOps = types.SimpleNamespace(exif_transpose=lambda value: value)
    pillow.UnidentifiedImageError = UnidentifiedImageError
    pillow.features = features
    monkeypatch.setitem(sys.modules, "PIL", pillow)


class InterruptingImageAdapter(RecordingMultimodalAdapter):
    def embed_image(self, image_base64: str) -> tuple[float, ...]:
        self.image_calls += 1
        self.encoded_images.append(image_base64)
        raise SyntheticInterruption


class PriorConfigurationAdapter(FakeEmbeddingAdapter):
    profile = EmbeddingProfile(
        model="synthetic-prior-model",
        task="retrieval.passage",
        dimensions=2048,
    )


def test_configuration_identity_covers_every_meaning_bearing_profile_field():
    """Break caught: a public embedding rule changes without new eligibility."""

    profile = FakeEmbeddingAdapter.profile
    assert (
        profile.long_text_aggregation
        == "l2-token-count-weighted-mean-float32-v1"
    )
    assert (
        JinaEmbeddingAdapter.profile.long_text_aggregation
        == "l2-token-count-weighted-mean-float32-v1"
    )
    assert JinaEmbeddingAdapter.profile.context_token_limit == 8_000
    assert 8_192 - JinaEmbeddingAdapter.profile.context_token_limit == 192
    assert JinaEmbeddingAdapter.profile.tokenizer_engine == "tokenizers-0.21.4"
    assert JinaEmbeddingAdapter.profile.long_text_part_attempt_limit == 3
    assert "d1e5d70b7b34d927a8cddac458583c4fbe50a914" in (
        JinaEmbeddingAdapter.profile.tokenizer_revision
    )
    assert (
        FakeEmbeddingAdapter.profile.image_transform
        == "pillow-11.3.0-exif-transpose-alpha-white-rgb-jpeg-q85-420-lanczos-"
        "if-over20000000px-optimize0-progressive0-metadata-none-v1"
    )
    assert (
        JinaEmbeddingAdapter.profile.image_transform
        == "pillow-11.3.0-exif-transpose-alpha-white-rgb-jpeg-q85-420-lanczos-"
        "if-over20000000px-optimize0-progressive0-metadata-none-v1"
    )
    assert (
        JinaEmbeddingAdapter.profile.image_input_rules
        == "static-jpeg-png-webp-max5000000b-max20000000px-decode-max40000000px-"
        "exact-source-v1"
    )
    alternatives = {
        "model": "changed-model-alias",
        "observed_model": "changed-observed-model",
        "observed_api_version": "changed-api-version",
        "task": "changed-task",
        "dimensions": 1024,
        "output_type": "changed-output-type",
        "normalization": "changed-normalization",
        "tokenizer_revision": "changed-tokenizer",
        "tokenizer_engine": "changed-tokenizer-engine",
        "context_token_limit": 4096,
        "context_rules": "changed-context-rules",
        "long_text_aggregation": "changed-aggregation",
        "long_text_part_attempt_limit": 4,
        "image_input_rules": "changed-image-rules",
        "image_transform": "changed-image-transform",
        "multimodal_formula": "changed-multimodal-formula",
        "client_configuration_version": "changed-client-version",
    }

    assert set(asdict(profile)) == set(alternatives)
    baseline = configuration_id(profile)
    assert all(
        configuration_id(replace(profile, **{name: value})) != baseline
        for name, value in alternatives.items()
    )


def test_config_rejects_truthy_non_boolean_hosted_processing_authorization(
    tmp_path,
):
    """Break caught: an arbitrary truthy value is treated as affirmative consent."""

    with pytest.raises(EmbeddingConfigError, match="authorization must be boolean"):
        EmbeddingPipelineConfig(
            source_root=tmp_path / "source",
            preprocessing_output_root=tmp_path / "preprocessed",
            embedding_output_root=tmp_path / "embeddings",
            hosted_processing_authorized="yes",  # type: ignore[arg-type]
        )


def test_coordinator_refuses_unauthorized_run_before_markdown_or_output(
    tmp_path,
    monkeypatch,
):
    """Break caught: direct coordinator callers can bypass the CLI consent gate."""

    config = replace(
        write_generated_preprocessing_fixture(tmp_path),
        hosted_processing_authorized=False,
    )

    def fail_if_markdown_read(*_args, **_kwargs):
        raise AssertionError("unauthorized coordinator read canonical Markdown")

    monkeypatch.setattr(
        embedding_pipeline_module,
        "read_canonical_markdown",
        fail_if_markdown_read,
    )

    with pytest.raises(HostedProcessingAuthorizationError):
        run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)

    assert not config.embedding_output_root.exists()


@pytest.mark.parametrize(
    "root_state",
    ("absent", "not-directory", "inaccessible"),
)
def test_coordinator_rejects_invalid_archive_root_before_output_or_text_purchase(
    tmp_path,
    root_state,
):
    """Break caught: an unavailable archive becomes one retryable image leaf."""

    config = write_generated_preprocessing_fixture(tmp_path)
    declare_missing_generated_header_image(config)
    if root_state == "absent":
        config.source_root.rmdir()
    elif root_state == "not-directory":
        config.source_root.rmdir()
        config.source_root.write_bytes(b"generated non-directory root")
    else:
        config.source_root.chmod(0)
    adapter = RecordingMultimodalAdapter()

    try:
        with pytest.raises(EmbeddingPipelineError) as raised:
            run_embedding_pipeline(config, adapter, limit=1)
    finally:
        if root_state == "inaccessible":
            config.source_root.chmod(0o700)

    assert raised.value.code == "source_root"
    assert adapter.calls == 0
    assert adapter.image_calls == 0
    assert not config.embedding_output_root.exists()


def replace_markdown_with_unsafe_leaf(
    config: EmbeddingPipelineConfig,
    leaf_kind: str,
) -> Path:
    markdown_path = (
        config.preprocessing_output_root / "text" / "2024" / "01" / "02" / "article.md"
    )
    markdown_path.unlink()
    referent = config.preprocessing_output_root / "synthetic-referent.md"
    referent.write_text("# synthetic replacement\n", encoding="utf-8")
    if leaf_kind == "symlink":
        markdown_path.symlink_to(referent)
    elif leaf_kind == "hardlink":
        markdown_path.hardlink_to(referent)
    else:
        markdown_path.mkdir()
    return markdown_path


def test_pipeline_publishes_one_normalized_article_text_embedding(tmp_path):
    config = write_generated_preprocessing_fixture(tmp_path)
    before = snapshot_fixture_inputs(config)

    result = run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)

    assert result == EmbeddingRunResult(
        articles=1,
        embeddings=1,
        attempted=1,
        succeeded=1,
        header_absent=1,
    )
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        row = db.execute(
            """
            SELECT article_id, modality, published_at_utc,
                   publication_date_new_york, dimensions,
                   input_sha256, stored_vector_sha256, vector
            FROM embeddings
            """
        ).fetchone()
        vector_type = db.execute("PRAGMA table_info('embeddings')").fetchall()[9][2]
    assert row[:5] == (
        "wsj:SYNTHETIC-EMBEDDING",
        "article_text",
        datetime(2024, 1, 2, 15, 30, tzinfo=UTC),
        date(2024, 1, 2),
        2048,
    )
    assert vector_type == "FLOAT[2048]"
    assert row[7] == (1.0,) + (0.0,) * 2047
    assert snapshot_fixture_inputs(config) == before


def test_over_context_markdown_packs_complete_blocks_and_aggregates_every_part(
    tmp_path,
):
    """Break caught: oversized Markdown is sent once or tail blocks are discarded."""

    config = write_generated_preprocessing_fixture(tmp_path)
    markdown = "# A\n\nBBBB\n\nCC"
    article_input_sha256 = replace_generated_embedding_markdown(config, markdown)
    adapter = CharacterTokenLongTextAdapter()

    result = run_embedding_pipeline(config, adapter, limit=1)

    assert result == EmbeddingRunResult(
        articles=1,
        embeddings=1,
        attempted=1,
        succeeded=1,
        header_absent=1,
    )
    assert adapter.embedded_texts == ["# A\n\n", "BBBB\n\nCC"]
    assert "".join(adapter.embedded_texts) == markdown
    configuration_identifier = configuration_id(adapter.profile)
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute(
            """
            SELECT part_index, part_count, part_input_sha256, token_count,
                   state, attempt_count, generation_run_id IS NOT NULL,
                   stored_vector_sha256 IS NOT NULL, vector IS NOT NULL
            FROM long_text_parts
            WHERE article_id = 'wsj:SYNTHETIC-EMBEDDING'
              AND configuration_id = ? AND article_input_sha256 = ?
            ORDER BY part_index
            """,
            [configuration_identifier, article_input_sha256],
        ).fetchall() == [
            (
                0,
                2,
                hashlib.sha256(b"# A\n\n").hexdigest(),
                5,
                "succeeded",
                1,
                True,
                True,
                True,
            ),
            (
                1,
                2,
                hashlib.sha256(b"BBBB\n\nCC").hexdigest(),
                8,
                "succeeded",
                1,
                True,
                True,
                True,
            ),
        ]
        vector = db.execute(
            """
            SELECT vector FROM embeddings
            WHERE article_id = 'wsj:SYNTHETIC-EMBEDDING'
              AND modality = 'article_text' AND configuration_id = ?
            """,
            [configuration_identifier],
        ).fetchone()[0]
        provenance = db.execute(
            """
            SELECT part_index, article_input_sha256, part_count, token_count,
                   aggregation_version
            FROM article_text_aggregation_provenance
            WHERE article_id = 'wsj:SYNTHETIC-EMBEDDING'
              AND configuration_id = ?
            ORDER BY part_index
            """,
            [configuration_identifier],
        ).fetchall()
    assert vector == (
        0.5299989581108093,
        0.847998321056366,
        *(0.0 for _ in range(2046)),
    )
    assert provenance == [
        (
            0,
            article_input_sha256,
            2,
            5,
            "l2-token-count-weighted-mean-float32-v1",
        ),
        (
            1,
            article_input_sha256,
            2,
            8,
            "l2-token-count-weighted-mean-float32-v1",
        ),
    ]
    assert validate_embedding_outputs(config).ok


def test_markdown_at_exact_token_ceiling_keeps_one_hosted_input(tmp_path):
    """Break caught: the inclusive hosted boundary is split unnecessarily."""

    config = write_generated_preprocessing_fixture(tmp_path)
    markdown = "ABCDEFGH"
    replace_generated_embedding_markdown(config, markdown)
    adapter = CharacterTokenLongTextAdapter()

    run_embedding_pipeline(config, adapter, limit=1)

    assert adapter.embedded_texts == [markdown]
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute("SELECT count(*) FROM long_text_parts").fetchone() == (0,)
        assert db.execute(
            "SELECT count(*) FROM article_text_aggregation_provenance"
        ).fetchone() == (0,)


def test_one_oversized_markdown_block_splits_at_reversible_token_boundaries(
    tmp_path,
):
    """Break caught: splitting one huge block drops, overlaps, or rewrites text."""

    config = write_generated_preprocessing_fixture(tmp_path)
    markdown = "ABCDEFGHIJKLMNOPQR"
    replace_generated_embedding_markdown(config, markdown)
    adapter = CharacterTokenLongTextAdapter()

    run_embedding_pipeline(config, adapter, limit=1)

    assert adapter.embedded_texts == ["ABCDEFGH", "IJKLMNOP", "QR"]
    assert "".join(adapter.embedded_texts) == markdown
    assert [len(part) for part in adapter.embedded_texts] == [8, 8, 2]


def test_oversized_block_respects_overlapping_byte_fallback_token_offsets(tmp_path):
    """Break caught: multiple tokenizer tokens for one codepoint lose that text."""

    class ByteFallbackTokenizerAdapter(CharacterTokenLongTextAdapter):
        profile = replace(
            CharacterTokenLongTextAdapter.profile,
            context_token_limit=2,
        )

        def token_offsets(self, text: str) -> tuple[tuple[int, int], ...]:
            offsets: list[tuple[int, int]] = []
            for index, character in enumerate(text):
                offsets.append((index, index + 1))
                if character == "😊":
                    offsets.append((index, index + 1))
            return tuple(offsets)

    config = write_generated_preprocessing_fixture(tmp_path)
    markdown = "a😊b"
    replace_generated_embedding_markdown(config, markdown)
    adapter = ByteFallbackTokenizerAdapter()

    run_embedding_pipeline(config, adapter, limit=1)

    assert adapter.embedded_texts == ["a", "😊", "b"]
    assert "".join(adapter.embedded_texts) == markdown
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        expected_spans = [
            (0, 0, 1, 0, 1),
            (1, 1, 2, 1, 5),
            (2, 2, 3, 5, 6),
        ]
        assert db.execute(
            """
            SELECT part_index, char_start, char_end, byte_start, byte_end
            FROM long_text_parts
            ORDER BY part_index
            """
        ).fetchall() == expected_spans
        assert db.execute(
            """
            SELECT part_index, char_start, char_end, byte_start, byte_end
            FROM long_text_part_generations
            ORDER BY part_index
            """
        ).fetchall() == expected_spans


class StablePartVectorAdapter(CharacterTokenLongTextAdapter):
    """Return one stable orthogonal fixture vector per generated block."""

    def embed_text(self, text: str) -> tuple[float, ...]:
        self.embedded_texts.append(text)
        axis = {"A": 0, "B": 1, "C": 2}[text[0]]
        return tuple(1.0 if index == axis else 0.0 for index in range(2048))


def test_interrupted_long_text_part_resume_does_not_repurchase_committed_part(
    tmp_path,
):
    """Break caught: a crash after part one causes its hosted call to repeat."""

    config = write_generated_preprocessing_fixture(tmp_path)
    markdown = "AAAA\n\nBBBB\n\nCCCC"
    article_input_sha256 = replace_generated_embedding_markdown(config, markdown)

    class InterruptSecondPartAdapter(StablePartVectorAdapter):
        def embed_text(self, text: str) -> tuple[float, ...]:
            if text.startswith("BBBB"):
                self.embedded_texts.append(text)
                raise SyntheticInterruption
            return super().embed_text(text)

    interrupted_adapter = InterruptSecondPartAdapter()
    with pytest.raises(SyntheticInterruption):
        run_embedding_pipeline(config, interrupted_adapter, limit=1)

    configuration_identifier = configuration_id(interrupted_adapter.profile)
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute(
            """
            SELECT part_index, state, attempt_count, vector IS NOT NULL
            FROM long_text_parts
            WHERE configuration_id = ? AND article_input_sha256 = ?
            ORDER BY part_index
            """,
            [configuration_identifier, article_input_sha256],
        ).fetchall() == [
            (0, "succeeded", 1, True),
            (1, "in_progress", 1, False),
            (2, "queued", 0, False),
        ]

    resumed_adapter = StablePartVectorAdapter()
    resumed = run_embedding_pipeline(config, resumed_adapter, limit=1)

    assert resumed == EmbeddingRunResult(
        articles=1,
        embeddings=1,
        attempted=1,
        succeeded=1,
        interrupted=1,
        header_absent=1,
    )
    assert resumed_adapter.embedded_texts == ["BBBB\n\n", "CCCC"]
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute(
            """
            SELECT part_index, state, attempt_count
            FROM long_text_parts
            WHERE configuration_id = ? AND article_input_sha256 = ?
            ORDER BY part_index
            """,
            [configuration_identifier, article_input_sha256],
        ).fetchall() == [
            (0, "succeeded", 1),
            (1, "succeeded", 2),
            (2, "succeeded", 1),
        ]


def test_failed_long_text_part_is_content_free_and_resumes_remaining_parts(tmp_path):
    """Break caught: a part failure leaks text or discards successful checkpoints."""

    config = write_generated_preprocessing_fixture(tmp_path)
    markdown = "AAAA\n\nBBBB\n\nCCCC"
    article_input_sha256 = replace_generated_embedding_markdown(config, markdown)

    class FailSecondPartAdapter(StablePartVectorAdapter):
        def embed_text(self, text: str) -> tuple[float, ...]:
            if text.startswith("BBBB"):
                self.embedded_texts.append(text)
                raise JinaHostedAdapterError(
                    "rate_limit",
                    retryable=True,
                    status_code=429,
                    retry_after_seconds=2.0,
                )
            return super().embed_text(text)

    with pytest.raises(JinaHostedAdapterError) as raised:
        run_embedding_pipeline(config, FailSecondPartAdapter(), limit=1)
    assert "AAAA" not in str(raised.value)
    assert "BBBB" not in str(raised.value)
    assert "CCCC" not in str(raised.value)

    configuration_identifier = configuration_id(StablePartVectorAdapter.profile)
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute(
            """
            SELECT part_index, state, attempt_count, error_code, status_code,
                   retry_after_seconds
            FROM long_text_parts
            WHERE configuration_id = ? AND article_input_sha256 = ?
            ORDER BY part_index
            """,
            [configuration_identifier, article_input_sha256],
        ).fetchall() == [
            (0, "succeeded", 1, None, None, None),
            (1, "retryable", 1, "rate_limit", 429, 2.0),
            (2, "queued", 0, None, None, None),
        ]
        assert db.execute("SELECT count(*) FROM embeddings").fetchone() == (0,)

    resumed_adapter = StablePartVectorAdapter()
    result = run_embedding_pipeline(config, resumed_adapter, limit=1)

    assert result == EmbeddingRunResult(
        articles=1,
        embeddings=1,
        attempted=1,
        succeeded=1,
        header_absent=1,
    )
    assert resumed_adapter.embedded_texts == ["BBBB\n\n", "CCCC"]


def test_retryable_long_text_part_becomes_terminal_at_profile_attempt_limit(
    tmp_path,
):
    """Break caught: a retryable part is repurchased forever across replays."""

    config = write_generated_preprocessing_fixture(tmp_path)
    article_input_sha256 = replace_generated_embedding_markdown(
        config,
        "AAAA\n\nBBBB\n\nCCCC",
    )

    class AlwaysRetrySecondPartAdapter(StablePartVectorAdapter):
        def embed_text(self, text: str) -> tuple[float, ...]:
            if text.startswith("BBBB"):
                self.embedded_texts.append(text)
                raise JinaHostedAdapterError(
                    "rate_limit",
                    retryable=True,
                    status_code=429,
                    retry_after_seconds=2.0,
                )
            return super().embed_text(text)

    adapter = AlwaysRetrySecondPartAdapter()
    for _attempt in range(adapter.profile.long_text_part_attempt_limit):
        with pytest.raises(JinaHostedAdapterError):
            run_embedding_pipeline(config, adapter, limit=1)

    configuration_identifier = configuration_id(adapter.profile)
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute(
            """
            SELECT state, attempt_count, error_code
            FROM long_text_parts
            WHERE configuration_id = ? AND article_input_sha256 = ?
              AND part_index = 1
            """,
            [configuration_identifier, article_input_sha256],
        ).fetchone() == ("terminal", 3, "rate_limit")
        assert db.execute(
            """
            SELECT state, attempt_count, error_code
            FROM embedding_work_items
            WHERE configuration_id = ? AND modality = 'article_text'
            """,
            [configuration_identifier],
        ).fetchone() == ("terminal", 3, "rate_limit")

    calls_before_replay = tuple(adapter.embedded_texts)
    replay = run_embedding_pipeline(config, adapter, limit=1)

    assert replay == EmbeddingRunResult(
        articles=1,
        embeddings=0,
        terminal=1,
        header_absent=1,
    )
    assert tuple(adapter.embedded_texts) == calls_before_replay


def test_explicit_reprocess_regenerates_terminal_same_input_long_text(tmp_path):
    """Break caught: terminal replay repair blocks deliberate reprocessing."""

    config = write_generated_preprocessing_fixture(tmp_path)
    article_input_sha256 = replace_generated_embedding_markdown(
        config,
        "AAAA\n\nBBBB\n\nCCCC",
    )
    initial_adapter = StablePartVectorAdapter()
    run_embedding_pipeline(config, initial_adapter, limit=1)
    configuration_identifier = configuration_id(initial_adapter.profile)
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        first_aggregate_run = db.execute(
            """
            SELECT generation_run_id
            FROM embedding_work_items
            WHERE configuration_id = ? AND modality = 'article_text'
            """,
            [configuration_identifier],
        ).fetchone()[0]
        first_provenance = db.execute(
            """
            SELECT p.part_index, p.part_generation_run_id,
                   g.part_input_sha256, g.stored_vector_sha256
            FROM article_text_aggregation_provenance AS p
            JOIN long_text_part_generations AS g
              ON g.article_id = p.article_id
             AND g.configuration_id = p.configuration_id
             AND g.article_input_sha256 = p.article_input_sha256
             AND g.part_index = p.part_index
             AND g.generation_run_id = p.part_generation_run_id
            WHERE p.generation_run_id = ?
            ORDER BY p.part_index
            """,
            [first_aggregate_run],
        ).fetchall()

    class AlwaysRetrySecondPartAdapter(StablePartVectorAdapter):
        def embed_text(self, text: str) -> tuple[float, ...]:
            if text.startswith("BBBB"):
                self.embedded_texts.append(text)
                raise JinaHostedAdapterError("rate_limit", retryable=True)
            return super().embed_text(text)

    failing_adapter = AlwaysRetrySecondPartAdapter()
    with pytest.raises(JinaHostedAdapterError):
        run_embedding_pipeline(
            config,
            failing_adapter,
            limit=1,
            reprocess=True,
        )
    for _attempt in range(2):
        with pytest.raises(JinaHostedAdapterError):
            run_embedding_pipeline(config, failing_adapter, limit=1)

    calls_before_replay = tuple(failing_adapter.embedded_texts)
    terminal_replay = run_embedding_pipeline(config, failing_adapter, limit=1)
    assert terminal_replay == EmbeddingRunResult(
        articles=1,
        embeddings=0,
        terminal=1,
        header_absent=1,
    )
    assert tuple(failing_adapter.embedded_texts) == calls_before_replay

    checkpoint_states: list[list[tuple[int, str, int]]] = []

    def observe_first_reprocess_checkpoint(
        part_index: int,
        attempt_count: int,
    ) -> None:
        if part_index != 0:
            return
        assert attempt_count == 1
        with duckdb.connect(str(config.embedding_catalog)) as db:
            checkpoint_states.append(
                db.execute(
                    """
                    SELECT part_index, state, attempt_count
                    FROM long_text_parts
                    WHERE configuration_id = ? AND article_input_sha256 = ?
                    ORDER BY part_index
                    """,
                    [configuration_identifier, article_input_sha256],
                ).fetchall()
            )

    regenerated_adapter = StablePartVectorAdapter()
    regenerated = run_embedding_pipeline(
        config,
        regenerated_adapter,
        limit=1,
        reprocess=True,
        failpoints=EmbeddingPipelineFailpoints(
            after_long_text_part_attempt_checkpoint=(
                observe_first_reprocess_checkpoint
            ),
        ),
    )

    assert regenerated == EmbeddingRunResult(
        articles=1,
        embeddings=1,
        attempted=1,
        succeeded=1,
        header_absent=1,
    )
    assert regenerated_adapter.embedded_texts == [
        "AAAA\n\n",
        "BBBB\n\n",
        "CCCC",
    ]
    assert checkpoint_states == [
        [
            (0, "in_progress", 1),
            (1, "queued", 0),
            (2, "queued", 0),
        ]
    ]
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        parent = db.execute(
            """
            SELECT state, attempt_count, generation_run_id
            FROM embedding_work_items
            WHERE configuration_id = ? AND modality = 'article_text'
            """,
            [configuration_identifier],
        ).fetchone()
        assert parent[:2] == ("succeeded", 1)
        final_aggregate_run = parent[2]
        assert final_aggregate_run != first_aggregate_run
        assert db.execute(
            """
            SELECT part_index, state, attempt_count, generation_run_id
            FROM long_text_parts
            WHERE configuration_id = ? AND article_input_sha256 = ?
            ORDER BY part_index
            """,
            [configuration_identifier, article_input_sha256],
        ).fetchall() == [
            (0, "succeeded", 1, final_aggregate_run),
            (1, "succeeded", 1, final_aggregate_run),
            (2, "succeeded", 1, final_aggregate_run),
        ]
        assert db.execute(
            """
            SELECT part_index, part_generation_run_id
            FROM article_text_aggregation_provenance
            WHERE generation_run_id = ?
            ORDER BY part_index
            """,
            [final_aggregate_run],
        ).fetchall() == [
            (0, final_aggregate_run),
            (1, final_aggregate_run),
            (2, final_aggregate_run),
        ]
        assert db.execute(
            """
            SELECT count(*) FROM long_text_part_generations
            WHERE generation_run_id = ?
            """,
            [final_aggregate_run],
        ).fetchone() == (3,)
        assert db.execute(
            """
            SELECT count(*) FROM embedding_generation_history
            WHERE modality = 'article_text' AND generation_run_id = ?
              AND superseded_reason = 'explicit_reprocess'
            """,
            [first_aggregate_run],
        ).fetchone() == (1,)
        assert db.execute(
            """
            SELECT p.part_index, p.part_generation_run_id,
                   g.part_input_sha256, g.stored_vector_sha256
            FROM article_text_aggregation_provenance AS p
            JOIN long_text_part_generations AS g
              ON g.article_id = p.article_id
             AND g.configuration_id = p.configuration_id
             AND g.article_input_sha256 = p.article_input_sha256
             AND g.part_index = p.part_index
             AND g.generation_run_id = p.part_generation_run_id
            WHERE p.generation_run_id = ?
            ORDER BY p.part_index
            """,
            [first_aggregate_run],
        ).fetchall() == first_provenance
    assert validate_embedding_outputs(config).ok


def test_terminal_part_transition_survives_crash_before_outer_failure_handling(
    tmp_path,
):
    """Break caught: replay repurchases a terminal part after a seam crash."""

    config = write_generated_preprocessing_fixture(tmp_path)
    replace_generated_embedding_markdown(config, "AAAA\n\nBBBB\n\nCCCC")

    class AlwaysRetrySecondPartAdapter(StablePartVectorAdapter):
        def embed_text(self, text: str) -> tuple[float, ...]:
            if text.startswith("BBBB"):
                self.embedded_texts.append(text)
                raise JinaHostedAdapterError("rate_limit", retryable=True)
            return super().embed_text(text)

    adapter = AlwaysRetrySecondPartAdapter()
    for _attempt in range(2):
        with pytest.raises(JinaHostedAdapterError):
            run_embedding_pipeline(config, adapter, limit=1)

    def interrupt_after_terminal_transition(part_index: int) -> None:
        assert part_index == 1
        raise SyntheticInterruption

    failpoints = EmbeddingPipelineFailpoints(
        after_long_text_terminal_transition=interrupt_after_terminal_transition,
    )
    with pytest.raises(SyntheticInterruption):
        run_embedding_pipeline(config, adapter, limit=1, failpoints=failpoints)

    configuration_identifier = configuration_id(adapter.profile)
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute(
            """
            SELECT state, attempt_count FROM long_text_parts
            WHERE configuration_id = ? AND part_index = 1
            """,
            [configuration_identifier],
        ).fetchone() == ("terminal", 3)
        assert db.execute(
            """
            SELECT state, attempt_count, error_code
            FROM embedding_work_items
            WHERE configuration_id = ? AND modality = 'article_text'
            """,
            [configuration_identifier],
        ).fetchone() == ("terminal", 3, "rate_limit")
        assert db.execute(
            """
            SELECT attempted, terminal, interrupted
            FROM runs
            WHERE run_id = (
                SELECT last_run_id FROM embedding_work_items
                WHERE configuration_id = ? AND modality = 'article_text'
            )
            """,
            [configuration_identifier],
        ).fetchone() == (1, 1, 0)

    replay_adapter = StablePartVectorAdapter()
    replay = run_embedding_pipeline(config, replay_adapter, limit=1)

    assert replay == EmbeddingRunResult(
        articles=1,
        embeddings=0,
        terminal=1,
        header_absent=1,
    )
    assert replay_adapter.embedded_texts == []


def test_attempt_limit_checkpoint_crash_replay_terminalizes_without_adapter_call(
    tmp_path,
):
    """Break caught: replay buys attempt four after attempt three checkpointed."""

    config = write_generated_preprocessing_fixture(tmp_path)
    replace_generated_embedding_markdown(config, "AAAA\n\nBBBB\n\nCCCC")

    class AlwaysRetrySecondPartAdapter(StablePartVectorAdapter):
        def embed_text(self, text: str) -> tuple[float, ...]:
            if text.startswith("BBBB"):
                self.embedded_texts.append(text)
                raise JinaHostedAdapterError("rate_limit", retryable=True)
            return super().embed_text(text)

    adapter = AlwaysRetrySecondPartAdapter()
    for _attempt in range(2):
        with pytest.raises(JinaHostedAdapterError):
            run_embedding_pipeline(config, adapter, limit=1)
    calls_before_checkpoint = tuple(adapter.embedded_texts)

    def interrupt_after_attempt_checkpoint(
        part_index: int,
        attempt_count: int,
    ) -> None:
        if part_index == 1 and attempt_count == 3:
            raise SyntheticInterruption

    failpoints = EmbeddingPipelineFailpoints(
        after_long_text_part_attempt_checkpoint=(
            interrupt_after_attempt_checkpoint
        ),
    )
    with pytest.raises(SyntheticInterruption):
        run_embedding_pipeline(config, adapter, limit=1, failpoints=failpoints)
    assert tuple(adapter.embedded_texts) == calls_before_checkpoint

    replay_adapter = StablePartVectorAdapter()
    replay = run_embedding_pipeline(config, replay_adapter, limit=1)

    assert replay == EmbeddingRunResult(
        articles=1,
        embeddings=0,
        terminal=1,
        interrupted=1,
        header_absent=1,
    )
    assert replay_adapter.embedded_texts == []
    configuration_identifier = configuration_id(adapter.profile)
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute(
            """
            SELECT state, attempt_count, error_code
            FROM long_text_parts
            WHERE configuration_id = ? AND part_index = 1
            """,
            [configuration_identifier],
        ).fetchone() == ("terminal", 3, "attempt_limit_exhausted")
        assert db.execute(
            """
            SELECT state, attempt_count, error_code
            FROM embedding_work_items
            WHERE configuration_id = ? AND modality = 'article_text'
            """,
            [configuration_identifier],
        ).fetchone() == ("terminal", 3, "attempt_limit_exhausted")


def test_validator_requires_complete_long_text_aggregation_provenance(tmp_path):
    """Break caught: deleting part linkage leaves an aggregate looking complete."""

    config = write_generated_preprocessing_fixture(tmp_path)
    replace_generated_embedding_markdown(config, "AAAA\n\nBBBB\n\nCCCC")
    run_embedding_pipeline(config, StablePartVectorAdapter(), limit=1)
    with duckdb.connect(str(config.embedding_catalog)) as db:
        db.execute(
            """
            DELETE FROM article_text_aggregation_provenance
            WHERE part_index = 1
            """
        )

    codes = {issue.code for issue in validate_embedding_outputs(config).issues}

    assert "missing_long_text_provenance" in codes


def test_validator_recomputes_long_text_aggregate_from_all_weighted_parts(tmp_path):
    """Break caught: self-consistent part tampering is not checked against aggregate."""

    config = write_generated_preprocessing_fixture(tmp_path)
    replace_generated_embedding_markdown(config, "AAAA\n\nBBBB\n\nCCCC")
    run_embedding_pipeline(config, StablePartVectorAdapter(), limit=1)
    replacement_vector = [0.0, 0.0, 0.0, 1.0, *([0.0] * 2044)]
    replacement_hash = hashlib.sha256(
        b"".join(struct.pack("<f", value) for value in replacement_vector)
    ).hexdigest()
    with duckdb.connect(str(config.embedding_catalog)) as db:
        db.execute(
            """
            UPDATE long_text_parts
            SET vector = ?, stored_vector_sha256 = ?
            WHERE part_index = 0
            """,
            [replacement_vector, replacement_hash],
        )
        db.execute(
            """
            UPDATE long_text_part_generations
            SET vector = ?, stored_vector_sha256 = ?
            WHERE part_index = 0
            """,
            [replacement_vector, replacement_hash],
        )
        db.execute(
            """
            UPDATE article_text_aggregation_provenance
            SET part_stored_vector_sha256 = ?
            WHERE part_index = 0
            """,
            [replacement_hash],
        )

    codes = {issue.code for issue in validate_embedding_outputs(config).issues}

    assert "long_text_vector_mismatch" in codes


def test_long_text_reprocess_preserves_every_provenance_linked_part_generation(
    tmp_path,
):
    """Break caught: reprocess erases parts named by archived aggregate provenance."""

    config = write_generated_preprocessing_fixture(tmp_path)
    replace_generated_embedding_markdown(config, "AAAA\n\nBBBB\n\nCCCC")
    run_embedding_pipeline(config, StablePartVectorAdapter(), limit=1)
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        first_aggregate_run = db.execute(
            """
            SELECT generation_run_id FROM embedding_work_items
            WHERE modality = 'article_text'
            """
        ).fetchone()[0]
        first_part_generations = db.execute(
            """
            SELECT p.part_index, p.part_generation_run_id,
                   g.part_input_sha256, g.token_count,
                   g.stored_vector_sha256, g.vector
            FROM article_text_aggregation_provenance AS p
            JOIN long_text_part_generations AS g
              ON g.article_id = p.article_id
             AND g.configuration_id = p.configuration_id
             AND g.article_input_sha256 = p.article_input_sha256
             AND g.part_index = p.part_index
             AND g.generation_run_id = p.part_generation_run_id
            WHERE p.generation_run_id = ?
            ORDER BY p.part_index
            """,
            [first_aggregate_run],
        ).fetchall()

    run_embedding_pipeline(
        config,
        StablePartVectorAdapter(),
        limit=1,
        reprocess=True,
    )

    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute(
            """
            SELECT count(*)
            FROM embedding_generation_history
            WHERE modality = 'article_text'
              AND generation_run_id = ?
              AND superseded_reason = 'explicit_reprocess'
            """,
            [first_aggregate_run],
        ).fetchone() == (1,)
        assert db.execute(
            """
            SELECT p.part_index, p.part_generation_run_id,
                   g.part_input_sha256, g.token_count,
                   g.stored_vector_sha256, g.vector
            FROM article_text_aggregation_provenance AS p
            JOIN long_text_part_generations AS g
              ON g.article_id = p.article_id
             AND g.configuration_id = p.configuration_id
             AND g.article_input_sha256 = p.article_input_sha256
             AND g.part_index = p.part_index
             AND g.generation_run_id = p.part_generation_run_id
            WHERE p.generation_run_id = ?
            ORDER BY p.part_index
            """,
            [first_aggregate_run],
        ).fetchall() == first_part_generations
        assert db.execute(
            "SELECT count(*) FROM long_text_part_generations"
        ).fetchone() == (6,)
    assert validate_embedding_outputs(config).ok


def test_validator_resolves_archived_aggregate_part_generations(tmp_path):
    """Break caught: archived aggregate provenance points to a missing part."""

    config = write_generated_preprocessing_fixture(tmp_path)
    replace_generated_embedding_markdown(config, "AAAA\n\nBBBB\n\nCCCC")
    run_embedding_pipeline(config, StablePartVectorAdapter(), limit=1)
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        first_aggregate_run = db.execute(
            """
            SELECT generation_run_id FROM embedding_work_items
            WHERE modality = 'article_text'
            """
        ).fetchone()[0]
    run_embedding_pipeline(
        config,
        StablePartVectorAdapter(),
        limit=1,
        reprocess=True,
    )
    with duckdb.connect(str(config.embedding_catalog)) as db:
        db.execute(
            """
            DELETE FROM long_text_part_generations
            WHERE generation_run_id = (
                SELECT part_generation_run_id
                FROM article_text_aggregation_provenance
                WHERE generation_run_id = ? AND part_index = 1
            ) AND part_index = 1
            """,
            [first_aggregate_run],
        )

    codes = {issue.code for issue in validate_embedding_outputs(config).issues}

    assert "invalid_long_text_provenance" in codes


@pytest.mark.parametrize(
    "spans",
    (
        ((0, 5), (6, 12), (12, 16)),
        ((0, 7), (6, 12), (12, 16)),
        ((6, 12), (0, 6), (12, 16)),
    ),
    ids=("gap", "overlap", "reorder"),
)
def test_validator_reopens_canonical_markdown_for_exact_part_span_coverage(
    tmp_path,
    spans,
):
    """Break caught: self-consistent part spans do not cover canonical Markdown."""

    config = write_generated_preprocessing_fixture(tmp_path)
    markdown = "AAAA\n\nBBBB\n\nCCCC"
    replace_generated_embedding_markdown(config, markdown)
    run_embedding_pipeline(config, StablePartVectorAdapter(), limit=1)
    with duckdb.connect(str(config.embedding_catalog)) as db:
        for part_index, (start, end) in enumerate(spans):
            part_sha256 = hashlib.sha256(markdown[start:end].encode()).hexdigest()
            parameters = [
                start,
                end,
                start,
                end,
                part_sha256,
                end - start,
                part_index,
            ]
            db.execute(
                """
                UPDATE long_text_parts
                SET char_start = ?, char_end = ?, byte_start = ?, byte_end = ?,
                    part_input_sha256 = ?, token_count = ?
                WHERE part_index = ?
                """,
                parameters,
            )
            db.execute(
                """
                UPDATE long_text_part_generations
                SET char_start = ?, char_end = ?, byte_start = ?, byte_end = ?,
                    part_input_sha256 = ?, token_count = ?
                WHERE part_index = ?
                """,
                parameters,
            )
            db.execute(
                """
                UPDATE article_text_aggregation_provenance
                SET part_input_sha256 = ?, token_count = ?
                WHERE part_index = ?
                """,
                [part_sha256, end - start, part_index],
            )

    codes = {issue.code for issue in validate_embedding_outputs(config).issues}

    assert "invalid_long_text_source_coverage" in codes


def test_changed_long_markdown_replaces_aggregate_and_multimodal_not_old_parts(
    tmp_path,
):
    """Break caught: new Markdown reuses old parts or leaves its composite stale."""

    config = write_generated_preprocessing_fixture(tmp_path)
    first_markdown = "AAAA\n\nBBBB\n\nCCCC"
    first_sha256 = replace_generated_embedding_markdown(config, first_markdown)
    attach_generated_header_image(config, b"generated long-text header")
    run_embedding_pipeline(config, StablePartVectorAdapter(), limit=1)
    configuration_identifier = configuration_id(StablePartVectorAdapter.profile)
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        before_generations = dict(
            db.execute(
                """
                SELECT modality, generation_run_id
                FROM embedding_work_items
                WHERE configuration_id = ?
                """,
                [configuration_identifier],
            ).fetchall()
        )

    second_markdown = "AAAA\n\nBBBB\n\nCCCD"
    second_sha256 = replace_generated_embedding_markdown(config, second_markdown)
    adapter = StablePartVectorAdapter()
    result = run_embedding_pipeline(config, adapter, limit=1)

    assert result == EmbeddingRunResult(
        articles=1,
        embeddings=2,
        reused=1,
        attempted=1,
        succeeded=2,
    )
    assert adapter.embedded_texts == ["AAAA\n\n", "BBBB\n\n", "CCCD"]
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        after_generations = dict(
            db.execute(
                """
                SELECT modality, generation_run_id
                FROM embedding_work_items
                WHERE configuration_id = ?
                """,
                [configuration_identifier],
            ).fetchall()
        )
        assert db.execute(
            """
            SELECT article_input_sha256, count(*)
            FROM long_text_parts
            WHERE configuration_id = ?
            GROUP BY article_input_sha256
            ORDER BY article_input_sha256
            """,
            [configuration_identifier],
        ).fetchall() == sorted([(first_sha256, 3), (second_sha256, 3)])
        assert db.execute(
            """
            SELECT modality, superseded_reason
            FROM embedding_generation_history
            WHERE configuration_id = ?
            ORDER BY modality
            """,
            [configuration_identifier],
        ).fetchall() == [
            ("article_text", "input_changed"),
            ("multimodal_article", "input_changed"),
        ]
    assert after_generations["header_image"] == before_generations["header_image"]
    assert after_generations["article_text"] != before_generations["article_text"]
    assert (
        after_generations["multimodal_article"]
        != before_generations["multimodal_article"]
    )
    assert validate_embedding_outputs(config).ok


def test_changed_chunking_configuration_creates_distinct_parts_and_aggregate(
    tmp_path,
):
    """Break caught: a changed ceiling silently reinterprets old part vectors."""

    config = write_generated_preprocessing_fixture(tmp_path)
    replace_generated_embedding_markdown(config, "AAAA\n\nBBBB\n\nCCCC")
    first_adapter = StablePartVectorAdapter()
    run_embedding_pipeline(config, first_adapter, limit=1)

    class SmallerCeilingAdapter(StablePartVectorAdapter):
        profile = replace(
            StablePartVectorAdapter.profile,
            context_token_limit=7,
        )

    second_adapter = SmallerCeilingAdapter()
    run_embedding_pipeline(config, second_adapter, limit=1)
    first_configuration = configuration_id(first_adapter.profile)
    second_configuration = configuration_id(second_adapter.profile)

    assert first_configuration != second_configuration
    assert first_adapter.embedded_texts == ["AAAA\n\n", "BBBB\n\n", "CCCC"]
    assert second_adapter.embedded_texts == ["AAAA\n\n", "BBBB\n\n", "CCCC"]
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute(
            """
            SELECT configuration_id, count(*)
            FROM long_text_parts
            GROUP BY configuration_id
            ORDER BY configuration_id
            """
        ).fetchall() == sorted(
            [(first_configuration, 3), (second_configuration, 3)]
        )
        assert db.execute(
            """
            SELECT configuration_id, count(*)
            FROM embeddings
            WHERE modality = 'article_text'
            GROUP BY configuration_id
            ORDER BY configuration_id
            """
        ).fetchall() == sorted(
            [(first_configuration, 1), (second_configuration, 1)]
        )
    assert validate_embedding_outputs(
        config, configuration_id=first_configuration
    ).ok
    assert validate_embedding_outputs(
        config, configuration_id=second_configuration
    ).ok


def test_pipeline_publishes_all_three_modalities_with_source_provenance(tmp_path):
    """Break caught: eligible source vectors do not produce the exact composite."""

    config = write_generated_preprocessing_fixture(tmp_path)
    image_bytes = b"\x89PNG\r\n\x1a\nsynthetic-header-image"
    relative_path = attach_generated_header_image(config, image_bytes)
    adapter = ScenarioImageAdapter()
    image_codec = ScenarioImageCodec(
        {image_bytes: ImageInfo("PNG", 640, 360)}
    )

    result = run_embedding_pipeline(
        config,
        adapter,
        limit=1,
        image_codec=image_codec,
    )

    assert result == EmbeddingRunResult(
        articles=1,
        embeddings=3,
        attempted=2,
        succeeded=3,
    )
    assert adapter.calls == 1
    assert adapter.image_calls == 1
    assert adapter.encoded_images == [base64.b64encode(image_bytes).decode("ascii")]
    configuration_identifier = configuration_id(adapter.profile)
    image_sha256 = hashlib.sha256(image_bytes).hexdigest()
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        rows = db.execute(
            """
            SELECT article_id, modality, configuration_id, source_relative_path,
                   input_sha256, stored_vector_sha256, vector
            FROM embeddings
            ORDER BY modality
            """
        ).fetchall()
        work = db.execute(
            """
            SELECT modality, source_relative_path, input_sha256, state,
                   attempt_count
            FROM embedding_work_items
            ORDER BY modality
            """
        ).fetchall()
        provenance = db.execute(
            """
            SELECT p.article_id, p.configuration_id, p.generation_run_id,
                   p.text_generation_run_id, p.text_stored_vector_sha256,
                   p.header_image_generation_run_id,
                   p.header_image_stored_vector_sha256, p.formula_version,
                   text_work.generation_run_id, text_vector.stored_vector_sha256,
                   image_work.generation_run_id, image_vector.stored_vector_sha256,
                   composite_work.generation_run_id
            FROM multimodal_embedding_provenance AS p
            JOIN embedding_work_items AS text_work
              ON text_work.article_id = p.article_id
             AND text_work.configuration_id = p.configuration_id
             AND text_work.modality = 'article_text'
            JOIN embeddings AS text_vector
              ON text_vector.article_id = p.article_id
             AND text_vector.configuration_id = p.configuration_id
             AND text_vector.modality = 'article_text'
            JOIN embedding_work_items AS image_work
              ON image_work.article_id = p.article_id
             AND image_work.configuration_id = p.configuration_id
             AND image_work.modality = 'header_image'
            JOIN embeddings AS image_vector
              ON image_vector.article_id = p.article_id
             AND image_vector.configuration_id = p.configuration_id
             AND image_vector.modality = 'header_image'
            JOIN embedding_work_items AS composite_work
              ON composite_work.article_id = p.article_id
             AND composite_work.configuration_id = p.configuration_id
             AND composite_work.modality = 'multimodal_article'
            """
        ).fetchone()
        image_provenance = db.execute(
            """
            SELECT source_sha256, source_format, source_bytes,
                   source_width, source_height, source_frames,
                   embedded_input_sha256,
                   embedded_format, embedded_bytes, embedded_width,
                   embedded_height, embedded_frames, transform_id,
                   rendition_relative_path
            FROM image_input_provenance
            """
        ).fetchone()
    assert rows[0][:4] == (
        "wsj:SYNTHETIC-EMBEDDING",
        "article_text",
        configuration_identifier,
        None,
    )
    assert rows[1][:5] == (
        "wsj:SYNTHETIC-EMBEDDING",
        "header_image",
        configuration_identifier,
        relative_path,
        image_sha256,
    )
    assert len(rows[1][5]) == 64
    assert rows[1][6] == (0.0, 1.0) + (0.0,) * 2046
    assert image_provenance == (
        image_sha256,
        "PNG",
        len(image_bytes),
        640,
        360,
        1,
        image_sha256,
        "PNG",
        len(image_bytes),
        640,
        360,
        1,
        "exact-source-bytes-v1",
        None,
    )
    assert not (config.embedding_output_root / "renditions").exists()
    assert rows[2][:5] == (
        "wsj:SYNTHETIC-EMBEDDING",
        "multimodal_article",
        configuration_identifier,
        None,
        rows[2][4],
    )
    assert len(rows[2][4]) == 64
    assert rows[2][6] == (
        0.7071067690849304,
        0.7071067690849304,
    ) + (0.0,) * 2046
    assert work == [
        ("article_text", None, hashlib.sha256(b"#\n").hexdigest(), "succeeded", 1),
        ("header_image", relative_path, image_sha256, "succeeded", 1),
        ("multimodal_article", None, rows[2][4], "succeeded", 0),
    ]
    assert provenance is not None
    assert provenance[:2] == (
        "wsj:SYNTHETIC-EMBEDDING",
        configuration_identifier,
    )
    assert provenance[2] == provenance[12]
    assert provenance[3:7] == provenance[8:12]
    assert provenance[7] == "l2-normalize-0.5-text-0.5-image-v1"
    assert validate_embedding_outputs(
        config,
        image_codec=image_codec,
    ) == EmbeddingValidationResult(issues=())


def test_composite_only_recovery_reuses_both_source_generations(
    tmp_path,
    monkeypatch,
):
    """Break caught: composite recovery calls the adapter or rewrites a source."""

    config = write_generated_preprocessing_fixture(tmp_path)
    attach_generated_header_image(config, b"generated recovery image")

    def interrupt_composite(*args, **kwargs):
        raise SyntheticInterruption

    monkeypatch.setattr(
        embedding_pipeline_module,
        "_publish_multimodal_article",
        interrupt_composite,
        raising=False,
    )
    with pytest.raises(SyntheticInterruption):
        run_embedding_pipeline(config, RecordingMultimodalAdapter(), limit=1)
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        source_generations = db.execute(
            """
            SELECT modality, generation_run_id
            FROM embedding_work_items
            WHERE modality IN ('article_text', 'header_image')
            ORDER BY modality
            """
        ).fetchall()
        assert db.execute(
            "SELECT modality FROM embeddings ORDER BY modality"
        ).fetchall() == [("article_text",), ("header_image",)]

    monkeypatch.undo()
    adapter = RecordingMultimodalAdapter()
    recovered = run_embedding_pipeline(config, adapter, limit=1)

    assert recovered == EmbeddingRunResult(
        articles=1,
        embeddings=1,
        reused=2,
        succeeded=1,
    )
    assert adapter.calls == 0
    assert adapter.image_calls == 0
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute(
            """
            SELECT modality, generation_run_id
            FROM embedding_work_items
            WHERE modality IN ('article_text', 'header_image')
            ORDER BY modality
            """
        ).fetchall() == source_generations
        assert db.execute(
            "SELECT modality FROM embeddings ORDER BY modality"
        ).fetchall() == [
            ("article_text",),
            ("header_image",),
            ("multimodal_article",),
        ]


@pytest.mark.parametrize(
    ("damage", "expected_code"),
    (
        ("vector", "multimodal_vector_mismatch"),
        ("source_link", "invalid_multimodal_source_linkage"),
    ),
)
def test_validator_recomputes_composite_and_checks_immutable_linkage(
    tmp_path,
    damage,
    expected_code,
):
    """Break caught: self-consistent composite corruption passes validation."""

    config = write_generated_preprocessing_fixture(tmp_path)
    attach_generated_header_image(config, b"generated validation image")
    run_embedding_pipeline(config, RecordingMultimodalAdapter(), limit=1)
    with duckdb.connect(str(config.embedding_catalog)) as db:
        if damage == "vector":
            replacement = [0.0, 0.0, 1.0, *([0.0] * 2045)]
            replacement_hash = hashlib.sha256(
                b"".join(struct.pack("<f", value) for value in replacement)
            ).hexdigest()
            db.execute(
                """
                UPDATE embeddings
                SET vector = ?, stored_vector_sha256 = ?
                WHERE modality = 'multimodal_article'
                """,
                [replacement, replacement_hash],
            )
        else:
            db.execute(
                """
                UPDATE multimodal_embedding_provenance
                SET text_stored_vector_sha256 = ?
                """,
                [hashlib.sha256(b"wrong source link").hexdigest()],
            )

    codes = {issue.code for issue in validate_embedding_outputs(config).issues}

    assert expected_code in codes


def test_run_coverage_cannot_use_another_runs_generation_to_mask_deletion(
    tmp_path,
):
    """Break caught: configuration totals hide one generating run's loss."""

    config = write_generated_preprocessing_fixture(tmp_path)
    run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)
    add_generated_embedding_article(
        config,
        article_id="wsj:ZZZ-SYNTHETIC-EMBEDDING",
        markdown="# Second generation\n",
    )
    run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=2)
    with duckdb.connect(str(config.embedding_catalog)) as db:
        assert db.execute(
            "SELECT embeddings FROM runs ORDER BY started_at"
        ).fetchall() == [(1,), (1,)]
        db.execute(
            """
            DELETE FROM embeddings
            WHERE article_id = 'wsj:SYNTHETIC-EMBEDDING'
              AND modality = 'article_text'
            """
        )
        db.execute(
            """
            DELETE FROM embedding_work_items
            WHERE article_id = 'wsj:SYNTHETIC-EMBEDDING'
              AND modality = 'article_text'
            """
        )

    codes = {issue.code for issue in validate_embedding_outputs(config).issues}

    assert "missing_published_embedding" in codes


def test_validator_requires_provenance_for_archived_composite_generation(tmp_path):
    """Break caught: deleting archived composite provenance is invisible."""

    config = write_generated_preprocessing_fixture(tmp_path)
    relative_path = attach_generated_header_image(config, b"first source image")
    run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        archived_generation = db.execute(
            """
            SELECT generation_run_id
            FROM embedding_work_items
            WHERE modality = 'multimodal_article'
            """
        ).fetchone()[0]
    (config.source_root / relative_path).write_bytes(b"second source image")
    run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)
    with duckdb.connect(str(config.embedding_catalog)) as db:
        db.execute(
            """
            DELETE FROM multimodal_embedding_provenance
            WHERE generation_run_id = ?
            """,
            [archived_generation],
        )

    codes = {issue.code for issue in validate_embedding_outputs(config).issues}

    assert "missing_multimodal_provenance" in codes


def test_validator_recomputes_archived_composite_input_identity(tmp_path):
    """Break caught: archived provenance can retarget a same-hash generation."""

    config = write_generated_preprocessing_fixture(tmp_path)
    attach_generated_header_image(config, b"generated identity image")
    run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        archived_composite_generation = db.execute(
            """
            SELECT generation_run_id
            FROM embedding_work_items
            WHERE modality = 'multimodal_article'
            """
        ).fetchone()[0]
        original_text_hash = db.execute(
            """
            SELECT stored_vector_sha256
            FROM embeddings
            WHERE modality = 'article_text'
            """
        ).fetchone()[0]
    run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1, reprocess=True)
    with duckdb.connect(str(config.embedding_catalog)) as db:
        active_text_generation, active_text_hash = db.execute(
            """
            SELECT w.generation_run_id, e.stored_vector_sha256
            FROM embedding_work_items AS w
            JOIN embeddings AS e USING (article_id, modality, configuration_id)
            WHERE w.modality = 'article_text'
            """
        ).fetchone()
        assert active_text_hash == original_text_hash
        db.execute(
            """
            UPDATE multimodal_embedding_provenance
            SET text_generation_run_id = ?
            WHERE generation_run_id = ?
            """,
            [active_text_generation, archived_composite_generation],
        )

    codes = {issue.code for issue in validate_embedding_outputs(config).issues}

    assert "multimodal_input_identity_mismatch" in codes


def test_missing_source_vector_invalidates_composite_before_failed_regeneration(
    tmp_path,
):
    """Break caught: source recovery failure leaves the old composite active."""

    config = write_generated_preprocessing_fixture(tmp_path)
    attach_generated_header_image(config, b"generated missing source vector")
    run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)
    with duckdb.connect(str(config.embedding_catalog)) as db:
        db.execute("DELETE FROM embeddings WHERE modality = 'header_image'")

    class FailingImageAdapter(RecordingMultimodalAdapter):
        def embed_image(self, image_base64: str) -> tuple[float, ...]:
            raise JinaHostedAdapterError("rate_limit", retryable=True)

    result = run_embedding_pipeline(config, FailingImageAdapter(), limit=1)

    assert result.retryable == 1
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute(
            "SELECT modality FROM embeddings ORDER BY modality"
        ).fetchall() == [("article_text",)]
        assert db.execute(
            """
            SELECT modality, superseded_reason
            FROM embedding_generation_history
            WHERE modality = 'multimodal_article'
            """
        ).fetchone() == ("multimodal_article", "input_changed")


def test_validator_reports_antipodal_multimodal_recomputation_failure(tmp_path):
    """Break caught: an impossible normalized midpoint is silently skipped."""

    config = write_generated_preprocessing_fixture(tmp_path)
    attach_generated_header_image(config, b"generated antipodal image")
    run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)
    antipodal = [-1.0, *([0.0] * 2047)]
    antipodal_hash = hashlib.sha256(
        b"".join(struct.pack("<f", value) for value in antipodal)
    ).hexdigest()
    with duckdb.connect(str(config.embedding_catalog)) as db:
        (
            formula_version,
            text_generation,
            text_hash,
            image_generation,
        ) = db.execute(
            """
            SELECT p.formula_version, p.text_generation_run_id,
                   p.text_stored_vector_sha256,
                   p.header_image_generation_run_id
            FROM multimodal_embedding_provenance AS p
            """
        ).fetchone()
        identity = hashlib.sha256(
            json.dumps(
                {
                    "formula_version": formula_version,
                    "header_image_generation_run_id": image_generation,
                    "header_image_stored_vector_sha256": antipodal_hash,
                    "text_generation_run_id": text_generation,
                    "text_stored_vector_sha256": text_hash,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        db.execute(
            """
            UPDATE embeddings
            SET vector = ?, stored_vector_sha256 = ?
            WHERE modality = 'header_image'
            """,
            [antipodal, antipodal_hash],
        )
        db.execute(
            """
            UPDATE multimodal_embedding_provenance
            SET header_image_stored_vector_sha256 = ?
            """,
            [antipodal_hash],
        )
        db.execute(
            """
            UPDATE embeddings
            SET input_sha256 = ?
            WHERE modality = 'multimodal_article'
            """,
            [identity],
        )
        db.execute(
            """
            UPDATE embedding_work_items
            SET input_sha256 = ?
            WHERE modality = 'multimodal_article'
            """,
            [identity],
        )

    codes = {issue.code for issue in validate_embedding_outputs(config).issues}

    assert "multimodal_recomputation_failed" in codes


def test_absent_then_added_header_image_does_not_regenerate_successful_text(tmp_path):
    """Break caught: absent image is implicit or later image work repurchases text."""

    config = write_generated_preprocessing_fixture(tmp_path)
    first_adapter = RecordingMultimodalAdapter()

    first = run_embedding_pipeline(config, first_adapter, limit=1)

    assert first == EmbeddingRunResult(
        articles=1,
        embeddings=1,
        attempted=1,
        succeeded=1,
        header_absent=1,
    )
    assert first_adapter.calls == 1
    assert first_adapter.image_calls == 0
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute(
            """
            SELECT modality, source_relative_path, input_sha256, state,
                   attempt_count
            FROM embedding_work_items
            ORDER BY modality
            """
        ).fetchall() == [
            (
                "article_text",
                None,
                hashlib.sha256(b"#\n").hexdigest(),
                "succeeded",
                1,
            ),
            ("header_image", None, None, "not_applicable", 0),
        ]

    image_bytes = b"generated-later-image"
    relative_path = attach_generated_header_image(config, image_bytes)
    second_adapter = RecordingMultimodalAdapter()

    second = run_embedding_pipeline(config, second_adapter, limit=1)

    assert second == EmbeddingRunResult(
        articles=1,
        embeddings=2,
        reused=1,
        attempted=1,
        succeeded=2,
    )
    assert second_adapter.calls == 0
    assert second_adapter.image_calls == 1
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute(
            """
            SELECT source_relative_path, input_sha256, state, attempt_count
            FROM embedding_work_items
            WHERE modality = 'header_image'
            """
        ).fetchone() == (
            relative_path,
            hashlib.sha256(image_bytes).hexdigest(),
            "succeeded",
            1,
        )


def test_missing_header_is_retryable_and_recovers_without_regenerating_text(tmp_path):
    """Break caught: a missing optional image aborts text or cannot recover alone."""

    config = write_generated_preprocessing_fixture(tmp_path)
    relative_path = declare_missing_generated_header_image(config)
    first_adapter = RecordingMultimodalAdapter()

    first = run_embedding_pipeline(config, first_adapter, limit=1)

    assert first == EmbeddingRunResult(
        articles=1,
        embeddings=1,
        attempted=1,
        succeeded=1,
        retryable=1,
        header_failed=1,
    )
    assert first_adapter.calls == 1
    assert first_adapter.image_calls == 0
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute(
            """
            SELECT modality, source_relative_path, input_sha256, state,
                   attempt_count, error_code
            FROM embedding_work_items
            ORDER BY modality
            """
        ).fetchall() == [
            ("article_text", None, hashlib.sha256(b"#\n").hexdigest(),
             "succeeded", 1, None),
            ("header_image", relative_path, None, "retryable", 0,
             "missing_header_image"),
        ]
        assert db.execute(
            "SELECT modality FROM embeddings ORDER BY modality"
        ).fetchall() == [("article_text",)]
    assert validate_embedding_outputs(config) == EmbeddingValidationResult(
        issues=(
            EmbeddingValidationIssue(
                "missing_header_image",
                "canonical header image is unavailable",
            ),
        )
    )

    recovered_bytes = b"generated recovered header image"
    recovered_path = config.source_root / relative_path
    recovered_path.parent.mkdir(parents=True)
    recovered_path.write_bytes(recovered_bytes)
    recovered_adapter = RecordingMultimodalAdapter()

    recovered = run_embedding_pipeline(config, recovered_adapter, limit=1)

    assert recovered == EmbeddingRunResult(
        articles=1,
        embeddings=2,
        reused=1,
        attempted=1,
        succeeded=2,
    )
    assert recovered_adapter.calls == 0
    assert recovered_adapter.image_calls == 1
    assert validate_embedding_outputs(config) == EmbeddingValidationResult(issues=())


def test_retryable_header_failure_recovers_without_regenerating_text(tmp_path):
    """Break caught: one image transport failure aborts or repurchases text."""

    config = write_generated_preprocessing_fixture(tmp_path)
    attach_generated_header_image(config, b"generated retryable image")

    class RetryableImageAdapter(RecordingMultimodalAdapter):
        def embed_image(self, image_base64: str) -> tuple[float, ...]:
            self.image_calls += 1
            self.encoded_images.append(image_base64)
            raise JinaHostedAdapterError(
                "rate_limit",
                retryable=True,
                status_code=429,
                retry_after_seconds=2.0,
            )

    first_adapter = RetryableImageAdapter()
    first = run_embedding_pipeline(config, first_adapter, limit=1)

    assert first == EmbeddingRunResult(
        articles=1,
        embeddings=1,
        attempted=2,
        succeeded=1,
        retryable=1,
        header_failed=1,
    )
    assert first_adapter.calls == 1
    assert first_adapter.image_calls == 1
    assert {issue.code for issue in validate_embedding_outputs(config).issues} == {
        "header_image_retryable_failure"
    }

    recovered_adapter = RecordingMultimodalAdapter()
    recovered = run_embedding_pipeline(config, recovered_adapter, limit=1)

    assert recovered == EmbeddingRunResult(
        articles=1,
        embeddings=2,
        reused=1,
        attempted=1,
        succeeded=2,
    )
    assert recovered_adapter.calls == 0
    assert recovered_adapter.image_calls == 1
    assert validate_embedding_outputs(config).ok

    unchanged_adapter = RecordingMultimodalAdapter()
    unchanged = run_embedding_pipeline(config, unchanged_adapter, limit=1)

    assert unchanged.reused == 3
    assert unchanged_adapter.calls == 0
    assert unchanged_adapter.image_calls == 0


def test_oversized_header_image_publishes_verified_versioned_rendition(tmp_path):
    """Break caught: oversized source bytes are sent or rendition identity is lost."""

    config = write_generated_preprocessing_fixture(tmp_path)
    source_bytes = b"generated oversized png"
    derived_bytes = b"generated deterministic jpeg rendition"
    attach_generated_header_image(config, source_bytes)
    before = snapshot_fixture_inputs(config)
    codec = ScenarioImageCodec(
        {
            source_bytes: ImageInfo("PNG", 6000, 4000),
            derived_bytes: ImageInfo("JPEG", 5477, 3651),
        }
    )
    adapter = ScenarioImageAdapter()

    run_embedding_pipeline(
        config,
        adapter,
        limit=1,
        image_codec=codec,
    )

    assert codec.rendered == [(source_bytes, 5477, 3651)]
    assert adapter.encoded_images == [base64.b64encode(derived_bytes).decode("ascii")]
    derived_sha256 = hashlib.sha256(derived_bytes).hexdigest()
    transform_namespace = hashlib.sha256(codec.transform_id.encode()).hexdigest()
    expected_relative_path = (
        f"renditions/{transform_namespace}/{derived_sha256}.jpg"
    )
    rendition_path = config.embedding_output_root / expected_relative_path
    assert rendition_path.read_bytes() == derived_bytes
    assert not list(config.embedding_output_root.rglob("*.tmp"))
    assert snapshot_fixture_inputs(config) == before
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute(
            """
            SELECT source_sha256, source_format, source_bytes,
                   source_width, source_height, source_frames,
                   embedded_input_sha256,
                   embedded_format, embedded_bytes, embedded_width,
                   embedded_height, embedded_frames, transform_id,
                   rendition_relative_path
            FROM image_input_provenance
            """
        ).fetchone() == (
            hashlib.sha256(source_bytes).hexdigest(),
            "PNG",
            len(source_bytes),
            6000,
            4000,
            1,
            derived_sha256,
            "JPEG",
            len(derived_bytes),
            5477,
            3651,
            1,
            codec.transform_id,
            expected_relative_path,
        )
    assert validate_embedding_outputs(config, image_codec=codec).ok


@pytest.mark.parametrize(
    ("source_outcome", "derived_outcome", "expected_code"),
    (
        (ImageInfo("TIFF", 100, 100), None, "unsupported_image"),
        (ImageCodecError("corrupt_image"), None, "corrupt_image"),
        (
            ImageInfo("PNG", 6000, 4000),
            ImageInfo("JPEG", 6000, 4000),
            "derived_image_oversized",
        ),
    ),
    ids=("unsupported", "corrupt", "still-oversized"),
)
def test_ineligible_header_image_is_terminal_without_placeholder(
    tmp_path,
    source_outcome,
    derived_outcome,
    expected_code,
):
    """Break caught: unsafe local image input escapes or publishes a zero vector."""

    config = write_generated_preprocessing_fixture(tmp_path)
    source_bytes = b"generated ineligible image"
    derived_bytes = b"generated deterministic jpeg rendition"
    attach_generated_header_image(config, source_bytes)
    before = snapshot_fixture_inputs(config)
    scenarios = {source_bytes: source_outcome}
    if derived_outcome is not None:
        scenarios[derived_bytes] = derived_outcome
    codec = ScenarioImageCodec(scenarios)
    adapter = ScenarioImageAdapter()

    result = run_embedding_pipeline(
        config,
        adapter,
        limit=1,
        image_codec=codec,
    )

    assert result == EmbeddingRunResult(
        articles=1,
        embeddings=1,
        attempted=1,
        succeeded=1,
        terminal=1,
        header_failed=1,
    )
    assert adapter.image_calls == 0
    assert snapshot_fixture_inputs(config) == before
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute(
            """
            SELECT state, attempt_count, error_code, input_sha256
            FROM embedding_work_items
            WHERE modality = 'header_image'
            """
        ).fetchone() == (
            "terminal",
            0,
            expected_code,
            hashlib.sha256(source_bytes).hexdigest(),
        )
        assert db.execute(
            "SELECT modality FROM embeddings ORDER BY modality"
        ).fetchall() == [("article_text",)]
        assert db.execute("SELECT count(*) FROM image_input_provenance").fetchone() == (
            0,
        )
    assert {issue.code for issue in validate_embedding_outputs(config).issues} == {
        "header_image_terminal_failure"
    }


def test_rendition_install_interruption_precedes_embedding_state_commit(tmp_path):
    """Break caught: image state commits before the exact rendition is durable."""

    config = write_generated_preprocessing_fixture(tmp_path)
    source_bytes = b"generated interrupted oversized png"
    derived_bytes = b"generated deterministic jpeg rendition"
    attach_generated_header_image(config, source_bytes)
    before = snapshot_fixture_inputs(config)
    codec = ScenarioImageCodec(
        {
            source_bytes: ImageInfo("PNG", 6000, 4000),
            derived_bytes: ImageInfo("JPEG", 5477, 3651),
        }
    )
    adapter = ScenarioImageAdapter()
    observed_paths: list[str] = []

    def interrupt(relative_path: str) -> None:
        observed_paths.append(relative_path)
        raise SyntheticInterruption

    failpoints = EmbeddingPipelineFailpoints(
        after_image_rendition_install=interrupt,
    )
    with pytest.raises(SyntheticInterruption):
        run_embedding_pipeline(
            config,
            adapter,
            limit=1,
            image_codec=codec,
            failpoints=failpoints,
        )

    assert len(observed_paths) == 1
    assert (config.embedding_output_root / observed_paths[0]).read_bytes() == (
        derived_bytes
    )
    assert adapter.image_calls == 0
    assert snapshot_fixture_inputs(config) == before
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute("SELECT count(*) FROM runs").fetchone() == (0,)
        assert db.execute("SELECT count(*) FROM embedding_work_items").fetchone() == (
            0,
        )

    replay_adapter = ScenarioImageAdapter()
    replay = run_embedding_pipeline(
        config,
        replay_adapter,
        limit=1,
        image_codec=codec,
    )

    assert replay.embeddings == 3
    assert replay_adapter.encoded_images == [
        base64.b64encode(derived_bytes).decode("ascii")
    ]
    assert snapshot_fixture_inputs(config) == before


def test_rendition_directory_entries_are_fsynced_before_nested_publication(
    tmp_path,
    monkeypatch,
):
    """Break caught: a crash loses newly created rendition directories."""

    config = write_generated_preprocessing_fixture(tmp_path)
    source_bytes = b"generated fsync oversized png"
    derived_bytes = b"generated deterministic jpeg rendition"
    attach_generated_header_image(config, source_bytes)
    codec = ScenarioImageCodec(
        {
            source_bytes: ImageInfo("PNG", 6000, 4000),
            derived_bytes: ImageInfo("JPEG", 5477, 3651),
        }
    )
    events: list[tuple[str, str | int, int]] = []
    real_mkdir = os.mkdir
    real_fsync = os.fsync
    real_link = os.link

    def observed_mkdir(path, mode=0o777, *, dir_fd=None):
        result = real_mkdir(path, mode=mode, dir_fd=dir_fd)
        if dir_fd is not None:
            events.append(("mkdir", str(path), os.fstat(dir_fd).st_ino))
        return result

    def observed_fsync(descriptor):
        descriptor_stat = os.fstat(descriptor)
        if stat.S_ISDIR(descriptor_stat.st_mode):
            events.append(("fsync", descriptor_stat.st_ino, descriptor_stat.st_ino))
        return real_fsync(descriptor)

    def observed_link(
        source,
        destination,
        *,
        src_dir_fd=None,
        dst_dir_fd=None,
        follow_symlinks=True,
    ):
        assert dst_dir_fd is not None
        events.append(("link", str(destination), os.fstat(dst_dir_fd).st_ino))
        return real_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(image_rendition_module.os, "mkdir", observed_mkdir)
    monkeypatch.setattr(image_rendition_module.os, "fsync", observed_fsync)
    monkeypatch.setattr(image_rendition_module.os, "link", observed_link)

    run_embedding_pipeline(
        config,
        ScenarioImageAdapter(),
        limit=1,
        image_codec=codec,
    )

    output_inode = config.embedding_output_root.stat().st_ino
    rendition_inode = (config.embedding_output_root / "renditions").stat().st_ino
    transform_namespace = hashlib.sha256(codec.transform_id.encode()).hexdigest()
    transform_inode = (
        config.embedding_output_root / "renditions" / transform_namespace
    ).stat().st_ino
    derived_sha256 = hashlib.sha256(derived_bytes).hexdigest()
    relevant = [
        event
        for event in events
        if event
        in {
            ("mkdir", "renditions", output_inode),
            ("fsync", output_inode, output_inode),
            ("mkdir", transform_namespace, rendition_inode),
            ("fsync", rendition_inode, rendition_inode),
            ("link", f"{derived_sha256}.jpg", transform_inode),
            ("fsync", transform_inode, transform_inode),
        }
    ]
    assert relevant == [
        ("mkdir", "renditions", output_inode),
        ("fsync", output_inode, output_inode),
        ("mkdir", transform_namespace, rendition_inode),
        ("fsync", rendition_inode, rendition_inode),
        ("link", f"{derived_sha256}.jpg", transform_inode),
        ("fsync", transform_inode, transform_inode),
    ]


def test_rendition_directory_reuse_fsyncs_verified_parents_before_state(
    tmp_path,
    monkeypatch,
):
    """Break caught: replay trusts a prior mkdir whose parent was never durable."""

    config = write_generated_preprocessing_fixture(tmp_path)
    source_bytes = b"generated replay-fsync oversized png"
    derived_bytes = b"generated deterministic jpeg rendition"
    attach_generated_header_image(config, source_bytes)
    codec = ScenarioImageCodec(
        {
            source_bytes: ImageInfo("PNG", 6000, 4000),
            derived_bytes: ImageInfo("JPEG", 5477, 3651),
        }
    )
    with pytest.raises(SyntheticInterruption):
        run_embedding_pipeline(
            config,
            ScenarioImageAdapter(),
            limit=1,
            image_codec=codec,
            failpoints=EmbeddingPipelineFailpoints(
                after_image_rendition_install=lambda path: (_ for _ in ()).throw(
                    SyntheticInterruption
                )
            ),
        )
    events: list[tuple[str, int]] = []
    real_fsync = os.fsync
    real_open = os.open

    def observed_fsync(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            events.append(("fsync", os.fstat(descriptor).st_ino))
        return real_fsync(descriptor)

    def observed_open(path, flags, mode=0o777, *, dir_fd=None):
        if dir_fd is not None and path not in {"catalog.duckdb", "pipeline.lock"}:
            events.append((f"open:{path}", os.fstat(dir_fd).st_ino))
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(image_rendition_module.os, "fsync", observed_fsync)
    monkeypatch.setattr(image_rendition_module.os, "open", observed_open)

    result = run_embedding_pipeline(
        config,
        ScenarioImageAdapter(),
        limit=1,
        image_codec=codec,
    )

    output_inode = config.embedding_output_root.stat().st_ino
    rendition_inode = (config.embedding_output_root / "renditions").stat().st_ino
    transform_namespace = hashlib.sha256(codec.transform_id.encode()).hexdigest()
    assert events.index(("fsync", output_inode)) < events.index(
        (f"open:{transform_namespace}", rendition_inode)
    )
    assert events.index(("fsync", rendition_inode)) < next(
        index
        for index, event in enumerate(events)
        if event[0].startswith("open:.")
    )
    assert result.embeddings == 3


@pytest.mark.parametrize("replacement_kind", ("replacement", "hardlink", "symlink"))
def test_rendition_final_leaf_replacement_is_refused_before_state_commit(
    tmp_path,
    replacement_kind,
):
    """Break caught: final path identity changes after its verified descriptor read."""

    config = write_generated_preprocessing_fixture(tmp_path)
    source_bytes = b"generated raced oversized png"
    derived_bytes = b"generated deterministic jpeg rendition"
    attach_generated_header_image(config, source_bytes)
    transform_id = ScenarioImageCodec.transform_id
    transform_namespace = hashlib.sha256(transform_id.encode()).hexdigest()
    derived_sha256 = hashlib.sha256(derived_bytes).hexdigest()
    final_path = (
        config.embedding_output_root
        / "renditions"
        / transform_namespace
        / f"{derived_sha256}.jpg"
    )

    class ReplacingCodec(ScenarioImageCodec):
        def __init__(self):
            super().__init__(
                {
                    source_bytes: ImageInfo("PNG", 6000, 4000),
                    derived_bytes: ImageInfo("JPEG", 5477, 3651),
                }
            )
            self.derived_inspections = 0

        def inspect(self, data: bytes, *, max_decode_pixels: int) -> ImageInfo:
            result = super().inspect(data, max_decode_pixels=max_decode_pixels)
            if data == derived_bytes:
                self.derived_inspections += 1
                if self.derived_inspections == 3:
                    sibling = final_path.with_suffix(f".{replacement_kind}")
                    if replacement_kind == "hardlink":
                        os.link(final_path, sibling)
                    else:
                        final_path.replace(sibling)
                        if replacement_kind == "symlink":
                            final_path.symlink_to(sibling.name)
                        else:
                            final_path.write_bytes(derived_bytes)
            return result

    codec = ReplacingCodec()

    with pytest.raises(EmbeddingPipelineError) as raised:
        run_embedding_pipeline(
            config,
            ScenarioImageAdapter(),
            limit=1,
            image_codec=codec,
        )

    assert raised.value.code == "unsafe_rendition_output"
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute("SELECT count(*) FROM runs").fetchone() == (0,)
        assert db.execute("SELECT count(*) FROM embedding_work_items").fetchone() == (
            0,
        )


@pytest.mark.parametrize("ancestor", ("renditions", "transform"))
def test_rendition_ancestor_replacement_is_refused_before_state_commit(
    tmp_path,
    ancestor,
):
    """Break caught: held leaf succeeds after its namespace path was replaced."""

    config = write_generated_preprocessing_fixture(tmp_path)
    source_bytes = b"generated ancestor-race oversized png"
    derived_bytes = b"generated deterministic jpeg rendition"
    attach_generated_header_image(config, source_bytes)
    namespace = hashlib.sha256(ScenarioImageCodec.transform_id.encode()).hexdigest()
    rendition_path = config.embedding_output_root / "renditions"
    transform_path = rendition_path / namespace

    class AncestorReplacingCodec(ScenarioImageCodec):
        def __init__(self):
            super().__init__(
                {
                    source_bytes: ImageInfo("PNG", 6000, 4000),
                    derived_bytes: ImageInfo("JPEG", 5477, 3651),
                }
            )
            self.derived_inspections = 0

        def inspect(self, data: bytes, *, max_decode_pixels: int) -> ImageInfo:
            result = super().inspect(data, max_decode_pixels=max_decode_pixels)
            if data == derived_bytes:
                self.derived_inspections += 1
                if self.derived_inspections == 3:
                    target = (
                        rendition_path
                        if ancestor == "renditions"
                        else transform_path
                    )
                    target.replace(target.with_name(f"{target.name}.replaced"))
                    target.mkdir(parents=True)
            return result

    with pytest.raises(EmbeddingPipelineError) as raised:
        run_embedding_pipeline(
            config,
            ScenarioImageAdapter(),
            limit=1,
            image_codec=AncestorReplacingCodec(),
        )

    assert raised.value.code == "unsafe_rendition_output"
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute("SELECT count(*) FROM runs").fetchone() == (0,)


def test_rendition_ancestor_replacement_during_reanchored_decode_is_refused(
    tmp_path,
):
    """Break caught: namespace replacement during final decode escapes reanchor."""

    config = write_generated_preprocessing_fixture(tmp_path)
    source_bytes = b"generated decode-race oversized png"
    derived_bytes = b"generated deterministic jpeg rendition"
    relative_path = attach_generated_header_image(config, source_bytes)
    namespace = hashlib.sha256(ScenarioImageCodec.transform_id.encode()).hexdigest()
    transform_path = config.embedding_output_root / "renditions" / namespace

    class DecodeReplacingCodec(ScenarioImageCodec):
        def __init__(self):
            super().__init__(
                {
                    source_bytes: ImageInfo("PNG", 6000, 4000),
                    derived_bytes: ImageInfo("JPEG", 5477, 3651),
                }
            )
            self.derived_inspections = 0

        def inspect(self, data: bytes, *, max_decode_pixels: int) -> ImageInfo:
            result = super().inspect(data, max_decode_pixels=max_decode_pixels)
            if data == derived_bytes:
                self.derived_inspections += 1
                if self.derived_inspections == 4:
                    transform_path.replace(transform_path.with_suffix(".replaced"))
                    transform_path.mkdir()
            return result

    with pytest.raises(EmbeddingPipelineError) as raised:
        run_embedding_pipeline(
            config,
            ScenarioImageAdapter(),
            limit=1,
            image_codec=DecodeReplacingCodec(),
        )

    assert raised.value.code == "unsafe_rendition_output"
    assert (config.source_root / relative_path).read_bytes() == source_bytes
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute("SELECT count(*) FROM runs").fetchone() == (0,)


def test_rendition_chain_is_reverified_at_publication_boundary(tmp_path):
    """Break caught: namespace changes after preparation but before begin_run."""

    config = write_generated_preprocessing_fixture(tmp_path)
    source_bytes = b"generated boundary-race oversized png"
    derived_bytes = b"generated deterministic jpeg rendition"
    attach_generated_header_image(config, source_bytes)
    codec = ScenarioImageCodec(
        {
            source_bytes: ImageInfo("PNG", 6000, 4000),
            derived_bytes: ImageInfo("JPEG", 5477, 3651),
        }
    )

    def replace_transform(relative_path: str) -> None:
        transform_path = (config.embedding_output_root / relative_path).parent
        transform_path.replace(transform_path.with_suffix(".replaced"))
        transform_path.mkdir()

    with pytest.raises(EmbeddingPipelineError) as raised:
        run_embedding_pipeline(
            config,
            ScenarioImageAdapter(),
            limit=1,
            image_codec=codec,
            failpoints=EmbeddingPipelineFailpoints(
                after_image_rendition_install=replace_transform,
            ),
        )

    assert raised.value.code == "unsafe_rendition_output"
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute("SELECT count(*) FROM runs").fetchone() == (0,)
        assert db.execute("SELECT count(*) FROM embedding_work_items").fetchone() == (
            0,
        )


def test_validator_requires_exact_image_provenance_and_rendition_bytes(tmp_path):
    """Break caught: image generations survive missing or changed embedded input."""

    config = write_generated_preprocessing_fixture(tmp_path)
    source_bytes = b"generated validation oversized png"
    derived_bytes = b"generated deterministic jpeg rendition"
    attach_generated_header_image(config, source_bytes)
    codec = ScenarioImageCodec(
        {
            source_bytes: ImageInfo("PNG", 6000, 4000),
            derived_bytes: ImageInfo("JPEG", 5477, 3651),
        }
    )
    run_embedding_pipeline(
        config,
        ScenarioImageAdapter(),
        limit=1,
        image_codec=codec,
    )
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        relative_path = db.execute(
            "SELECT rendition_relative_path FROM image_input_provenance"
        ).fetchone()[0]
    rendition_path = config.embedding_output_root / relative_path
    rendition_path.write_bytes(b"changed generated rendition")

    codes = {issue.code for issue in validate_embedding_outputs(config).issues}

    assert "rendition_hash_mismatch" in codes

    rendition_path.write_bytes(derived_bytes)
    with duckdb.connect(str(config.embedding_catalog)) as db:
        db.execute("DELETE FROM image_input_provenance")

    codes = {issue.code for issue in validate_embedding_outputs(config).issues}

    assert "missing_image_input_provenance" in codes


def test_validator_decodes_source_bytes_to_verify_declared_image_facts(tmp_path):
    """Break caught: self-consistent false image facts pass read-only validation."""

    config = write_generated_preprocessing_fixture(tmp_path)
    source_bytes = b"generated decoded pass-through png"
    attach_generated_header_image(config, source_bytes)
    codec = ScenarioImageCodec({source_bytes: ImageInfo("PNG", 640, 360)})
    run_embedding_pipeline(
        config,
        ScenarioImageAdapter(),
        limit=1,
        image_codec=codec,
    )
    with duckdb.connect(str(config.embedding_catalog)) as db:
        db.execute(
            """
            UPDATE image_input_provenance
            SET source_format = 'JPEG', embedded_format = 'JPEG'
            """
        )

    codes = {
        issue.code
        for issue in validate_embedding_outputs(config, image_codec=codec).issues
    }

    assert "invalid_image_input_provenance" in codes


def test_validator_fails_closed_without_matching_image_decoder(tmp_path):
    """Break caught: provenance is accepted without a decoder for its profile."""

    config = write_generated_preprocessing_fixture(tmp_path)
    source_bytes = b"generated decoder-required png"
    attach_generated_header_image(config, source_bytes)
    codec = ScenarioImageCodec({source_bytes: ImageInfo("PNG", 640, 360)})
    run_embedding_pipeline(
        config,
        ScenarioImageAdapter(),
        limit=1,
        image_codec=codec,
    )

    codes = {issue.code for issue in validate_embedding_outputs(config).issues}

    assert "image_codec_unavailable" in codes


@pytest.mark.parametrize(
    "mutation",
    (
        "UPDATE image_input_provenance SET embedded_format = 'PNG'",
        """
        UPDATE image_input_provenance
        SET source_width = 640, source_height = 360
        """,
    ),
    ids=("derived-must-be-jpeg", "declared-within-limit-must-pass-through"),
)
def test_validator_rejects_malformed_derived_image_contract(tmp_path, mutation):
    """Break caught: a derived row can contradict the pinned hosted contract."""

    config = write_generated_preprocessing_fixture(tmp_path)
    source_bytes = b"generated malformed derived png"
    derived_bytes = b"generated deterministic jpeg rendition"
    attach_generated_header_image(config, source_bytes)
    codec = ScenarioImageCodec(
        {
            source_bytes: ImageInfo("PNG", 6000, 4000),
            derived_bytes: ImageInfo("JPEG", 5477, 3651),
        }
    )
    run_embedding_pipeline(
        config,
        ScenarioImageAdapter(),
        limit=1,
        image_codec=codec,
    )
    with duckdb.connect(str(config.embedding_catalog)) as db:
        db.execute(mutation)

    codes = {
        issue.code
        for issue in validate_embedding_outputs(config, image_codec=codec).issues
    }

    assert "invalid_image_input_provenance" in codes


@pytest.mark.parametrize("frame_column", ("source_frames", "embedded_frames"))
def test_validator_rejects_animated_image_provenance(tmp_path, frame_column):
    """Break caught: an animated source or embedded input is declared static."""

    config = write_generated_preprocessing_fixture(tmp_path)
    source_bytes = b"generated animated-policy oversized png"
    derived_bytes = b"generated deterministic jpeg rendition"
    attach_generated_header_image(config, source_bytes)
    codec = ScenarioImageCodec(
        {
            source_bytes: ImageInfo("PNG", 6000, 4000),
            derived_bytes: ImageInfo("JPEG", 5477, 3651),
        }
    )
    run_embedding_pipeline(
        config,
        ScenarioImageAdapter(),
        limit=1,
        image_codec=codec,
    )
    with duckdb.connect(str(config.embedding_catalog)) as db:
        db.execute(f"UPDATE image_input_provenance SET {frame_column} = 2")

    codes = {
        issue.code
        for issue in validate_embedding_outputs(config, image_codec=codec).issues
    }

    assert "invalid_image_input_provenance" in codes


def test_validator_rejects_arbitrary_within_limit_rendition_dimensions(tmp_path):
    """Break caught: a derived JPEG ignores the deterministic aspect scale."""

    config = write_generated_preprocessing_fixture(tmp_path)
    source_bytes = b"generated arbitrary-dimension oversized png"
    derived_bytes = b"generated deterministic jpeg rendition"
    attach_generated_header_image(config, source_bytes)
    codec = ScenarioImageCodec(
        {
            source_bytes: ImageInfo("PNG", 6000, 4000),
            derived_bytes: ImageInfo("JPEG", 5000, 4000),
        }
    )
    run_embedding_pipeline(
        config,
        ScenarioImageAdapter(),
        limit=1,
        image_codec=codec,
    )

    codes = {
        issue.code
        for issue in validate_embedding_outputs(config, image_codec=codec).issues
    }

    assert "invalid_image_input_provenance" in codes


def test_validator_rejects_archived_within_limit_source_with_rendition(tmp_path):
    """Break caught: archived derived provenance escapes pass-through policy."""

    config = write_generated_preprocessing_fixture(tmp_path)
    first_source = b"generated archived oversized png one"
    second_source = b"generated archived oversized png two"
    derived_bytes = b"generated deterministic jpeg rendition"
    relative_path = attach_generated_header_image(config, first_source)
    codec = ScenarioImageCodec(
        {
            first_source: ImageInfo("PNG", 6000, 4000),
            second_source: ImageInfo("PNG", 6000, 4000),
            derived_bytes: ImageInfo("JPEG", 5477, 3651),
        }
    )
    run_embedding_pipeline(
        config,
        ScenarioImageAdapter(),
        limit=1,
        image_codec=codec,
    )
    (config.source_root / relative_path).write_bytes(second_source)
    run_embedding_pipeline(
        config,
        ScenarioImageAdapter(),
        limit=1,
        image_codec=codec,
    )
    with duckdb.connect(str(config.embedding_catalog)) as db:
        archived_generation = db.execute(
            """
            SELECT generation_run_id FROM embedding_generation_history
            WHERE modality = 'header_image'
            """
        ).fetchone()[0]
        db.execute(
            """
            UPDATE image_input_provenance
            SET source_bytes = 1000, source_width = 640, source_height = 360
            WHERE generation_run_id = ?
            """,
            [archived_generation],
        )

    codes = {
        issue.code
        for issue in validate_embedding_outputs(config, image_codec=codec).issues
    }

    assert "invalid_image_input_provenance" in codes


def test_validator_rejects_archived_source_above_safe_decode_ceiling(tmp_path):
    """Break caught: archived unsafe source dimensions retain valid provenance."""

    config = write_generated_preprocessing_fixture(tmp_path)
    first_source = b"generated archived unsafe pixels one"
    second_source = b"generated archived unsafe pixels two"
    derived_bytes = b"generated deterministic jpeg rendition"
    relative_path = attach_generated_header_image(config, first_source)
    codec = ScenarioImageCodec(
        {
            first_source: ImageInfo("PNG", 6000, 4000),
            second_source: ImageInfo("PNG", 6000, 4000),
            derived_bytes: ImageInfo("JPEG", 5477, 3651),
        }
    )
    run_embedding_pipeline(
        config,
        ScenarioImageAdapter(),
        limit=1,
        image_codec=codec,
    )
    (config.source_root / relative_path).write_bytes(second_source)
    run_embedding_pipeline(
        config,
        ScenarioImageAdapter(),
        limit=1,
        image_codec=codec,
    )
    with duckdb.connect(str(config.embedding_catalog)) as db:
        archived_generation = db.execute(
            """
            SELECT generation_run_id FROM embedding_generation_history
            WHERE modality = 'header_image'
            """
        ).fetchone()[0]
        db.execute(
            """
            UPDATE image_input_provenance
            SET source_bytes = 6000000,
                source_width = 10000, source_height = 5000,
                embedded_width = 6324, embedded_height = 3162
            WHERE generation_run_id = ?
            """,
            [archived_generation],
        )

    issues = validate_embedding_outputs(config, image_codec=codec).issues

    assert any(
        issue.code == "invalid_image_input_provenance"
        and "safe decode pixel ceiling" in issue.message
        for issue in issues
    )


def test_validator_reports_nonpositive_image_dimensions_without_aborting(tmp_path):
    """Break caught: malformed dimensions raise from policy scaling."""

    config = write_generated_preprocessing_fixture(tmp_path)
    source_bytes = b"generated malformed-dimension oversized png"
    derived_bytes = b"generated deterministic jpeg rendition"
    attach_generated_header_image(config, source_bytes)
    codec = ScenarioImageCodec(
        {
            source_bytes: ImageInfo("PNG", 6000, 4000),
            derived_bytes: ImageInfo("JPEG", 5477, 3651),
        }
    )
    run_embedding_pipeline(
        config,
        ScenarioImageAdapter(),
        limit=1,
        image_codec=codec,
    )
    with duckdb.connect(str(config.embedding_catalog)) as db:
        db.execute("UPDATE image_input_provenance SET source_width = 0")

    result = validate_embedding_outputs(config, image_codec=codec)

    assert "invalid_image_input_provenance" in {
        issue.code for issue in result.issues
    }


def test_validator_reports_image_provenance_with_unknown_configuration(tmp_path):
    """Break caught: unselected provenance can reference no known configuration."""

    config = write_generated_preprocessing_fixture(tmp_path)
    attach_generated_header_image(config, b"generated unknown configuration image")
    run_embedding_pipeline(config, RecordingMultimodalAdapter(), limit=1)
    with duckdb.connect(str(config.embedding_catalog)) as db:
        db.execute(
            """
            INSERT INTO image_input_provenance
            SELECT article_id, repeat('f', 64), generation_run_id,
                   source_sha256, source_format, source_bytes,
                   source_width, source_height, source_frames,
                   embedded_input_sha256,
                   embedded_format, embedded_bytes, embedded_width,
                   embedded_height, embedded_frames, transform_id,
                   rendition_relative_path,
                   created_at
            FROM image_input_provenance
            """
        )

    codes = {issue.code for issue in validate_embedding_outputs(config).issues}

    assert "unknown_image_provenance_configuration" in codes


def test_validator_scans_renditions_for_orphans_temps_and_symlinks(tmp_path):
    """Break caught: unreferenced or unsafe rendition output remains invisible."""

    config = write_generated_preprocessing_fixture(tmp_path)
    source_bytes = b"generated orphan scan oversized png"
    derived_bytes = b"generated deterministic jpeg rendition"
    attach_generated_header_image(config, source_bytes)
    codec = ScenarioImageCodec(
        {
            source_bytes: ImageInfo("PNG", 6000, 4000),
            derived_bytes: ImageInfo("JPEG", 5477, 3651),
        }
    )
    run_embedding_pipeline(
        config,
        ScenarioImageAdapter(),
        limit=1,
        image_codec=codec,
    )
    rendition_root = config.embedding_output_root / "renditions"
    namespace = next(path for path in rendition_root.iterdir() if path.is_dir())
    (namespace / "unreferenced.jpg").write_bytes(b"generated orphan")
    (namespace / ".interrupted.tmp").write_bytes(b"generated stage")
    outside = config.embedding_output_root / "generated-outside-rendition"
    outside.write_bytes(b"generated outside")
    (namespace / "unsafe-link.jpg").symlink_to(outside)

    codes = {
        issue.code
        for issue in validate_embedding_outputs(config, image_codec=codec).issues
    }

    assert {
        "orphan_image_rendition",
        "orphan_image_rendition_temp",
        "unsafe_image_rendition_entry",
    } <= codes


def test_changed_image_transform_creates_distinct_vectors_and_renditions(tmp_path):
    """Break caught: a transform change overwrites prior image-vector meaning."""

    config = write_generated_preprocessing_fixture(tmp_path)
    source_bytes = b"generated transform identity oversized png"
    derived_bytes = b"generated deterministic jpeg rendition"
    attach_generated_header_image(config, source_bytes)
    first_codec = ScenarioImageCodec(
        {
            source_bytes: ImageInfo("PNG", 6000, 4000),
            derived_bytes: ImageInfo("JPEG", 5477, 3651),
        }
    )
    first_adapter = ScenarioImageAdapter()
    run_embedding_pipeline(
        config,
        first_adapter,
        limit=1,
        image_codec=first_codec,
    )

    second_codec = ScenarioImageCodec(first_codec.scenarios)
    second_codec.transform_id = "synthetic-jpeg-rendition-v2"

    class SecondTransformAdapter(ScenarioImageAdapter):
        profile = replace(
            ScenarioImageAdapter.profile,
            image_transform=second_codec.transform_id,
        )

    second_adapter = SecondTransformAdapter()
    run_embedding_pipeline(
        config,
        second_adapter,
        limit=1,
        image_codec=second_codec,
    )

    first_configuration = configuration_id(first_adapter.profile)
    second_configuration = configuration_id(second_adapter.profile)
    assert first_configuration != second_configuration
    derived_sha256 = hashlib.sha256(derived_bytes).hexdigest()
    first_rendition_path = (
        "renditions/"
        f"{hashlib.sha256(first_codec.transform_id.encode()).hexdigest()}/"
        f"{derived_sha256}.jpg"
    )
    second_rendition_path = (
        "renditions/"
        f"{hashlib.sha256(second_codec.transform_id.encode()).hexdigest()}/"
        f"{derived_sha256}.jpg"
    )
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        rows = db.execute(
            """
            SELECT configuration_id, transform_id, rendition_relative_path
            FROM image_input_provenance
            ORDER BY configuration_id
            """
        ).fetchall()
        assert rows == sorted(
            [
                (
                    first_configuration,
                    first_codec.transform_id,
                    first_rendition_path,
                ),
                (
                    second_configuration,
                    second_codec.transform_id,
                    second_rendition_path,
                ),
            ]
        )
        assert db.execute(
            """
            SELECT configuration_id, count(*)
            FROM embeddings
            WHERE modality = 'header_image'
            GROUP BY configuration_id
            ORDER BY configuration_id
            """
        ).fetchall() == sorted(
            [(first_configuration, 1), (second_configuration, 1)]
        )
    assert len({row[2] for row in rows}) == 2
    assert all((config.embedding_output_root / row[2]).is_file() for row in rows)
    assert validate_embedding_outputs(
        config,
        configuration_id=first_configuration,
        image_codec=first_codec,
    ).ok
    assert validate_embedding_outputs(
        config,
        configuration_id=second_configuration,
        image_codec=second_codec,
    ).ok


def test_image_codec_profile_mismatch_fails_before_state_or_hosted_request(tmp_path):
    """Break caught: bytes are transformed under a different declared profile."""

    config = write_generated_preprocessing_fixture(tmp_path)
    source_bytes = b"generated ambiguous image profile"
    attach_generated_header_image(config, source_bytes)
    codec = ScenarioImageCodec({source_bytes: ImageInfo("PNG", 640, 360)})
    codec.transform_id = "undeclared-transform"
    adapter = ScenarioImageAdapter()

    with pytest.raises(EmbeddingPipelineError) as raised:
        run_embedding_pipeline(
            config,
            adapter,
            limit=1,
            image_codec=codec,
        )

    assert raised.value.code == "ambiguous_image_configuration"
    assert adapter.calls == 0
    assert adapter.image_calls == 0
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute("SELECT count(*) FROM runs").fetchone() == (0,)
        assert db.execute("SELECT count(*) FROM embedding_work_items").fetchone() == (
            0,
        )


@pytest.mark.parametrize("bomb_kind", ("error", "warning"))
def test_pillow_decompression_bomb_is_terminal_unsafe_image_without_vector(
    tmp_path,
    monkeypatch,
    bomb_kind,
):
    """Break caught: Pillow's bomb exception escapes and aborts the run."""

    def raise_bomb(stream):
        del stream
        image = sys.modules["PIL"].Image
        if bomb_kind == "warning":
            warnings.warn(
                "synthetic header bomb",
                image.DecompressionBombWarning,
                stacklevel=2,
            )
            raise AssertionError("bomb warning was not promoted to an exception")
        raise image.DecompressionBombError

    install_synthetic_pillow(
        monkeypatch,
        jpeg_version="9e",
        turbo_version="3.1.0",
        open_image=raise_bomb,
    )
    codec = PillowImageCodec()

    class BombAdapter(RecordingMultimodalAdapter):
        profile = replace(
            FakeEmbeddingAdapter.profile,
            image_transform=codec.transform_id,
        )

    config = write_generated_preprocessing_fixture(tmp_path)
    source_bytes = b"generated bomb header"
    attach_generated_header_image(config, source_bytes)
    adapter = BombAdapter()

    result = run_embedding_pipeline(
        config,
        adapter,
        limit=1,
        image_codec=codec,
    )

    assert result == EmbeddingRunResult(
        articles=1,
        embeddings=1,
        attempted=1,
        succeeded=1,
        terminal=1,
        header_failed=1,
    )
    assert adapter.image_calls == 0
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute(
            """
            SELECT state, attempt_count, error_code
            FROM embedding_work_items
            WHERE modality = 'header_image'
            """
        ).fetchone() == ("terminal", 0, "unsafe_image")
        assert db.execute(
            "SELECT count(*) FROM embeddings WHERE modality = 'header_image'"
        ).fetchone() == (0,)


def test_pillow_jpeg_build_identity_changes_configuration_and_mismatch_fails_closed(
    tmp_path,
    monkeypatch,
):
    """Break caught: distinct linked JPEG builds claim identical vector meaning."""

    install_synthetic_pillow(
        monkeypatch,
        jpeg_version="9e",
        turbo_version="3.1.0",
        open_image=lambda stream: None,
    )
    first_codec = PillowImageCodec()
    install_synthetic_pillow(
        monkeypatch,
        jpeg_version="9f",
        turbo_version="3.2.0",
        open_image=lambda stream: None,
    )
    second_codec = PillowImageCodec()
    first_adapter = JinaEmbeddingAdapter(
        environment={"JINA_API_KEY": "generated-fixture-key"},
    )
    second_adapter = JinaEmbeddingAdapter(
        environment={"JINA_API_KEY": "generated-fixture-key"},
    )
    first_adapter.bind_image_codec(first_codec)
    second_adapter.bind_image_codec(second_codec)

    assert first_codec.transform_id != second_codec.transform_id
    assert first_adapter.profile.image_transform == first_codec.transform_id
    assert second_adapter.profile.image_transform == second_codec.transform_id
    assert configuration_id(first_adapter.profile) != configuration_id(
        second_adapter.profile
    )
    with pytest.raises(ImageCodecError) as rebound:
        first_adapter.bind_image_codec(second_codec)
    assert rebound.value.code == "ambiguous_image_configuration"

    class SecondBuildAdapter(RecordingMultimodalAdapter):
        profile = second_adapter.profile

    config = write_generated_preprocessing_fixture(tmp_path)
    source_bytes = b"generated mismatched jpeg build"
    attach_generated_header_image(config, source_bytes)
    adapter = SecondBuildAdapter()

    with pytest.raises(EmbeddingPipelineError) as raised:
        run_embedding_pipeline(
            config,
            adapter,
            limit=1,
            image_codec=first_codec,
        )

    assert raised.value.code == "ambiguous_image_configuration"
    assert adapter.image_calls == 0


def test_deterministic_header_rejection_becomes_terminal_after_three_attempts(
    tmp_path,
):
    """Break caught: deterministic image rejection retries forever or stores filler."""

    config = write_generated_preprocessing_fixture(tmp_path)
    attach_generated_header_image(config, b"generated rejected image")

    class RejectingImageAdapter(RecordingMultimodalAdapter):
        total_image_calls = 0

        def embed_image(self, image_base64: str) -> tuple[float, ...]:
            self.image_calls += 1
            self.encoded_images.append(image_base64)
            type(self).total_image_calls += 1
            raise JinaHostedAdapterError(
                "deterministic_request",
                retryable=False,
                status_code=413,
            )

    results = [
        run_embedding_pipeline(config, RejectingImageAdapter(), limit=1)
        for _ in range(4)
    ]

    assert [(result.retryable, result.terminal, result.header_failed)
            for result in results] == [
        (1, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (0, 1, 1),
    ]
    assert RejectingImageAdapter.total_image_calls == 3
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute(
            """
            SELECT state, attempt_count, error_code, status_code
            FROM embedding_work_items
            WHERE modality = 'header_image'
            """
        ).fetchone() == ("terminal", 3, "deterministic_request", 413)
        assert db.execute(
            "SELECT modality FROM embeddings ORDER BY modality"
        ).fetchall() == [("article_text",)]
    assert {issue.code for issue in validate_embedding_outputs(config).issues} == {
        "header_image_terminal_failure"
    }


def test_terminal_header_observation_and_summary_survive_registration_interrupt(
    tmp_path,
    monkeypatch,
):
    """Break caught: terminal replay observation commits before its run counters."""

    config = write_generated_preprocessing_fixture(tmp_path)
    attach_generated_header_image(config, b"generated terminal replay image")

    class RejectingImageAdapter(FakeEmbeddingAdapter):
        def embed_image(self, image_base64: str) -> tuple[float, ...]:
            del image_base64
            raise JinaHostedAdapterError(
                "deterministic_request",
                retryable=False,
                status_code=413,
            )

    for _ in range(3):
        run_embedding_pipeline(config, RejectingImageAdapter(), limit=1)

    real_transaction = EmbeddingCatalog.transaction
    interrupted = False

    @contextmanager
    def interrupt_after_terminal_registration(catalog):
        nonlocal interrupted
        with real_transaction(catalog):
            yield
        latest_run_id = catalog.connection.execute(
            "SELECT run_id FROM runs ORDER BY started_at DESC, run_id DESC LIMIT 1"
        ).fetchone()[0]
        terminal_observed = catalog.connection.execute(
            """
            SELECT count(*)
            FROM embedding_work_items
            WHERE modality = 'header_image' AND state = 'terminal'
              AND last_run_id = ?
            """,
            [latest_run_id],
        ).fetchone()[0]
        if terminal_observed and not interrupted:
            interrupted = True
            raise SyntheticInterruption

    monkeypatch.setattr(
        EmbeddingCatalog,
        "transaction",
        interrupt_after_terminal_registration,
    )
    with pytest.raises(SyntheticInterruption):
        run_embedding_pipeline(config, RecordingMultimodalAdapter(), limit=1)
    monkeypatch.setattr(EmbeddingCatalog, "transaction", real_transaction)

    assert interrupted
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        latest_run_id, terminal, header_failed = db.execute(
            """
            SELECT run_id, terminal, header_failed
            FROM runs
            ORDER BY started_at DESC, run_id DESC
            LIMIT 1
            """
        ).fetchone()
        assert (terminal, header_failed) == (1, 1)
        assert db.execute(
            """
            SELECT last_run_id
            FROM embedding_work_items
            WHERE modality = 'header_image'
            """
        ).fetchone() == (latest_run_id,)
    codes = {issue.code for issue in validate_embedding_outputs(config).issues}
    assert "header_image_terminal_failure" in codes
    assert "invalid_header_disposition_counts" not in codes


def test_changed_header_bytes_invalidate_only_image_and_same_config_multimodal(
    tmp_path,
):
    """Break caught: changed image bytes invalidate text, siblings, or old configs."""

    config = write_generated_preprocessing_fixture(tmp_path)
    relative_path = attach_generated_header_image(config, b"generated image one")
    run_embedding_pipeline(config, RecordingMultimodalAdapter(), limit=1)
    run_embedding_pipeline(config, PriorConfigurationAdapter(), limit=1)
    current_profile = FakeEmbeddingAdapter.profile
    prior_profile = PriorConfigurationAdapter.profile
    current_id = configuration_id(current_profile)
    prior_id = configuration_id(prior_profile)
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        original_generations = dict(
            db.execute(
                """
                SELECT modality, generation_run_id
                FROM embedding_work_items
                WHERE configuration_id = ?
                """,
                [current_id],
            ).fetchall()
        )
    replacement = b"generated image two"
    (config.source_root / relative_path).write_bytes(replacement)
    adapter = RecordingMultimodalAdapter()

    result = run_embedding_pipeline(config, adapter, limit=1)

    assert result == EmbeddingRunResult(
        articles=1,
        embeddings=2,
        reused=1,
        attempted=1,
        succeeded=2,
    )
    assert adapter.calls == 0
    assert adapter.image_calls == 1
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute(
            """
            SELECT configuration_id, modality
            FROM embeddings
            ORDER BY configuration_id, modality
            """
        ).fetchall() == sorted(
            [
                (current_id, "article_text"),
                (current_id, "header_image"),
                (current_id, "multimodal_article"),
                (prior_id, "article_text"),
                (prior_id, "header_image"),
                (prior_id, "multimodal_article"),
            ]
        )
        current_generations = dict(
            db.execute(
                """
                SELECT modality, generation_run_id
                FROM embedding_work_items
                WHERE configuration_id = ?
                """,
                [current_id],
            ).fetchall()
        )
        assert (
            current_generations["article_text"]
            == original_generations["article_text"]
        )
        assert (
            current_generations["header_image"]
            != original_generations["header_image"]
        )
        assert (
            current_generations["multimodal_article"]
            != original_generations["multimodal_article"]
        )
        assert db.execute(
            """
            SELECT modality, superseded_reason
            FROM embedding_generation_history
            WHERE configuration_id = ?
            ORDER BY modality
            """,
            [current_id],
        ).fetchall() == [
            ("header_image", "input_changed"),
            ("multimodal_article", "input_changed"),
        ]
    assert validate_embedding_outputs(
        config,
        configuration_id=current_id,
    ) == EmbeddingValidationResult(issues=())


def test_removed_canonical_header_invalidates_image_and_same_config_multimodal(
    tmp_path,
):
    """Break caught: becoming no-header leaves image-dependent vectors active."""

    config = write_generated_preprocessing_fixture(tmp_path)
    attach_generated_header_image(config, b"generated removed image")
    run_embedding_pipeline(config, RecordingMultimodalAdapter(), limit=1)
    run_embedding_pipeline(config, PriorConfigurationAdapter(), limit=1)
    current_profile = FakeEmbeddingAdapter.profile
    prior_profile = PriorConfigurationAdapter.profile
    with Catalog.open(config.preprocessing_catalog) as catalog, catalog.transaction():
        catalog.connection.execute(
            """
            UPDATE articles
            SET header_image_path = NULL
            WHERE article_id = 'wsj:SYNTHETIC-EMBEDDING'
            """
        )
    adapter = RecordingMultimodalAdapter()

    result = run_embedding_pipeline(config, adapter, limit=1)

    assert result == EmbeddingRunResult(
        articles=1,
        embeddings=0,
        reused=1,
        header_absent=1,
    )
    assert adapter.calls == 0
    assert adapter.image_calls == 0
    current_id = configuration_id(current_profile)
    prior_id = configuration_id(prior_profile)
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute(
            """
            SELECT configuration_id, modality
            FROM embeddings
            ORDER BY configuration_id, modality
            """
        ).fetchall() == sorted(
            [
                (current_id, "article_text"),
                (prior_id, "article_text"),
                (prior_id, "header_image"),
                (prior_id, "multimodal_article"),
            ]
        )
        assert db.execute(
            """
            SELECT modality, state, generation_run_id
            FROM embedding_work_items
            WHERE configuration_id = ? AND modality IN (
                'header_image', 'multimodal_article'
            )
            ORDER BY modality
            """,
            [current_id],
        ).fetchall() == [
            ("header_image", "not_applicable", None),
            ("multimodal_article", "queued", None),
        ]
        assert db.execute(
            """
            SELECT modality, superseded_reason
            FROM embedding_generation_history
            WHERE configuration_id = ?
            ORDER BY modality
            """,
            [current_id],
        ).fetchall() == [
            ("header_image", "input_changed"),
            ("multimodal_article", "input_changed"),
        ]
    assert validate_embedding_outputs(
        config,
        configuration_id=current_id,
    ) == EmbeddingValidationResult(issues=())


def test_interrupted_header_attempt_recovers_without_rebuying_successful_text(
    tmp_path,
):
    """Break caught: interrupted image work loses text or appears successful."""

    config = write_generated_preprocessing_fixture(tmp_path)
    attach_generated_header_image(config, b"generated interrupted image")

    with pytest.raises(SyntheticInterruption):
        run_embedding_pipeline(config, InterruptingImageAdapter(), limit=1)

    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute(
            """
            SELECT modality, state, attempt_count
            FROM embedding_work_items
            ORDER BY modality
            """
        ).fetchall() == [
            ("article_text", "succeeded", 1),
            ("header_image", "in_progress", 1),
        ]
        assert db.execute(
            "SELECT modality FROM embeddings ORDER BY modality"
        ).fetchall() == [("article_text",)]
    assert {issue.code for issue in validate_embedding_outputs(config).issues} == {
        "header_image_in_progress"
    }

    recovered_adapter = RecordingMultimodalAdapter()
    recovered = run_embedding_pipeline(config, recovered_adapter, limit=1)

    assert recovered == EmbeddingRunResult(
        articles=1,
        embeddings=2,
        reused=1,
        attempted=1,
        succeeded=2,
        interrupted=1,
    )
    assert recovered_adapter.calls == 0
    assert recovered_adapter.image_calls == 1
    assert validate_embedding_outputs(config).ok


@pytest.mark.parametrize("unsafe_path", ("../outside.png", "/absolute.png"))
def test_pipeline_rejects_uncontained_header_image_paths(tmp_path, unsafe_path):
    """Break caught: a catalogued image path escapes the immutable archive root."""

    config = write_generated_preprocessing_fixture(tmp_path)
    with Catalog.open(config.preprocessing_catalog) as catalog, catalog.transaction():
        catalog.connection.execute(
            """
            UPDATE articles SET header_image_path = ?
            WHERE article_id = 'wsj:SYNTHETIC-EMBEDDING'
            """,
            [unsafe_path],
        )

    with pytest.raises(EmbeddingPipelineError) as raised:
        run_embedding_pipeline(config, RecordingMultimodalAdapter(), limit=1)

    assert raised.value.code == "unsafe_header_image_path"
    assert not config.embedding_output_root.exists()


def test_pipeline_refuses_symlinked_header_image_without_reading_referent(tmp_path):
    """Break caught: a catalogued local image follows a symlink outside its root."""

    config = write_generated_preprocessing_fixture(tmp_path)
    outside_image = tmp_path / "outside-secret.png"
    outside_image.write_bytes(b"outside generated bytes")
    relative_path = "synthetic/link.png"
    image_path = config.source_root / relative_path
    image_path.parent.mkdir(parents=True)
    image_path.symlink_to(outside_image)
    with Catalog.open(config.preprocessing_catalog) as catalog, catalog.transaction():
        catalog.connection.execute(
            """
            UPDATE articles SET header_image_path = ?
            WHERE article_id = 'wsj:SYNTHETIC-EMBEDDING'
            """,
            [relative_path],
        )

    with pytest.raises(EmbeddingPipelineError) as raised:
        run_embedding_pipeline(config, RecordingMultimodalAdapter(), limit=1)

    assert raised.value.code == "unsafe_header_image_path"
    assert "outside generated bytes" not in str(raised.value)
    assert not config.embedding_output_root.exists()


def test_pipeline_detects_header_image_replacement_after_safe_open(
    tmp_path,
    monkeypatch,
):
    """Break caught: hashing uses bytes from a path replaced after safe open."""

    config = write_generated_preprocessing_fixture(tmp_path)
    relative_path = attach_generated_header_image(config, b"generated first image")
    image_path = config.source_root / relative_path
    real_open = source_image_module.open_relative_regular
    swapped = False

    def swap_after_open(root, relative_value):
        nonlocal swapped
        status, descriptor = real_open(root, relative_value)
        if status == "ok" and not swapped:
            swapped = True
            image_path.unlink()
            image_path.write_bytes(b"generated replacement image")
        return status, descriptor

    monkeypatch.setattr(
        source_image_module,
        "open_relative_regular",
        swap_after_open,
    )

    with pytest.raises(EmbeddingPipelineError) as raised:
        run_embedding_pipeline(config, RecordingMultimodalAdapter(), limit=1)

    assert swapped
    assert raised.value.code == "unsafe_header_image_path"
    assert "generated" not in str(raised.value)
    assert not config.embedding_output_root.exists()


def test_pipeline_refuses_hard_linked_header_image_leaf(tmp_path):
    """Break caught: a multiply linked image leaf bypasses source ownership."""

    config = write_generated_preprocessing_fixture(tmp_path)
    relative_path = attach_generated_header_image(config, b"generated linked image")
    image_path = config.source_root / relative_path
    linked_path = config.source_root / "synthetic" / "linked-copy.png"
    linked_path.hardlink_to(image_path)

    with pytest.raises(EmbeddingPipelineError) as raised:
        run_embedding_pipeline(config, RecordingMultimodalAdapter(), limit=1)

    assert raised.value.code == "unsafe_header_image_path"
    assert not config.embedding_output_root.exists()


def test_image_supersession_after_reuse_retains_true_generation_run(tmp_path):
    """Break caught: replay observation is misreported as vector generation."""

    config = write_generated_preprocessing_fixture(tmp_path)
    image_bytes = b"generated first provenance image"
    relative_path = attach_generated_header_image(config, image_bytes)
    run_embedding_pipeline(config, RecordingMultimodalAdapter(), limit=1)
    configuration_identifier = configuration_id(FakeEmbeddingAdapter.profile)
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        first_generation_run = db.execute(
            """
            SELECT generation_run_id
            FROM embedding_work_items
            WHERE modality = 'header_image'
            """
        ).fetchone()[0]

    replay = run_embedding_pipeline(config, RecordingMultimodalAdapter(), limit=1)
    assert replay.reused == 3
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        generation_run_id, replay_observation_run = db.execute(
            """
            SELECT generation_run_id, last_run_id
            FROM embedding_work_items
            WHERE modality = 'header_image'
            """
        ).fetchone()
    assert generation_run_id == first_generation_run
    assert replay_observation_run != first_generation_run

    replacement_bytes = b"generated replacement provenance image"
    (config.source_root / relative_path).write_bytes(replacement_bytes)
    run_embedding_pipeline(config, RecordingMultimodalAdapter(), limit=1)

    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        active_generation_run = db.execute(
            """
            SELECT generation_run_id
            FROM embedding_work_items
            WHERE modality = 'header_image'
            """
        ).fetchone()[0]
        history = db.execute(
            """
            SELECT generation_run_id, superseded_run_id, input_sha256,
                   source_relative_path
            FROM embedding_generation_history
            WHERE modality = 'header_image' AND configuration_id = ?
            """,
            [configuration_identifier],
        ).fetchone()
    assert history == (
        first_generation_run,
        active_generation_run,
        hashlib.sha256(image_bytes).hexdigest(),
        relative_path,
    )
    assert history[0] != replay_observation_run


def test_replaying_unchanged_success_reuses_checkpoint_without_adapter_call(tmp_path):
    """Break caught: a limited replay purchases an already committed vector again."""

    config = write_generated_preprocessing_fixture(tmp_path)
    first_adapter = RecordingAdapter()
    first = run_embedding_pipeline(config, first_adapter, limit=1)
    replay_adapter = RecordingAdapter()

    replay = run_embedding_pipeline(config, replay_adapter, limit=1)

    assert first == EmbeddingRunResult(
        articles=1,
        embeddings=1,
        reused=0,
        attempted=1,
        succeeded=1,
        retryable=0,
        terminal=0,
        interrupted=0,
        header_absent=1,
    )
    assert replay == EmbeddingRunResult(
        articles=1,
        embeddings=0,
        reused=1,
        attempted=0,
        succeeded=0,
        retryable=0,
        terminal=0,
        interrupted=0,
        header_absent=1,
    )
    assert first_adapter.calls == 1
    assert replay_adapter.calls == 0
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute("SELECT count(*) FROM embeddings").fetchone() == (1,)
        assert db.execute(
            """
            SELECT state, attempt_count, error_code, status_code,
                   retry_after_seconds
            FROM embedding_work_items
            """
        ).fetchone() == ("succeeded", 1, None, None, None)


def test_metadata_only_preprocessing_change_reuses_vector_and_refreshes_metadata(
    tmp_path,
):
    """Break caught: path/date-only canonical changes repurchase the same vector."""

    config = write_generated_preprocessing_fixture(tmp_path)
    run_embedding_pipeline(config, RecordingAdapter(), limit=1)
    old_path = (
        config.preprocessing_output_root / "text" / "2024" / "01" / "02" / "article.md"
    )
    new_path = config.preprocessing_output_root / "text" / "metadata-only.md"
    new_path.write_bytes(old_path.read_bytes())
    new_published = datetime(2024, 1, 4, 15, 30, tzinfo=UTC)
    with Catalog.open(config.preprocessing_catalog) as catalog, catalog.transaction():
        catalog.connection.execute(
            """
            UPDATE articles
            SET cleaned_markdown_path = ?, published_at_utc = ?,
                publication_date_new_york = ?
            WHERE article_id = 'wsj:SYNTHETIC-EMBEDDING'
            """,
            ["text/metadata-only.md", new_published, date(2024, 1, 4)],
        )
    replay_adapter = RecordingAdapter()

    replay = run_embedding_pipeline(config, replay_adapter, limit=1)

    assert replay.reused == 1
    assert replay_adapter.calls == 0
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute(
            """
            SELECT published_at_utc, publication_date_new_york
            FROM embeddings
            """
        ).fetchone() == (new_published, date(2024, 1, 4))
        assert db.execute(
            "SELECT count(*) FROM embedding_generation_history"
        ).fetchone() == (0,)


def test_changed_markdown_regenerates_only_affected_text_and_audits_prior_generation(
    tmp_path,
):
    """Break caught: one content change repurchases siblings or loses provenance."""

    config = write_generated_preprocessing_fixture(tmp_path)
    add_generated_embedding_article(
        config,
        article_id="wsj:ZZZ-SYNTHETIC-EMBEDDING",
        markdown="# Unchanged sibling\n",
    )
    run_embedding_pipeline(config, RecordingAdapter(), limit=2)
    changed_hash = replace_generated_embedding_markdown(
        config, "# Changed generated article\n"
    )
    adapter = RecordingAdapter()

    result = run_embedding_pipeline(config, adapter, limit=2)

    assert result.reused == 1
    assert result.succeeded == 1
    assert adapter.calls == 1
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute(
            "SELECT article_id, input_sha256 FROM embeddings ORDER BY article_id"
        ).fetchall()[0] == ("wsj:SYNTHETIC-EMBEDDING", changed_hash)
        history = db.execute(
            """
            SELECT article_id, modality, configuration_id, generation_run_id,
                   input_sha256, stored_vector_sha256, superseded_run_id,
                   superseded_reason
            FROM embedding_generation_history
            """
        ).fetchone()
    assert history is not None
    assert history[:3] == (
        "wsj:SYNTHETIC-EMBEDDING",
        "article_text",
        configuration_id(FakeEmbeddingAdapter.profile),
    )
    assert history[3] != history[6]
    assert history[4] == hashlib.sha256(b"#\n").hexdigest()
    assert len(history[5]) == 64
    assert history[7] == "input_changed"


@pytest.mark.parametrize(
    ("reprocess", "expected_reason"),
    [(False, "input_changed"), (True, "explicit_reprocess")],
)
def test_text_supersession_invalidates_only_its_multimodal_dependent(
    tmp_path,
    reprocess,
    expected_reason,
):
    """Break caught: superseded text leaves a stale multimodal vector queryable."""

    config = write_generated_preprocessing_fixture(tmp_path)
    sibling_id = "wsj:ZZZ-SYNTHETIC-EMBEDDING"
    target_id = "wsj:SYNTHETIC-EMBEDDING"
    add_generated_embedding_article(
        config,
        article_id=sibling_id,
        markdown="# Unchanged sibling\n",
    )
    attach_generated_header_image(config, b"generated target header")
    sibling_header = "synthetic/sibling_main_image.png"
    (config.source_root / sibling_header).write_bytes(b"generated sibling header")
    with Catalog.open(config.preprocessing_catalog) as catalog, catalog.transaction():
        catalog.connection.execute(
            "UPDATE articles SET header_image_path = ? WHERE article_id = ?",
            [sibling_header, sibling_id],
        )
    run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=2)
    run_embedding_pipeline(config, PriorConfigurationAdapter(), limit=2)
    current_profile = FakeEmbeddingAdapter.profile
    prior_profile = PriorConfigurationAdapter.profile
    current_id = configuration_id(current_profile)
    prior_id = configuration_id(prior_profile)
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        before = {
            (article_id, config_id, modality): generation_run_id
            for article_id, config_id, modality, generation_run_id in db.execute(
                """
                SELECT article_id, configuration_id, modality, generation_run_id
                FROM embedding_work_items
                """
            ).fetchall()
        }
    if not reprocess:
        replace_generated_embedding_markdown(config, "# Changed generated article\n")

    run_embedding_pipeline(
        config,
        FakeEmbeddingAdapter(),
        limit=1,
        reprocess=reprocess,
    )

    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        after = {
            (article_id, config_id, modality): generation_run_id
            for article_id, config_id, modality, generation_run_id in db.execute(
                """
                SELECT article_id, configuration_id, modality, generation_run_id
                FROM embedding_work_items
                """
            ).fetchall()
        }
        assert after[(target_id, current_id, "article_text")] != before[
            (target_id, current_id, "article_text")
        ]
        assert after[(target_id, current_id, "multimodal_article")] != before[
            (target_id, current_id, "multimodal_article")
        ]
        assert after[(target_id, current_id, "header_image")] == before[
            (target_id, current_id, "header_image")
        ]
        assert all(
            after[key] == generation_run_id
            for key, generation_run_id in before.items()
            if key[0] == sibling_id or key[1] == prior_id
        )
        assert db.execute(
            """
            SELECT state
            FROM embedding_work_items
            WHERE article_id = ? AND configuration_id = ?
              AND modality = 'multimodal_article'
            """,
            [target_id, current_id],
        ).fetchone() == ("succeeded",)
        assert db.execute(
            """
            SELECT modality, superseded_reason
            FROM embedding_generation_history
            WHERE article_id = ? AND configuration_id = ?
            ORDER BY modality
            """,
            [target_id, current_id],
        ).fetchall() == [
            ("article_text", expected_reason),
            ("multimodal_article", expected_reason),
        ]


def test_changed_configuration_keeps_both_vectors_and_requires_validation_selector(
    tmp_path,
):
    """Break caught: changed meaning overwrites or silently mixes old vectors."""

    config = write_generated_preprocessing_fixture(tmp_path)
    run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)
    run_embedding_pipeline(config, PriorConfigurationAdapter(), limit=1)
    active_id = configuration_id(FakeEmbeddingAdapter.profile)
    prior_id = configuration_id(PriorConfigurationAdapter.profile)

    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute(
            "SELECT configuration_id FROM embeddings ORDER BY configuration_id"
        ).fetchall() == sorted([(active_id,), (prior_id,)])
    assert validate_embedding_outputs(config) == EmbeddingValidationResult(
        issues=(
            EmbeddingValidationIssue(
                "configuration_id_required",
                "validation requires a configuration identity when generations coexist",
            ),
        )
    )
    assert validate_embedding_outputs(
        config, configuration_id=active_id
    ) == EmbeddingValidationResult(issues=())
    assert validate_embedding_outputs(
        config, configuration_id=prior_id
    ) == EmbeddingValidationResult(issues=())


def test_explicit_reprocess_is_bounded_and_audits_replaced_success(tmp_path):
    """Break caught: reprocess expands beyond its limit or destroys prior provenance."""

    config = write_generated_preprocessing_fixture(tmp_path)
    add_generated_embedding_article(
        config,
        article_id="wsj:ZZZ-SYNTHETIC-EMBEDDING",
        markdown="# Unchanged sibling\n",
    )
    run_embedding_pipeline(config, RecordingAdapter(), limit=2)
    adapter = RecordingAdapter()

    result = run_embedding_pipeline(config, adapter, limit=1, reprocess=True)

    assert result.reused == 0
    assert result.succeeded == 1
    assert adapter.calls == 1
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute("SELECT count(*) FROM embeddings").fetchone() == (2,)
        assert db.execute(
            """
            SELECT article_id, superseded_reason
            FROM embedding_generation_history
            """
        ).fetchall() == [
            ("wsj:SYNTHETIC-EMBEDDING", "explicit_reprocess")
        ]
        assert db.execute(
            """
                SELECT article_id, attempt_count
                FROM embedding_work_items
                WHERE modality = 'article_text'
                ORDER BY article_id
            """
        ).fetchall() == [
            ("wsj:SYNTHETIC-EMBEDDING", 1),
            ("wsj:ZZZ-SYNTHETIC-EMBEDDING", 1),
        ]


def test_validator_reports_malformed_generation_history(tmp_path):
    """Break caught: superseded provenance can be corrupted without detection."""

    config = write_generated_preprocessing_fixture(tmp_path)
    run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)
    run_embedding_pipeline(
        config,
        FakeEmbeddingAdapter(),
        limit=1,
        reprocess=True,
    )
    with duckdb.connect(str(config.embedding_catalog)) as db:
        db.execute(
            """
            UPDATE embedding_generation_history
            SET superseded_reason = 'synthetic-invalid-reason'
            """
        )

    assert validate_embedding_outputs(config) == EmbeddingValidationResult(
        issues=(
            EmbeddingValidationIssue(
                "invalid_supersession_reason",
                "embedding generation history uses an unsupported reason",
            ),
        )
    )


@pytest.mark.parametrize(
    "after_response",
    (False, True),
    ids=("before-request", "after-response"),
)
def test_restart_recovers_in_progress_work_after_adapter_interruption(
    tmp_path,
    after_response,
):
    """Break caught: interrupted in-progress work is lost or considered successful."""

    config = write_generated_preprocessing_fixture(tmp_path)
    interrupted_adapter = InterruptingAdapter(after_response=after_response)

    with pytest.raises(SyntheticInterruption):
        run_embedding_pipeline(config, interrupted_adapter, limit=1)

    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute(
            "SELECT state, attempt_count FROM embedding_work_items"
        ).fetchone() == ("in_progress", 1)
        assert db.execute("SELECT count(*) FROM embeddings").fetchone() == (0,)

    resumed_adapter = RecordingAdapter()
    resumed = run_embedding_pipeline(config, resumed_adapter, limit=1)

    assert resumed == EmbeddingRunResult(
        articles=1,
        embeddings=1,
        reused=0,
        attempted=1,
        succeeded=1,
        retryable=0,
        terminal=0,
        interrupted=1,
        header_absent=1,
    )
    assert interrupted_adapter.calls == 1
    assert resumed_adapter.calls == 1


@pytest.mark.parametrize(
    "commit_side",
    ("before", "after"),
    ids=("before-catalog-commit", "after-catalog-commit"),
)
def test_success_checkpoint_survives_interruptions_around_catalog_commit(
    tmp_path,
    monkeypatch,
    commit_side,
):
    """Break caught: commit ordering loses a vector or marks success prematurely."""

    config = write_generated_preprocessing_fixture(tmp_path)
    adapter = RecordingAdapter()
    real_transaction = EmbeddingCatalog.transaction
    interrupted = False

    @contextmanager
    def interrupt_around_success_commit(catalog):
        nonlocal interrupted
        if commit_side == "before":
            with real_transaction(catalog):
                yield
                if adapter.responded and not interrupted:
                    interrupted = True
                    raise SyntheticInterruption
        else:
            with real_transaction(catalog):
                yield
            if adapter.responded and not interrupted:
                interrupted = True
                raise SyntheticInterruption

    monkeypatch.setattr(
        EmbeddingCatalog,
        "transaction",
        interrupt_around_success_commit,
    )
    with pytest.raises(SyntheticInterruption):
        run_embedding_pipeline(config, adapter, limit=1)
    monkeypatch.setattr(EmbeddingCatalog, "transaction", real_transaction)

    replay_adapter = RecordingAdapter()
    resumed = run_embedding_pipeline(config, replay_adapter, limit=1)

    assert interrupted
    assert adapter.calls == 1
    expected_reused = 1 if commit_side == "after" else 0
    assert replay_adapter.calls == 1 - expected_reused
    assert resumed.reused == expected_reused
    assert resumed.succeeded == 1 - expected_reused
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute("SELECT count(*) FROM embeddings").fetchone() == (1,)
        assert db.execute(
            "SELECT state, attempt_count FROM embedding_work_items"
        ).fetchone() == ("succeeded", 1 if commit_side == "after" else 2)


def test_retryable_failure_can_succeed_after_an_earlier_checkpoint(tmp_path):
    """Break caught: a later failure rolls back an earlier successful vector."""

    config = write_generated_preprocessing_fixture(tmp_path)
    add_generated_embedding_article(
        config,
        article_id="wsj:ZZZ-SYNTHETIC-EMBEDDING",
        markdown="# Second generated article\n",
    )

    class FailSecondAdapter(RecordingAdapter):
        def embed_text(self, text: str) -> tuple[float, ...]:
            if self.calls == 0:
                return super().embed_text(text)
            self.calls += 1
            raise JinaHostedAdapterError(
                "rate_limit",
                retryable=True,
                status_code=429,
                retry_after_seconds=2.0,
            )

    first_adapter = FailSecondAdapter()
    with pytest.raises(JinaHostedAdapterError, match="rate_limit"):
        run_embedding_pipeline(config, first_adapter, limit=2)

    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute(
            "SELECT article_id FROM embeddings ORDER BY article_id"
        ).fetchall() == [("wsj:SYNTHETIC-EMBEDDING",)]
        assert db.execute(
            """
                SELECT article_id, state, attempt_count, error_code, status_code,
                       retry_after_seconds
                FROM embedding_work_items
                WHERE modality = 'article_text'
                ORDER BY article_id
            """
        ).fetchall() == [
            (
                "wsj:SYNTHETIC-EMBEDDING",
                "succeeded",
                1,
                None,
                None,
                None,
            ),
            (
                "wsj:ZZZ-SYNTHETIC-EMBEDDING",
                "retryable",
                1,
                "rate_limit",
                429,
                2.0,
            ),
        ]
        assert db.execute(
            """
            SELECT articles, embeddings, reused, attempted, succeeded,
                   retryable, terminal, interrupted
            FROM runs
            """
        ).fetchone() == (2, 1, 0, 2, 1, 1, 0, 0)

    resumed_adapter = RecordingAdapter()
    resumed = run_embedding_pipeline(config, resumed_adapter, limit=2)

    assert resumed == EmbeddingRunResult(
        articles=2,
        embeddings=1,
        reused=1,
        attempted=1,
        succeeded=1,
        retryable=0,
        terminal=0,
        interrupted=0,
        header_absent=2,
    )
    assert resumed_adapter.calls == 1
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute("SELECT count(*) FROM embeddings").fetchone() == (2,)


def test_terminal_failure_is_visible_and_not_retried_implicitly(tmp_path):
    """Break caught: a deterministic hosted rejection is silently retried."""

    config = write_generated_preprocessing_fixture(tmp_path)

    class TerminalAdapter(FakeEmbeddingAdapter):
        def embed_text(self, text: str) -> tuple[float, ...]:
            raise JinaHostedAdapterError(
                "deterministic_request",
                retryable=False,
                status_code=413,
            )

    with pytest.raises(JinaHostedAdapterError, match="deterministic_request"):
        run_embedding_pipeline(config, TerminalAdapter(), limit=1)

    replay_adapter = RecordingAdapter()
    replay = run_embedding_pipeline(config, replay_adapter, limit=1)

    assert replay == EmbeddingRunResult(
        articles=1,
        embeddings=0,
        reused=0,
        attempted=0,
        succeeded=0,
        retryable=0,
        terminal=1,
        interrupted=0,
        header_absent=1,
    )
    assert replay_adapter.calls == 0
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute(
            """
            SELECT state, attempt_count, error_code, status_code,
                   retry_after_seconds
            FROM embedding_work_items
            """
        ).fetchone() == ("terminal", 1, "deterministic_request", 413, None)


@pytest.mark.parametrize(
    ("failure_kind", "expected_state"),
    (
        ("interrupted", "in_progress"),
        ("retryable", "retryable"),
        ("terminal", "terminal"),
    ),
)
def test_changed_input_removes_stale_active_vector_until_recovery(
    tmp_path,
    failure_kind,
    expected_state,
):
    """Break caught: changed-input failure leaves the old active vector queryable."""

    config = write_generated_preprocessing_fixture(tmp_path)
    run_embedding_pipeline(config, PriorConfigurationAdapter(), limit=1)
    run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)
    active_configuration_id = configuration_id(FakeEmbeddingAdapter.profile)
    prior_configuration_id = configuration_id(PriorConfigurationAdapter.profile)
    changed_hash = replace_generated_embedding_markdown(
        config,
        "# Changed generated article\n",
    )

    if failure_kind == "interrupted":
        failing_adapter = InterruptingAdapter(after_response=True)
        expected_error = SyntheticInterruption
    else:
        retryable = failure_kind == "retryable"

        class ChangedInputFailureAdapter(FakeEmbeddingAdapter):
            def embed_text(self, text: str) -> tuple[float, ...]:
                raise JinaHostedAdapterError(
                    "rate_limit" if retryable else "deterministic_request",
                    retryable=retryable,
                    status_code=429 if retryable else 413,
                )

        failing_adapter = ChangedInputFailureAdapter()
        expected_error = JinaHostedAdapterError

    with pytest.raises(expected_error):
        run_embedding_pipeline(config, failing_adapter, limit=1)

    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute(
            """
            SELECT count(*)
            FROM embeddings
            WHERE configuration_id = ?
            """,
            [active_configuration_id],
        ).fetchone() == (0,)
        assert db.execute(
            """
            SELECT configuration_id, input_sha256
            FROM embeddings
            """
        ).fetchall() == [
            (
                prior_configuration_id,
                hashlib.sha256(b"#\n").hexdigest(),
            )
        ]
        assert db.execute(
            """
            SELECT state
            FROM embedding_work_items
            WHERE configuration_id = ?
            """,
            [active_configuration_id],
        ).fetchone() == (expected_state,)

    if failure_kind == "terminal":
        unchanged_replay = RecordingAdapter()
        run_embedding_pipeline(config, unchanged_replay, limit=1)
        assert unchanged_replay.calls == 0
        changed_hash = replace_generated_embedding_markdown(
            config,
            "# Changed generated article again\n",
        )

    recovered = run_embedding_pipeline(config, RecordingAdapter(), limit=1)

    assert recovered.succeeded == 1
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute(
            """
            SELECT input_sha256
            FROM embeddings
            WHERE configuration_id = ?
            """,
            [active_configuration_id],
        ).fetchall() == [(changed_hash,)]
        assert db.execute(
            """
            SELECT count(*)
            FROM embeddings
            WHERE configuration_id = ?
            """,
            [active_configuration_id],
        ).fetchone() == (1,)


def test_registration_checkpoint_survives_later_registration_interruption(
    tmp_path,
    monkeypatch,
):
    """Break caught: one large registration transaction loses earlier work."""

    config = write_generated_preprocessing_fixture(tmp_path)
    add_generated_embedding_article(
        config,
        article_id="wsj:ZZZ-SYNTHETIC-EMBEDDING",
        markdown="# Second generated article\n",
    )
    real_register_work = embedding_pipeline_module._register_work
    registrations = 0

    def interrupt_second_registration(*args, **kwargs):
        nonlocal registrations
        registrations += 1
        if registrations == 2:
            raise SyntheticInterruption
        return real_register_work(*args, **kwargs)

    monkeypatch.setattr(
        embedding_pipeline_module,
        "_register_work",
        interrupt_second_registration,
    )
    with pytest.raises(SyntheticInterruption):
        run_embedding_pipeline(config, RecordingAdapter(), limit=2)
    monkeypatch.setattr(
        embedding_pipeline_module,
        "_register_work",
        real_register_work,
    )

    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute(
            """
            SELECT article_id, modality, state
            FROM embedding_work_items
            ORDER BY article_id, modality
            """
        ).fetchall() == [
            ("wsj:SYNTHETIC-EMBEDDING", "article_text", "queued"),
            ("wsj:SYNTHETIC-EMBEDDING", "header_image", "not_applicable"),
        ]
        assert db.execute("SELECT articles FROM runs").fetchall() == [(2,)]

    resumed = run_embedding_pipeline(config, RecordingAdapter(), limit=2)

    assert resumed.succeeded == 2
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute("SELECT count(*) FROM embeddings").fetchone() == (2,)


def test_limited_run_does_not_remove_embedding_for_an_unselected_article(tmp_path):
    """Break caught: a later limited slice reconciles embeddings outside its scope."""

    config = write_generated_preprocessing_fixture(tmp_path)
    second_markdown = "# Second generated article\n"
    second_path = config.preprocessing_output_root / "text" / "second.md"
    second_path.write_text(second_markdown, encoding="utf-8")
    with Catalog.open(config.preprocessing_catalog) as catalog, catalog.transaction():
        catalog.replace_article(
            ArticleRecord(
                article_id="wsj:ZZZ-SYNTHETIC-EMBEDDING",
                source_html_path="synthetic/second.html",
                cleaned_markdown_path="text/second.md",
                header_image_path=None,
                inline_image_urls=(),
                published_at_utc=datetime(2024, 1, 3, 15, 30, tzinfo=UTC),
                publication_date_new_york=date(2024, 1, 3),
                cleaned_markdown_sha256=hashlib.sha256(
                    second_markdown.encode()
                ).hexdigest(),
            )
        )

    run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=2)
    with Catalog.open(config.preprocessing_catalog) as catalog, catalog.transaction():
        catalog.connection.execute(
            "DELETE FROM articles WHERE article_id = ?",
            ["wsj:ZZZ-SYNTHETIC-EMBEDDING"],
        )

    result = run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)

    assert result == EmbeddingRunResult(
        articles=1,
        embeddings=0,
        reused=1,
        header_absent=1,
    )
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        article_ids = db.execute(
            "SELECT article_id FROM embeddings ORDER BY article_id"
        ).fetchall()
    assert article_ids == [
        ("wsj:SYNTHETIC-EMBEDDING",),
        ("wsj:ZZZ-SYNTHETIC-EMBEDDING",),
    ]


def test_embedding_catalog_has_exact_version_eleven_image_frame_schema(
    tmp_path,
):
    database_path = tmp_path / "catalog.duckdb"

    with EmbeddingCatalog.open(database_path):
        pass

    with duckdb.connect(str(database_path), read_only=True) as connection:
        table_types = dict(
            connection.execute(
                """
                SELECT table_name, table_type
                FROM information_schema.tables
                WHERE table_catalog = current_database()
                  AND table_schema = 'main'
                """
            ).fetchall()
        )
        table_columns = {
            table: tuple(
                (row[1], row[2], row[3], row[4], row[5])
                for row in connection.execute(
                    f"PRAGMA table_info('{table}')"
                ).fetchall()
            )
            for table in EXPECTED_EMBEDDING_TABLE_COLUMNS
        }
        constraints = Counter(
            (
                row[0],
                row[1],
                tuple(row[2] or ()),
                row[3],
                row[4],
                tuple(row[5] or ()),
            )
            for row in connection.execute(
                """
                SELECT table_name, constraint_type, constraint_column_names,
                       expression, referenced_table, referenced_column_names
                FROM duckdb_constraints()
                WHERE database_name = current_database()
                  AND schema_name = 'main'
                """
            ).fetchall()
        )
        indexes = connection.execute(
            """
            SELECT table_name, index_name, is_unique, expressions
            FROM duckdb_indexes()
            ORDER BY table_name, index_name
            """
        ).fetchall()
        metadata = connection.execute(
            "SELECT key, value FROM metadata ORDER BY key"
        ).fetchall()

    expected_constraints = Counter(
        (table, "PRIMARY KEY", columns, None, None, ())
        for table, columns in EXPECTED_EMBEDDING_PRIMARY_KEYS
    )
    expected_constraints.update(
        (table, "NOT NULL", (name,), None, None, ())
        for table, columns in EXPECTED_EMBEDDING_TABLE_COLUMNS.items()
        for name, _data_type, not_null, _default, _primary_key in columns
        if not_null
    )
    assert table_types == {
        "article_text_aggregation_provenance": "BASE TABLE",
        "embedding_configurations": "BASE TABLE",
        "embedding_generation_history": "BASE TABLE",
        "image_input_provenance": "BASE TABLE",
        "embedding_work_items": "BASE TABLE",
        "embeddings": "BASE TABLE",
        "long_text_part_generations": "BASE TABLE",
        "long_text_parts": "BASE TABLE",
        "metadata": "BASE TABLE",
        "multimodal_embedding_provenance": "BASE TABLE",
        "runs": "BASE TABLE",
    }
    assert table_columns == EXPECTED_EMBEDDING_TABLE_COLUMNS
    assert constraints == expected_constraints
    assert indexes == []
    assert metadata == [("schema_version", "11")]


def test_pipeline_refuses_version_one_embedding_catalog_without_migration(tmp_path):
    """Break caught: opening old derived output silently upgrades it in place."""

    config = write_generated_preprocessing_fixture(tmp_path)
    config.embedding_output_root.mkdir()
    with duckdb.connect(str(config.embedding_catalog)) as db:
        db.execute(
            "CREATE TABLE metadata (key VARCHAR PRIMARY KEY, value VARCHAR NOT NULL)"
        )
        db.execute("INSERT INTO metadata VALUES ('schema_version', '1')")
    before = config.embedding_catalog.read_bytes()

    with pytest.raises(EmbeddingCatalogError, match="unsupported embedding catalog"):
        run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)

    assert config.embedding_catalog.read_bytes() == before
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute("SHOW TABLES").fetchall() == [("metadata",)]
        assert db.execute("SELECT * FROM metadata").fetchall() == [
            ("schema_version", "1")
        ]


def test_pipeline_refuses_version_two_embedding_catalog_without_migration(tmp_path):
    """Break caught: opening checkpoint output silently upgrades it in place."""

    config = write_generated_preprocessing_fixture(tmp_path)
    config.embedding_output_root.mkdir()
    with duckdb.connect(str(config.embedding_catalog)) as db:
        db.execute(
            "CREATE TABLE metadata (key VARCHAR PRIMARY KEY, value VARCHAR NOT NULL)"
        )
        db.execute("INSERT INTO metadata VALUES ('schema_version', '2')")
    before = config.embedding_catalog.read_bytes()

    with pytest.raises(EmbeddingCatalogError, match="unsupported embedding catalog"):
        run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)

    assert config.embedding_catalog.read_bytes() == before
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute("SHOW TABLES").fetchall() == [("metadata",)]
        assert db.execute("SELECT * FROM metadata").fetchall() == [
            ("schema_version", "2")
        ]


def test_pipeline_refuses_version_three_embedding_catalog_without_migration(tmp_path):
    """Break caught: image provenance columns are silently added in place."""

    config = write_generated_preprocessing_fixture(tmp_path)
    config.embedding_output_root.mkdir()
    with duckdb.connect(str(config.embedding_catalog)) as db:
        db.execute(
            "CREATE TABLE metadata (key VARCHAR PRIMARY KEY, value VARCHAR NOT NULL)"
        )
        db.execute("INSERT INTO metadata VALUES ('schema_version', '3')")
    before = config.embedding_catalog.read_bytes()

    with pytest.raises(EmbeddingCatalogError, match="unsupported embedding catalog"):
        run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)

    assert config.embedding_catalog.read_bytes() == before
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute("SHOW TABLES").fetchall() == [("metadata",)]
        assert db.execute("SELECT * FROM metadata").fetchall() == [
            ("schema_version", "3")
        ]


def test_pipeline_refuses_version_four_embedding_catalog_without_migration(tmp_path):
    """Break caught: immutable generation provenance is silently added in place."""

    config = write_generated_preprocessing_fixture(tmp_path)
    config.embedding_output_root.mkdir()
    with duckdb.connect(str(config.embedding_catalog)) as db:
        db.execute(
            "CREATE TABLE metadata (key VARCHAR PRIMARY KEY, value VARCHAR NOT NULL)"
        )
        db.execute("INSERT INTO metadata VALUES ('schema_version', '4')")
    before = config.embedding_catalog.read_bytes()

    with pytest.raises(EmbeddingCatalogError, match="unsupported embedding catalog"):
        run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)

    assert config.embedding_catalog.read_bytes() == before
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute("SHOW TABLES").fetchall() == [("metadata",)]
        assert db.execute("SELECT * FROM metadata").fetchall() == [
            ("schema_version", "4")
        ]


def test_pipeline_refuses_version_five_embedding_catalog_without_migration(tmp_path):
    """Break caught: header disposition counters are silently added in place."""

    config = write_generated_preprocessing_fixture(tmp_path)
    config.embedding_output_root.mkdir()
    with EmbeddingCatalog.open(config.embedding_catalog):
        pass
    with duckdb.connect(str(config.embedding_catalog)) as db:
        run_columns = {
            row[1] for row in db.execute("PRAGMA table_info('runs')").fetchall()
        }
        if "header_failed" in run_columns:
            db.execute("ALTER TABLE runs DROP COLUMN header_failed")
        if "header_absent" in run_columns:
            db.execute("ALTER TABLE runs DROP COLUMN header_absent")
        db.execute("UPDATE metadata SET value = '5' WHERE key = 'schema_version'")
    before = config.embedding_catalog.read_bytes()

    with pytest.raises(EmbeddingCatalogError, match="unsupported embedding catalog"):
        run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)

    assert config.embedding_catalog.read_bytes() == before
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone() == ("5",)
        assert {
            row[1] for row in db.execute("PRAGMA table_info('runs')").fetchall()
        }.isdisjoint({"header_absent", "header_failed"})


def test_pipeline_refuses_version_six_embedding_catalog_without_migration(tmp_path):
    """Break caught: multimodal provenance is silently added to derived output."""

    config = write_generated_preprocessing_fixture(tmp_path)
    config.embedding_output_root.mkdir()
    with EmbeddingCatalog.open(config.embedding_catalog):
        pass
    with duckdb.connect(str(config.embedding_catalog)) as db:
        if "multimodal_embedding_provenance" in {
            row[0] for row in db.execute("SHOW TABLES").fetchall()
        }:
            db.execute("DROP TABLE multimodal_embedding_provenance")
        db.execute("UPDATE metadata SET value = '6' WHERE key = 'schema_version'")
    before = config.embedding_catalog.read_bytes()

    with pytest.raises(EmbeddingCatalogError, match="unsupported embedding catalog"):
        run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)

    assert config.embedding_catalog.read_bytes() == before
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone() == ("6",)
        assert "multimodal_embedding_provenance" not in {
            row[0] for row in db.execute("SHOW TABLES").fetchall()
        }


def test_pipeline_refuses_version_seven_embedding_catalog_without_migration(tmp_path):
    """Break caught: durable long-text state is silently added in place."""

    config = write_generated_preprocessing_fixture(tmp_path)
    config.embedding_output_root.mkdir()
    with EmbeddingCatalog.open(config.embedding_catalog):
        pass
    with duckdb.connect(str(config.embedding_catalog)) as db:
        db.execute("DROP TABLE IF EXISTS article_text_aggregation_provenance")
        db.execute("DROP TABLE IF EXISTS long_text_parts")
        db.execute("UPDATE metadata SET value = '7' WHERE key = 'schema_version'")
    before = config.embedding_catalog.read_bytes()

    with pytest.raises(EmbeddingCatalogError, match="unsupported embedding catalog"):
        run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)

    assert config.embedding_catalog.read_bytes() == before
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone() == ("7",)
        assert {
            "article_text_aggregation_provenance",
            "long_text_parts",
        }.isdisjoint({row[0] for row in db.execute("SHOW TABLES").fetchall()})


def test_pipeline_refuses_exact_version_eight_catalog_without_migration(tmp_path):
    """Break caught: the real prior-v8 shape is silently upgraded in place."""

    config = write_generated_preprocessing_fixture(tmp_path)
    config.embedding_output_root.mkdir()
    with EmbeddingCatalog.open(config.embedding_catalog):
        pass
    with duckdb.connect(str(config.embedding_catalog)) as db:
        db.execute("DROP TABLE image_input_provenance")
        db.execute("DROP TABLE long_text_part_generations")
        for column in ("char_start", "char_end", "byte_start", "byte_end"):
            db.execute(f"ALTER TABLE long_text_parts DROP COLUMN {column}")
        for column in ("tokenizer_engine", "long_text_part_attempt_limit"):
            db.execute(
                f"ALTER TABLE embedding_configurations DROP COLUMN {column}"
            )
        expected_v8_columns = {
            table: columns
            for table, columns in EXPECTED_EMBEDDING_TABLE_COLUMNS.items()
            if table
            not in {"image_input_provenance", "long_text_part_generations"}
        }
        expected_v8_columns["embedding_configurations"] = tuple(
            column
            for column in expected_v8_columns["embedding_configurations"]
            if column[0]
            not in {"tokenizer_engine", "long_text_part_attempt_limit"}
        )
        expected_v8_columns["long_text_parts"] = tuple(
            column
            for column in expected_v8_columns["long_text_parts"]
            if column[0]
            not in {"char_start", "char_end", "byte_start", "byte_end"}
        )
        assert {row[0] for row in db.execute("SHOW TABLES").fetchall()} == (
            set(expected_v8_columns)
        )
        assert {
            table: tuple(
                (row[1], row[2], row[3], row[4], row[5])
                for row in db.execute(f"PRAGMA table_info('{table}')").fetchall()
            )
            for table in expected_v8_columns
        } == expected_v8_columns
        assert [
            row[1]
            for row in db.execute(
                "PRAGMA table_info('embedding_configurations')"
            ).fetchall()
        ] == [
            "configuration_id",
            "model",
            "observed_model",
            "observed_api_version",
            "task",
            "dimensions",
            "output_type",
            "normalization",
            "tokenizer_revision",
            "context_token_limit",
            "context_rules",
            "long_text_aggregation",
            "image_input_rules",
            "image_transform",
            "multimodal_formula",
            "client_configuration_version",
        ]
        assert [
            row[1]
            for row in db.execute("PRAGMA table_info('long_text_parts')").fetchall()
        ] == [
            "article_id",
            "configuration_id",
            "article_input_sha256",
            "part_index",
            "part_count",
            "part_input_sha256",
            "token_count",
            "state",
            "attempt_count",
            "error_code",
            "status_code",
            "retry_after_seconds",
            "last_run_id",
            "generation_run_id",
            "stored_vector_sha256",
            "vector",
            "updated_at",
        ]
        expected_v8_primary_keys = (
            EXPECTED_EMBEDDING_PRIMARY_KEYS
            - {
                (
                    "image_input_provenance",
                    ("article_id", "configuration_id", "generation_run_id"),
                ),
                (
                    "long_text_part_generations",
                    (
                        "article_id",
                        "configuration_id",
                        "article_input_sha256",
                        "part_index",
                        "generation_run_id",
                    ),
                )
            }
        )
        expected_v8_constraints = Counter(
            (table, "PRIMARY KEY", columns, None, None, ())
            for table, columns in expected_v8_primary_keys
        )
        expected_v8_constraints.update(
            (table, "NOT NULL", (name,), None, None, ())
            for table, columns in expected_v8_columns.items()
            for name, _data_type, not_null, _default, _primary_key in columns
            if not_null
        )
        assert Counter(
            (
                row[0],
                row[1],
                tuple(row[2] or ()),
                row[3],
                row[4],
                tuple(row[5] or ()),
            )
            for row in db.execute(
                """
                SELECT table_name, constraint_type, constraint_column_names,
                       expression, referenced_table, referenced_column_names
                FROM duckdb_constraints()
                WHERE database_name = current_database()
                  AND schema_name = 'main'
                """
            ).fetchall()
        ) == expected_v8_constraints
        assert db.execute("SELECT * FROM duckdb_indexes()").fetchall() == []
        db.execute("UPDATE metadata SET value = '8' WHERE key = 'schema_version'")
    before = config.embedding_catalog.read_bytes()

    with pytest.raises(EmbeddingCatalogError, match="unsupported embedding catalog"):
        run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)

    assert config.embedding_catalog.read_bytes() == before
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone() == ("8",)
        assert "long_text_part_generations" not in {
            row[0] for row in db.execute("SHOW TABLES").fetchall()
        }
        assert "image_input_provenance" not in {
            row[0] for row in db.execute("SHOW TABLES").fetchall()
        }
        assert {
            row[1]
            for row in db.execute("PRAGMA table_info('long_text_parts')").fetchall()
        }.isdisjoint({"char_start", "char_end", "byte_start", "byte_end"})
        assert {
            row[1]
            for row in db.execute(
                "PRAGMA table_info('embedding_configurations')"
            ).fetchall()
        }.isdisjoint({"tokenizer_engine", "long_text_part_attempt_limit"})


def test_pipeline_refuses_exact_version_nine_catalog_without_migration(tmp_path):
    """Break caught: image provenance is silently added to prior derived output."""

    config = write_generated_preprocessing_fixture(tmp_path)
    config.embedding_output_root.mkdir()
    with EmbeddingCatalog.open(config.embedding_catalog):
        pass
    with duckdb.connect(str(config.embedding_catalog)) as db:
        db.execute("DROP TABLE image_input_provenance")
        db.execute("UPDATE metadata SET value = '9' WHERE key = 'schema_version'")
    before = config.embedding_catalog.read_bytes()

    with pytest.raises(EmbeddingCatalogError, match="unsupported embedding catalog"):
        run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)

    assert config.embedding_catalog.read_bytes() == before
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone() == ("9",)
        assert "image_input_provenance" not in {
            row[0] for row in db.execute("SHOW TABLES").fetchall()
        }


def test_pipeline_refuses_exact_version_ten_catalog_without_migration(tmp_path):
    """Break caught: frame provenance is silently added to prior output."""

    config = write_generated_preprocessing_fixture(tmp_path)
    config.embedding_output_root.mkdir()
    with EmbeddingCatalog.open(config.embedding_catalog):
        pass
    with duckdb.connect(str(config.embedding_catalog)) as db:
        db.execute("ALTER TABLE image_input_provenance DROP source_frames")
        db.execute("ALTER TABLE image_input_provenance DROP embedded_frames")
        db.execute("UPDATE metadata SET value = '10' WHERE key = 'schema_version'")
    before = config.embedding_catalog.read_bytes()

    with pytest.raises(EmbeddingCatalogError, match="unsupported embedding catalog"):
        run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)

    assert config.embedding_catalog.read_bytes() == before
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone() == ("10",)
        assert {
            row[1]
            for row in db.execute(
                "PRAGMA table_info('image_input_provenance')"
            ).fetchall()
        }.isdisjoint({"source_frames", "embedded_frames"})


def test_pipeline_refuses_malformed_current_schema_with_version_eight_label(
    tmp_path,
):
    """Break caught: a false v8 label makes a malformed current shape acceptable."""

    config = write_generated_preprocessing_fixture(tmp_path)
    config.embedding_output_root.mkdir()
    with EmbeddingCatalog.open(config.embedding_catalog):
        pass
    with duckdb.connect(str(config.embedding_catalog)) as db:
        db.execute("UPDATE metadata SET value = '8' WHERE key = 'schema_version'")
    before = config.embedding_catalog.read_bytes()

    with pytest.raises(EmbeddingCatalogError, match="unsupported embedding catalog"):
        run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)

    assert config.embedding_catalog.read_bytes() == before
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone() == ("8",)
        assert "long_text_part_generations" in {
            row[0] for row in db.execute("SHOW TABLES").fetchall()
        }


def test_validator_accepts_fresh_synthetic_embedding_output(tmp_path):
    config = write_generated_preprocessing_fixture(tmp_path)

    run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)

    assert validate_embedding_outputs(config) == EmbeddingValidationResult(issues=())


@pytest.mark.parametrize("disposition", ("absent", "missing"))
def test_validator_reports_missing_header_disposition_checkpoint(
    tmp_path,
    disposition,
):
    """Break caught: deleting no-header/failed-header work escapes validation."""

    config = write_generated_preprocessing_fixture(tmp_path)
    if disposition == "missing":
        declare_missing_generated_header_image(config)
    run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)
    with duckdb.connect(str(config.embedding_catalog)) as connection:
        connection.execute(
            "DELETE FROM embedding_work_items WHERE modality = 'header_image'"
        )

    codes = {issue.code for issue in validate_embedding_outputs(config).issues}

    assert "missing_header_disposition" in codes
    assert "invalid_header_disposition_counts" in codes


@pytest.mark.parametrize(
    ("disposition", "counter_update"),
    (
        ("absent", "header_absent = 0, header_failed = 1"),
        ("missing", "header_absent = 1, header_failed = 0"),
    ),
)
def test_validator_rejects_header_disposition_counter_mismatch(
    tmp_path,
    disposition,
    counter_update,
):
    """Break caught: run claims can swap absent and failed header coverage."""

    config = write_generated_preprocessing_fixture(tmp_path)
    if disposition == "missing":
        declare_missing_generated_header_image(config)
    run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)
    with duckdb.connect(str(config.embedding_catalog)) as connection:
        connection.execute(f"UPDATE runs SET {counter_update}")

    codes = {issue.code for issue in validate_embedding_outputs(config).issues}

    assert "invalid_header_disposition_counts" in codes


def test_validator_reports_missing_vector_claimed_by_successful_run(tmp_path):
    config = write_generated_preprocessing_fixture(tmp_path)
    run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)
    with duckdb.connect(str(config.embedding_catalog)) as connection:
        connection.execute("DELETE FROM embeddings")

    assert validate_embedding_outputs(config) == EmbeddingValidationResult(
        issues=(
            EmbeddingValidationIssue(
                "missing_published_embedding",
                "successful embedding runs lack their published vectors",
            ),
        )
    )


def test_validator_reports_invalid_durable_work_state(tmp_path):
    """Break caught: corrupted checkpoint state is accepted as valid output."""

    config = write_generated_preprocessing_fixture(tmp_path)
    run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)
    with duckdb.connect(str(config.embedding_catalog)) as connection:
        connection.execute(
            "UPDATE embedding_work_items SET state = 'synthetic_invalid'"
        )

    assert validate_embedding_outputs(config) == EmbeddingValidationResult(
        issues=(
            EmbeddingValidationIssue(
                "invalid_work_state",
                "embedding work items use an unsupported lifecycle state",
            ),
        )
    )


@pytest.mark.parametrize("invalid_hash", (None, "not-a-sha256"))
def test_validator_rejects_missing_or_malformed_active_work_hash(
    tmp_path,
    invalid_hash,
):
    """Break caught: applicable work can lose its exact input identity."""

    config = write_generated_preprocessing_fixture(tmp_path)
    run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)
    with duckdb.connect(str(config.embedding_catalog)) as connection:
        connection.execute(
            """
            UPDATE embedding_work_items
            SET input_sha256 = ?
            WHERE modality = 'article_text'
            """,
            [invalid_hash],
        )

    codes = {issue.code for issue in validate_embedding_outputs(config).issues}

    assert "invalid_work_input_hash" in codes


@pytest.mark.parametrize(
    ("modality", "invalid_path"),
    (
        ("article_text", "synthetic/unexpected.png"),
        ("header_image", "../outside.png"),
        ("header_image", "/absolute.png"),
        ("header_image", "https://example.invalid/remote.png"),
    ),
)
def test_validator_rejects_invalid_work_and_vector_source_paths(
    tmp_path,
    modality,
    invalid_path,
):
    """Break caught: persisted source paths are ambiguous or remotely addressed."""

    config = write_generated_preprocessing_fixture(tmp_path)
    if modality == "header_image":
        attach_generated_header_image(config, b"generated validation image")
    run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)
    with duckdb.connect(str(config.embedding_catalog)) as connection:
        connection.execute(
            """
            UPDATE embedding_work_items
            SET source_relative_path = ?
            WHERE modality = ?
            """,
            [invalid_path, modality],
        )
        connection.execute(
            """
            UPDATE embeddings
            SET source_relative_path = ?
            WHERE modality = ?
            """,
            [invalid_path, modality],
        )

    codes = {issue.code for issue in validate_embedding_outputs(config).issues}

    assert "invalid_source_relative_path" in codes


def test_validator_rejects_vector_for_not_applicable_header_image(tmp_path):
    """Break caught: no-header work can retain a queryable placeholder vector."""

    config = write_generated_preprocessing_fixture(tmp_path)
    run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)
    configuration_identifier = configuration_id(FakeEmbeddingAdapter.profile)
    vector = [1.0, *([0.0] * 2047)]
    vector_hash = hashlib.sha256(
        b"".join(struct.pack("<f", value) for value in vector)
    ).hexdigest()
    with duckdb.connect(str(config.embedding_catalog)) as connection:
        published_at_utc, publication_date = connection.execute(
            """
            SELECT published_at_utc, publication_date_new_york
            FROM embeddings
            WHERE modality = 'article_text'
            """
        ).fetchone()
        connection.execute(
            """
            INSERT INTO embeddings
            VALUES ('wsj:SYNTHETIC-EMBEDDING', 'header_image', ?, NULL,
                    ?, ?, 2048, ?, ?, ?)
            """,
            [
                configuration_identifier,
                published_at_utc,
                publication_date,
                hashlib.sha256(b"no applicable input").hexdigest(),
                vector_hash,
                vector,
            ],
        )

    codes = {issue.code for issue in validate_embedding_outputs(config).issues}

    assert "embedding_without_success_checkpoint" in codes


def test_validator_rejects_generation_metadata_on_not_applicable_image(tmp_path):
    """Break caught: absent image work retains attempt/failure/generation fields."""

    config = write_generated_preprocessing_fixture(tmp_path)
    run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)
    with duckdb.connect(str(config.embedding_catalog)) as connection:
        run_id = connection.execute("SELECT run_id FROM runs").fetchone()[0]
        connection.execute(
            """
            UPDATE embedding_work_items
            SET source_relative_path = '../outside.png', input_sha256 = ?,
                attempt_count = 1, error_code = 'synthetic_failure',
                generation_run_id = ?
            WHERE modality = 'header_image'
            """,
            [hashlib.sha256(b"synthetic").hexdigest(), run_id],
        )

    codes = {issue.code for issue in validate_embedding_outputs(config).issues}

    assert "invalid_not_applicable_checkpoint" in codes


@pytest.mark.parametrize("generation_run_id", (None, "missing-run"))
def test_validator_rejects_success_without_valid_generation_run(
    tmp_path,
    generation_run_id,
):
    """Break caught: a successful active vector loses immutable provenance."""

    config = write_generated_preprocessing_fixture(tmp_path)
    run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)
    with duckdb.connect(str(config.embedding_catalog)) as connection:
        connection.execute(
            """
            UPDATE embedding_work_items
            SET generation_run_id = ?
            WHERE modality = 'article_text'
            """,
            [generation_run_id],
        )

    codes = {issue.code for issue in validate_embedding_outputs(config).issues}

    assert "invalid_generation_checkpoint" in codes


@pytest.mark.parametrize(
    ("history_modality", "history_path", "expected_code"),
    (
        ("remote_image", "synthetic/header.png", "invalid_generation_modality"),
        ("header_image", "../outside.png", "invalid_generation_source_path"),
        ("article_text", "synthetic/unexpected.png", "invalid_generation_source_path"),
    ),
)
def test_validator_rejects_malformed_generation_modality_and_path(
    tmp_path,
    history_modality,
    history_path,
    expected_code,
):
    """Break caught: superseded provenance accepts impossible source identities."""

    config = write_generated_preprocessing_fixture(tmp_path)
    if history_modality == "article_text":
        run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)
        run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1, reprocess=True)
    else:
        relative_path = attach_generated_header_image(
            config,
            b"generated history image one",
        )
        run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)
        (config.source_root / relative_path).write_bytes(
            b"generated history image two"
        )
        run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)
    with duckdb.connect(str(config.embedding_catalog)) as connection:
        connection.execute(
            """
            UPDATE embedding_generation_history
            SET modality = ?, source_relative_path = ?
            WHERE modality = ?
            """,
            [
                history_modality,
                history_path,
                (
                    "article_text"
                    if history_modality == "article_text"
                    else "header_image"
                ),
            ],
        )

    codes = {issue.code for issue in validate_embedding_outputs(config).issues}

    assert expected_code in codes


def test_pipeline_refuses_unknown_existing_embedding_schema(tmp_path):
    config = write_generated_preprocessing_fixture(tmp_path)
    config.embedding_output_root.mkdir()
    with duckdb.connect(str(config.embedding_catalog)) as db:
        db.execute("CREATE TABLE legacy_vectors(value INTEGER)")

    with pytest.raises(EmbeddingCatalogError, match="unsupported embedding catalog"):
        run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)


def test_pipeline_validates_existing_embedding_catalog_for_empty_article_slice(
    tmp_path,
):
    config = write_generated_preprocessing_fixture(tmp_path)
    with Catalog.open(config.preprocessing_catalog) as catalog, catalog.transaction():
        catalog.connection.execute("DELETE FROM articles")
    config.embedding_output_root.mkdir()
    with duckdb.connect(str(config.embedding_catalog)) as db:
        db.execute("CREATE TABLE legacy_vectors(value INTEGER)")

    with pytest.raises(EmbeddingCatalogError, match="unsupported embedding catalog"):
        run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)


def test_pipeline_refuses_embedding_catalog_with_unexpected_index(tmp_path):
    config = write_generated_preprocessing_fixture(tmp_path)
    run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)
    with duckdb.connect(str(config.embedding_catalog)) as db:
        db.execute(
            "CREATE UNIQUE INDEX unexpected_embedding_index "
            "ON embedding_configurations(model)"
        )

    with pytest.raises(EmbeddingCatalogError, match="unsupported embedding catalog"):
        run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)


@pytest.mark.parametrize(
    "damage",
    (
        "ALTER TABLE runs DROP COLUMN embeddings",
        "ALTER TABLE embedding_configurations ALTER COLUMN model DROP NOT NULL",
        "CREATE OR REPLACE TABLE embeddings AS SELECT * FROM embeddings",
        "UPDATE metadata SET value = '999' WHERE key = 'schema_version'",
    ),
    ids=("missing-column", "changed-nullability", "missing-primary-key", "version"),
)
def test_pipeline_refuses_damaged_embedding_schema(tmp_path, damage):
    config = write_generated_preprocessing_fixture(tmp_path)
    run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)
    with duckdb.connect(str(config.embedding_catalog)) as db:
        db.execute(damage)

    with pytest.raises(EmbeddingCatalogError, match="unsupported embedding catalog"):
        run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)


def test_existing_embedding_catalog_is_inspected_read_only_before_write(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "catalog.duckdb"
    with EmbeddingCatalog.open(database_path):
        pass
    real_connect = embedding_catalog_module.duckdb.connect
    open_modes = []

    def track_connect(*args, **kwargs):
        open_modes.append(bool(kwargs.get("read_only", False)))
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(embedding_catalog_module.duckdb, "connect", track_connect)

    with EmbeddingCatalog.open(database_path):
        pass

    assert open_modes == [True, False]


def test_damaged_embedding_catalog_fails_before_writable_open(tmp_path, monkeypatch):
    database_path = tmp_path / "catalog.duckdb"
    with EmbeddingCatalog.open(database_path):
        pass
    with duckdb.connect(str(database_path)) as connection:
        connection.execute("ALTER TABLE runs DROP COLUMN embeddings")
    real_connect = embedding_catalog_module.duckdb.connect
    open_modes = []

    def track_connect(*args, **kwargs):
        open_modes.append(bool(kwargs.get("read_only", False)))
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(embedding_catalog_module.duckdb, "connect", track_connect)

    with (
        pytest.raises(EmbeddingCatalogError, match="unsupported embedding catalog"),
        EmbeddingCatalog.open(database_path),
    ):
        pass

    assert open_modes == [True]


def test_embedding_catalog_rejects_in_place_schema_damage_at_writable_open(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "catalog.duckdb"
    with EmbeddingCatalog.open(database_path):
        pass
    real_connect = embedding_catalog_module.duckdb.connect
    inspected_read_only = False
    mutated = False

    def mutate_at_writable_open(*args, **kwargs):
        nonlocal inspected_read_only, mutated
        if kwargs.get("read_only", False):
            inspected_read_only = True
        elif not mutated:
            assert inspected_read_only
            with real_connect(str(database_path)) as competing_connection:
                competing_connection.execute("ALTER TABLE runs DROP COLUMN embeddings")
            mutated = True
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(
        embedding_catalog_module.duckdb,
        "connect",
        mutate_at_writable_open,
    )

    with (
        pytest.raises(EmbeddingCatalogError, match="unsupported embedding catalog"),
        EmbeddingCatalog.open(database_path),
    ):
        pass

    assert mutated


def test_embedding_catalog_rejects_leaf_replaced_after_read_only_inspection(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "catalog.duckdb"
    replacement_path = tmp_path / "replacement.duckdb"
    with EmbeddingCatalog.open(database_path):
        pass
    with EmbeddingCatalog.open(replacement_path):
        pass
    real_open_descriptor = embedding_catalog_module._open_catalog_leaf_descriptor
    inspected_read_only = False

    def replace_before_writable_open(
        parent_descriptor,
        leaf_name,
        identity,
        *,
        read_only,
    ):
        nonlocal inspected_read_only
        if read_only:
            inspected_read_only = True
        if not read_only:
            assert inspected_read_only
            database_path.unlink()
            database_path.symlink_to(replacement_path)
        return real_open_descriptor(
            parent_descriptor,
            leaf_name,
            identity,
            read_only=read_only,
        )

    monkeypatch.setattr(
        embedding_catalog_module,
        "_open_catalog_leaf_descriptor",
        replace_before_writable_open,
    )

    with (
        pytest.raises(EmbeddingCatalogError, match="unsafe embedding catalog path"),
        EmbeddingCatalog.open(database_path),
    ):
        pass


def test_pipeline_refuses_malformed_preprocessing_catalog(tmp_path):
    source_root = tmp_path / "source"
    preprocessing_output_root = tmp_path / "preprocessed"
    source_root.mkdir()
    preprocessing_output_root.mkdir()
    with duckdb.connect(str(preprocessing_output_root / "catalog.duckdb")) as db:
        db.execute("CREATE TABLE articles(article_id VARCHAR PRIMARY KEY)")
    config = EmbeddingPipelineConfig(
        source_root=source_root,
        preprocessing_output_root=preprocessing_output_root,
        embedding_output_root=tmp_path / "embeddings",
        hosted_processing_authorized=True,
    )

    with pytest.raises(EmbeddingPipelineError, match="preprocessing_catalog"):
        run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)


def test_embedding_pipeline_lock_is_exclusive_and_owned_lock_is_cleaned(tmp_path):
    lock_path = tmp_path / "embeddings" / "pipeline.lock"

    with embedding_pipeline_lock(lock_path):
        assert lock_path.is_file()
        with (
            pytest.raises(EmbeddingPipelineLockedError, match="already running"),
            embedding_pipeline_lock(lock_path),
        ):
            raise AssertionError("unreachable")
        assert lock_path.is_file()

    assert not lock_path.exists()


def test_embedding_pipeline_refuses_replaced_output_root_without_writing_through_it(
    tmp_path,
):
    config = write_generated_preprocessing_fixture(tmp_path)
    outside_root = tmp_path / "outside"
    outside_root.mkdir()
    config.embedding_output_root.symlink_to(outside_root, target_is_directory=True)

    with pytest.raises(RuntimeError, match="unsafe embedding output root"):
        run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)

    assert not (outside_root / "pipeline.lock").exists()
    assert not (outside_root / "catalog.duckdb").exists()


def test_embedding_pipeline_lock_does_not_remove_replacement(tmp_path):
    lock_path = tmp_path / "embeddings" / "pipeline.lock"

    with embedding_pipeline_lock(lock_path):
        lock_path.unlink()
        lock_path.write_text("replacement\n", encoding="utf-8")

    assert lock_path.read_text(encoding="utf-8") == "replacement\n"


def test_embedding_pipeline_lock_removes_owned_path_without_removing_hardlink(tmp_path):
    lock_path = tmp_path / "embeddings" / "pipeline.lock"
    linked_path = tmp_path / "embeddings" / "linked.lock"

    with embedding_pipeline_lock(lock_path):
        linked_path.hardlink_to(lock_path)

    assert not lock_path.exists()
    assert linked_path.is_file()


def test_embedding_catalog_stays_with_anchored_output_after_path_replacement(tmp_path):
    output_root = tmp_path / "embeddings"
    original_root = tmp_path / "original-embeddings"
    outside_root = tmp_path / "outside"
    lock_path = output_root / "pipeline.lock"

    with embedding_pipeline_lock(lock_path) as output_descriptor:
        output_root.rename(original_root)
        outside_root.mkdir()
        output_root.symlink_to(outside_root, target_is_directory=True)
        with EmbeddingCatalog.open_in(output_descriptor):
            pass

    assert (original_root / "catalog.duckdb").is_file()
    assert not (original_root / "pipeline.lock").exists()
    assert not (outside_root / "catalog.duckdb").exists()
    assert not (outside_root / "pipeline.lock").exists()


def test_embedding_pipeline_lock_cleans_owned_lock_after_exception(tmp_path):
    lock_path = tmp_path / "embeddings" / "pipeline.lock"

    with (
        pytest.raises(RuntimeError, match="synthetic failure"),
        embedding_pipeline_lock(lock_path),
    ):
        raise RuntimeError("synthetic failure")

    assert not lock_path.exists()


def test_validator_refuses_malformed_preprocessing_catalog(tmp_path):
    config = write_generated_preprocessing_fixture(tmp_path)
    run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)
    with duckdb.connect(str(config.preprocessing_catalog)) as db:
        db.execute("DROP TABLE runs")

    assert validate_embedding_outputs(config) == EmbeddingValidationResult(
        issues=(
            EmbeddingValidationIssue(
                "invalid_preprocessing_catalog",
                "canonical preprocessing catalog could not be queried",
            ),
        )
    )


def test_pipeline_refuses_symlinked_embedding_catalog(tmp_path):
    config = write_generated_preprocessing_fixture(tmp_path)
    outside_catalog = tmp_path / "outside.duckdb"
    with EmbeddingCatalog.open(outside_catalog):
        pass
    config.embedding_output_root.mkdir()
    config.embedding_catalog.symlink_to(outside_catalog)

    with pytest.raises(EmbeddingCatalogError, match="unsafe embedding catalog path"):
        run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)


def test_validator_reports_changed_canonical_markdown_without_text(tmp_path):
    config = write_generated_preprocessing_fixture(tmp_path)
    run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)
    markdown_path = (
        config.preprocessing_output_root / "text" / "2024" / "01" / "02" / "article.md"
    )
    markdown_path.write_text("# changed\n", encoding="utf-8")

    result = validate_embedding_outputs(config)

    assert result == EmbeddingValidationResult(
        issues=(
            EmbeddingValidationIssue(
                "input_hash_mismatch",
                "embedding input hash differs from canonical Markdown",
            ),
        )
    )


@pytest.mark.parametrize("leaf_kind", ("symlink", "hardlink", "directory"))
def test_pipeline_refuses_unsafe_canonical_markdown_leaf(tmp_path, leaf_kind):
    config = write_generated_preprocessing_fixture(tmp_path)
    replace_markdown_with_unsafe_leaf(config, leaf_kind)

    with pytest.raises(EmbeddingPipelineError) as raised:
        run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)

    assert raised.value.code == "unsafe_markdown_path"
    assert "synthetic replacement" not in str(raised.value)


@pytest.mark.parametrize("leaf_kind", ("symlink", "hardlink", "directory"))
def test_validator_reports_unsafe_canonical_markdown_leaf(tmp_path, leaf_kind):
    config = write_generated_preprocessing_fixture(tmp_path)
    run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)
    replace_markdown_with_unsafe_leaf(config, leaf_kind)

    assert validate_embedding_outputs(config) == EmbeddingValidationResult(
        issues=(
            EmbeddingValidationIssue(
                "unsafe_markdown_path",
                "canonical Markdown path is unsafe",
            ),
        )
    )


def test_pipeline_detects_canonical_markdown_replacement_after_safe_open(
    tmp_path,
    monkeypatch,
):
    config = write_generated_preprocessing_fixture(tmp_path)
    markdown_path = (
        config.preprocessing_output_root / "text" / "2024" / "01" / "02" / "article.md"
    )
    real_open = canonical_markdown_module.open_relative_regular
    swapped = False

    def swap_after_open(root, relative_value):
        nonlocal swapped
        status, descriptor = real_open(root, relative_value)
        if status == "ok" and not swapped:
            swapped = True
            markdown_path.unlink()
            markdown_path.write_text("# synthetic replacement\n", encoding="utf-8")
        return status, descriptor

    monkeypatch.setattr(
        canonical_markdown_module,
        "open_relative_regular",
        swap_after_open,
    )

    with pytest.raises(EmbeddingPipelineError) as raised:
        run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)

    assert raised.value.code == "unsafe_markdown_path"


def test_validator_reports_vector_and_publication_metadata_corruption(tmp_path):
    config = write_generated_preprocessing_fixture(tmp_path)
    run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)
    with duckdb.connect(str(config.embedding_catalog)) as db:
        db.execute(
            """
            UPDATE embeddings
            SET published_at_utc = ?, vector = ?
            """,
            [datetime(2024, 1, 3, 15, 30, tzinfo=UTC), [0.0] * 2048],
        )

    result = validate_embedding_outputs(config)

    assert result == EmbeddingValidationResult(
        issues=(
            EmbeddingValidationIssue(
                "publication_metadata_mismatch",
                "embedding publication metadata differs from the canonical article",
            ),
            EmbeddingValidationIssue(
                "vector_hash_mismatch",
                "embedding vector hashes do not match float32 values",
            ),
            EmbeddingValidationIssue(
                "zero_vector",
                "embedding rows contain zero vectors",
            ),
        )
    )


def test_validator_reports_run_without_configuration_reference(tmp_path):
    config = write_generated_preprocessing_fixture(tmp_path)
    run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)
    with duckdb.connect(str(config.embedding_catalog)) as db:
        db.execute("UPDATE runs SET configuration_id = 'missing'")

    result = validate_embedding_outputs(config)

    assert result == EmbeddingValidationResult(
        issues=(
            EmbeddingValidationIssue(
                "invalid_generation_checkpoint",
                "successful work lacks its generating run identity",
            ),
            EmbeddingValidationIssue(
                "missing_run_configuration_reference",
                "embedding runs lack a configuration reference",
            ),
        )
    )


def test_validator_checks_intrinsic_properties_for_orphaned_embedding_rows(tmp_path):
    config = write_generated_preprocessing_fixture(tmp_path)
    run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)
    with duckdb.connect(str(config.embedding_catalog)) as db:
        db.execute(
            "UPDATE embeddings SET article_id = 'orphan', modality = 'unsupported'"
        )

    result = validate_embedding_outputs(config)

    assert result == EmbeddingValidationResult(
        issues=(
            EmbeddingValidationIssue(
                "invalid_modality",
                "embedding rows use an unsupported modality",
            ),
                EmbeddingValidationIssue(
                    "missing_canonical_article",
                    "embedding rows lack a canonical article",
                ),
                EmbeddingValidationIssue(
                    "missing_published_embedding",
                    "successful embedding runs lack their published vectors",
                ),
            )
        )
