from __future__ import annotations

import hashlib
import struct
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import pytest

import wsj_embeddings.canonical_markdown as canonical_markdown_module
import wsj_embeddings.catalog as embedding_catalog_module
import wsj_embeddings.pipeline as embedding_pipeline_module
from wsj_embeddings import (
    EmbeddingCatalogError,
    EmbeddingPipelineConfig,
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
        ("context_token_limit", "INTEGER", True, None, False),
        ("context_rules", "VARCHAR", True, None, False),
        ("long_text_aggregation", "VARCHAR", True, None, False),
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
        ("started_at", "TIMESTAMP WITH TIME ZONE", True, None, False),
    ),
    "embedding_work_items": (
        ("article_id", "VARCHAR", True, None, True),
        ("modality", "VARCHAR", True, None, True),
        ("configuration_id", "VARCHAR", True, None, True),
        ("input_sha256", "VARCHAR", True, None, False),
        ("state", "VARCHAR", True, None, False),
        ("attempt_count", "INTEGER", True, None, False),
        ("error_code", "VARCHAR", False, None, False),
        ("status_code", "INTEGER", False, None, False),
        ("retry_after_seconds", "DOUBLE", False, None, False),
        ("last_run_id", "VARCHAR", True, None, False),
        ("updated_at", "TIMESTAMP WITH TIME ZONE", True, None, False),
    ),
    "embeddings": (
        ("article_id", "VARCHAR", True, None, True),
        ("modality", "VARCHAR", True, None, True),
        ("configuration_id", "VARCHAR", True, None, True),
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
        ("input_sha256", "VARCHAR", True, None, False),
        ("stored_vector_sha256", "VARCHAR", True, None, False),
        ("superseded_run_id", "VARCHAR", True, None, False),
        ("superseded_reason", "VARCHAR", True, None, False),
        ("superseded_at", "TIMESTAMP WITH TIME ZONE", True, None, False),
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


def seed_generated_modality_success(
    config: EmbeddingPipelineConfig,
    *,
    article_id: str,
    modality: str,
    profile: EmbeddingProfile,
) -> None:
    """Seed one content-free modality generation through generated fixture SQL."""

    configuration_identifier = configuration_id(profile)
    vector = [1.0, *(0.0 for _ in range(2047))]
    vector_sha256 = hashlib.sha256(
        b"".join(struct.pack("<f", value) for value in vector)
    ).hexdigest()
    input_sha256 = hashlib.sha256(
        f"{article_id}:{modality}:{configuration_identifier}".encode()
    ).hexdigest()
    with duckdb.connect(str(config.embedding_catalog)) as db:
        generation_run_id = db.execute(
            """
            SELECT run_id FROM runs
            WHERE configuration_id = ?
            ORDER BY started_at
            LIMIT 1
            """,
            [configuration_identifier],
        ).fetchone()[0]
        published_at_utc, publication_date_new_york = db.execute(
            """
            SELECT published_at_utc, publication_date_new_york
            FROM embeddings
            WHERE article_id = ? AND modality = 'article_text'
              AND configuration_id = ?
            """,
            [article_id, configuration_identifier],
        ).fetchone()
        db.execute(
            """
            INSERT INTO embedding_work_items
            VALUES (?, ?, ?, ?, 'succeeded', 1, NULL, NULL, NULL, ?, current_timestamp)
            """,
            [
                article_id,
                modality,
                configuration_identifier,
                input_sha256,
                generation_run_id,
            ],
        )
        db.execute(
            """
            INSERT INTO embeddings
            VALUES (?, ?, ?, ?, ?, 2048, ?, ?, ?)
            """,
            [
                article_id,
                modality,
                configuration_identifier,
                published_at_utc,
                publication_date_new_york,
                input_sha256,
                vector_sha256,
                vector,
            ],
        )


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


class PriorConfigurationAdapter(FakeEmbeddingAdapter):
    profile = EmbeddingProfile(
        model="synthetic-prior-model",
        task="retrieval.passage",
        dimensions=2048,
    )


def test_configuration_identity_covers_every_meaning_bearing_profile_field():
    """Break caught: a public embedding rule changes without new eligibility."""

    profile = FakeEmbeddingAdapter.profile
    assert profile.long_text_aggregation == "single-input-provider-pooling-v1"
    assert (
        JinaEmbeddingAdapter.profile.long_text_aggregation
        == "single-input-provider-pooling-v1"
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
        "context_token_limit": 8192,
        "context_rules": "changed-context-rules",
        "long_text_aggregation": "changed-aggregation",
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
        vector_type = db.execute("PRAGMA table_info('embeddings')").fetchall()[8][2]
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
    run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=2)
    run_embedding_pipeline(config, PriorConfigurationAdapter(), limit=2)
    current_profile = FakeEmbeddingAdapter.profile
    prior_profile = PriorConfigurationAdapter.profile
    seed_generated_modality_success(
        config,
        article_id=target_id,
        modality="header_image",
        profile=current_profile,
    )
    seed_generated_modality_success(
        config,
        article_id=target_id,
        modality="multimodal_article",
        profile=current_profile,
    )
    seed_generated_modality_success(
        config,
        article_id=sibling_id,
        modality="multimodal_article",
        profile=current_profile,
    )
    seed_generated_modality_success(
        config,
        article_id=target_id,
        modality="multimodal_article",
        profile=prior_profile,
    )
    if not reprocess:
        replace_generated_embedding_markdown(config, "# Changed generated article\n")

    run_embedding_pipeline(
        config,
        FakeEmbeddingAdapter(),
        limit=1,
        reprocess=reprocess,
    )

    current_id = configuration_id(current_profile)
    prior_id = configuration_id(prior_profile)
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        assert db.execute(
            """
            SELECT article_id, modality, configuration_id
            FROM embeddings
            WHERE modality != 'article_text'
            ORDER BY article_id, modality, configuration_id
            """
        ).fetchall() == sorted(
            [
                (target_id, "header_image", current_id),
                (target_id, "multimodal_article", prior_id),
                (sibling_id, "multimodal_article", current_id),
            ]
        )
        assert db.execute(
            """
            SELECT state, attempt_count, error_code, status_code,
                   retry_after_seconds
            FROM embedding_work_items
            WHERE article_id = ? AND modality = 'multimodal_article'
              AND configuration_id = ?
            """,
            [target_id, current_id],
        ).fetchone() == ("queued", 0, None, None, None)
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
            "SELECT article_id, state FROM embedding_work_items"
        ).fetchall() == [("wsj:SYNTHETIC-EMBEDDING", "queued")]
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
    )
    with duckdb.connect(str(config.embedding_catalog), read_only=True) as db:
        article_ids = db.execute(
            "SELECT article_id FROM embeddings ORDER BY article_id"
        ).fetchall()
    assert article_ids == [
        ("wsj:SYNTHETIC-EMBEDDING",),
        ("wsj:ZZZ-SYNTHETIC-EMBEDDING",),
    ]


def test_embedding_catalog_has_exact_version_three_generation_schema(tmp_path):
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
        "embedding_configurations": "BASE TABLE",
        "embedding_generation_history": "BASE TABLE",
        "embedding_work_items": "BASE TABLE",
        "embeddings": "BASE TABLE",
        "metadata": "BASE TABLE",
        "runs": "BASE TABLE",
    }
    assert table_columns == EXPECTED_EMBEDDING_TABLE_COLUMNS
    assert constraints == expected_constraints
    assert indexes == []
    assert metadata == [("schema_version", "3")]


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


def test_validator_accepts_fresh_synthetic_embedding_output(tmp_path):
    config = write_generated_preprocessing_fixture(tmp_path)

    run_embedding_pipeline(config, FakeEmbeddingAdapter(), limit=1)

    assert validate_embedding_outputs(config) == EmbeddingValidationResult(issues=())


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
