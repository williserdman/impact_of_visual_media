from __future__ import annotations

import hashlib
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import pytest

import wsj_embeddings.canonical_markdown as canonical_markdown_module
import wsj_embeddings.catalog as embedding_catalog_module
from wsj_embeddings import (
    EmbeddingCatalogError,
    EmbeddingPipelineConfig,
    EmbeddingRunResult,
    EmbeddingValidationIssue,
    EmbeddingValidationResult,
    FakeEmbeddingAdapter,
    run_embedding_pipeline,
    validate_embedding_outputs,
)
from wsj_embeddings.catalog import EmbeddingCatalog
from wsj_embeddings.pipeline import (
    EmbeddingPipelineError,
    EmbeddingPipelineLockedError,
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
        ("task", "VARCHAR", True, None, False),
        ("dimensions", "INTEGER", True, None, False),
        ("output_type", "VARCHAR", True, None, False),
        ("normalization", "VARCHAR", True, None, False),
    ),
    "runs": (
        ("run_id", "VARCHAR", True, None, True),
        ("configuration_id", "VARCHAR", True, None, False),
        ("articles", "INTEGER", True, None, False),
        ("embeddings", "INTEGER", True, None, False),
        ("started_at", "TIMESTAMP WITH TIME ZONE", True, None, False),
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
}
EXPECTED_EMBEDDING_PRIMARY_KEYS = {
    ("metadata", ("key",)),
    ("embedding_configurations", ("configuration_id",)),
    ("runs", ("run_id",)),
    ("embeddings", ("article_id", "modality", "configuration_id")),
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
    )


def snapshot_fixture_inputs(config: EmbeddingPipelineConfig) -> dict[str, str]:
    roots = (config.source_root, config.preprocessing_output_root)
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for root in roots
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


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

    assert result == EmbeddingRunResult(articles=1, embeddings=1)
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


def test_embedding_catalog_has_exact_version_one_schema(tmp_path):
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
        "embeddings": "BASE TABLE",
        "metadata": "BASE TABLE",
        "runs": "BASE TABLE",
    }
    assert table_columns == EXPECTED_EMBEDDING_TABLE_COLUMNS
    assert constraints == expected_constraints
    assert indexes == []
    assert metadata == [("schema_version", "1")]


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
        )
    )
