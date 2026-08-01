from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import pytest

from wsj_pipeline.catalog import (
    ArticleRecord,
    Catalog,
    CatalogError,
    DuplicateRecord,
)
from wsj_pipeline.models import ExtractedArticle, SourceCandidate

EXPECTED_TABLES = {
    "articles",
    "candidates",
    "duplicates",
    "failures",
    "metadata",
    "runs",
    "source_manifest",
}

EXPECTED_ARTICLE_TYPES = {
    "article_id": "VARCHAR",
    "source_html_path": "VARCHAR",
    "cleaned_markdown_path": "VARCHAR",
    "header_image_path": "VARCHAR",
    "inline_image_urls": "VARCHAR[]",
    "published_at_utc": "TIMESTAMP WITH TIME ZONE",
    "publication_date_new_york": "DATE",
    "cleaned_markdown_sha256": "VARCHAR",
}


def _candidate(
    *,
    html_size: int = 123,
    html_mtime_ns: int = 456,
    header_image_path: str | None = "2024/story_main_image.jpg",
) -> SourceCandidate:
    return SourceCandidate(
        relative_html_path="2024/story.html",
        html_size=html_size,
        html_mtime_ns=html_mtime_ns,
        relative_header_image_path=header_image_path,
        header_image_size=789 if header_image_path else None,
        header_image_mtime_ns=101112 if header_image_path else None,
    )


def _article() -> ExtractedArticle:
    return ExtractedArticle(
        article_id="wsj:synthetic",
        relative_html_path="2024/story.html",
        source_html_sha256="a" * 64,
        extractor_version="3",
        canonical_url="https://example.test/synthetic",
        published_at_utc=datetime(2024, 1, 2, 4, 5, tzinfo=UTC),
        publication_date_new_york=date(2024, 1, 1),
        updated_at_utc=None,
        markdown_text="# Synthetic headline\n\nFixture paragraph.",
        editorial_character_count=37,
        editorial_block_count=2,
        metadata_completeness=4,
        relative_header_image_path="2024/story_main_image.jpg",
        inline_image_urls=("https://images.example.test/inline.jpg",),
        warnings=(),
    )


def _article_record() -> ArticleRecord:
    article = _article()
    return ArticleRecord(
        article_id=article.article_id,
        source_html_path=article.relative_html_path,
        cleaned_markdown_path="text/2024/01/01/synthetic.md",
        header_image_path=article.relative_header_image_path,
        inline_image_urls=article.inline_image_urls,
        published_at_utc=article.published_at_utc,
        publication_date_new_york=article.publication_date_new_york,
        cleaned_markdown_sha256="b" * 64,
    )


def test_open_creates_the_single_versioned_catalog_schema(tmp_path: Path) -> None:
    database_path = tmp_path / "nested" / "catalog.duckdb"

    with Catalog.open(database_path):
        pass
    with Catalog.open(database_path):
        pass

    with duckdb.connect(str(database_path), read_only=True) as connection:
        tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
        schema_version = connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()[0]
        article_types = {
            row[1]: row[2]
            for row in connection.execute("PRAGMA table_info('articles')").fetchall()
        }
        indexes = connection.execute(
            "SELECT index_name, expressions FROM duckdb_indexes()"
        ).fetchall()

    assert tables == EXPECTED_TABLES
    assert schema_version == "1"
    assert article_types == EXPECTED_ARTICLE_TYPES
    assert indexes == [
        (
            "articles_publication_date_idx",
            "[publication_date_new_york, published_at_utc, article_id]",
        )
    ]


def test_article_paths_are_unique(tmp_path: Path) -> None:
    first = _article_record()
    same_source = ArticleRecord(
        article_id="wsj:other-source",
        source_html_path=first.source_html_path,
        cleaned_markdown_path="text/2024/01/01/other-source.md",
        header_image_path=None,
        inline_image_urls=(),
        published_at_utc=first.published_at_utc,
        publication_date_new_york=first.publication_date_new_york,
        cleaned_markdown_sha256="c" * 64,
    )
    same_markdown = ArticleRecord(
        article_id="wsj:other-markdown",
        source_html_path="2024/other.html",
        cleaned_markdown_path=first.cleaned_markdown_path,
        header_image_path=None,
        inline_image_urls=(),
        published_at_utc=first.published_at_utc,
        publication_date_new_york=first.publication_date_new_york,
        cleaned_markdown_sha256="d" * 64,
    )

    with Catalog.open(tmp_path / "catalog.duckdb") as catalog:
        catalog.replace_article(first)
        with pytest.raises(duckdb.ConstraintException):
            catalog.replace_article(same_source)
        with pytest.raises(duckdb.ConstraintException):
            catalog.replace_article(same_markdown)


def test_run_lifecycle_records_deterministic_content_free_json(
    tmp_path: Path,
) -> None:
    with Catalog.open(tmp_path / "catalog.duckdb") as catalog:
        run_id = catalog.begin_run("run", "limit:5")
        catalog.finish_run(
            run_id,
            "succeeded",
            {"succeeded": 2, "failed": 1, "discovered": 3},
        )
        row = catalog.connection.execute(
            """
            SELECT command, scope, status, summary_json, finished_at
            FROM runs
            WHERE run_id = ?
            """,
            [run_id],
        ).fetchone()

    assert row[0:3] == ("run", "limit:5", "succeeded")
    assert row[3] == '{"discovered":3,"failed":1,"succeeded":2}'
    assert "Synthetic headline" not in row[3]
    assert row[4] is not None


def test_run_and_failure_records_reject_free_form_text(tmp_path: Path) -> None:
    with Catalog.open(tmp_path / "catalog.duckdb") as catalog:
        run_id = catalog.begin_run("run", "limit:1")

        with pytest.raises(CatalogError, match="summary values must be content-free"):
            catalog.finish_run(
                run_id,
                "failed",
                {"error": "Synthetic headline and article paragraph"},
            )
        with pytest.raises(CatalogError, match="failure messages are catalog-owned"):
            catalog.replace_failure(
                run_id,
                "2024/broken.html",
                stage="extract",
                error_code="missing_publication_timestamp",
                message="Synthetic headline and article paragraph",
                extractor_version="3",
            )


def test_open_rejects_incompatible_schema_without_altering_it(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "catalog.duckdb"
    with duckdb.connect(str(database_path)) as connection:
        connection.execute(
            "CREATE TABLE metadata (key VARCHAR PRIMARY KEY, value VARCHAR)"
        )
        connection.execute("INSERT INTO metadata VALUES ('schema_version', '99')")

    with pytest.raises(
        CatalogError,
        match=(
            "unsupported catalog schema version 99; expected 1; "
            "move derived output aside or choose a fresh output root"
        ),
    ):
        Catalog.open(database_path)

    with duckdb.connect(str(database_path), read_only=True) as connection:
        assert connection.execute("SHOW TABLES").fetchall() == [("metadata",)]
        assert connection.execute("SELECT value FROM metadata").fetchone() == ("99",)


def test_open_rejects_malformed_version_one_schema_without_altering_it(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "catalog.duckdb"
    with duckdb.connect(str(database_path)) as connection:
        connection.execute(
            "CREATE TABLE metadata (key VARCHAR PRIMARY KEY, value VARCHAR)"
        )
        connection.execute("INSERT INTO metadata VALUES ('schema_version', '1')")
        connection.execute("CREATE TABLE rogue (value VARCHAR)")

    with pytest.raises(CatalogError, match="malformed catalog schema version 1"):
        Catalog.open(database_path)

    with duckdb.connect(str(database_path), read_only=True) as connection:
        assert connection.execute("SHOW TABLES").fetchall() == [
            ("metadata",),
            ("rogue",),
        ]


@pytest.mark.parametrize(
    "damage",
    (
        "ALTER TABLE runs DROP COLUMN summary_json",
        "CREATE OR REPLACE TABLE articles AS SELECT * FROM articles",
        "DROP INDEX articles_publication_date_idx",
        "DROP TABLE runs; CREATE VIEW runs AS SELECT 'x'::VARCHAR AS run_id",
    ),
    ids=("operational-column", "article-constraints", "date-index", "view"),
)
def test_open_rejects_damaged_version_one_contract(
    tmp_path: Path,
    damage: str,
) -> None:
    database_path = tmp_path / "catalog.duckdb"
    with Catalog.open(database_path):
        pass
    with duckdb.connect(str(database_path)) as connection:
        for statement in damage.split("; "):
            connection.execute(statement)

    with pytest.raises(CatalogError, match="malformed catalog schema version 1"):
        Catalog.open(database_path)


@pytest.mark.parametrize(
    ("command", "scope"),
    (
        ("Synthetic headline", "limit:1"),
        ("run", "Synthetic headline and article paragraph"),
    ),
    ids=("command", "scope"),
)
def test_begin_run_rejects_free_form_text(
    tmp_path: Path,
    command: str,
    scope: str,
) -> None:
    with (
        Catalog.open(tmp_path / "catalog.duckdb") as catalog,
        pytest.raises(CatalogError, match="invalid run"),
    ):
        catalog.begin_run(command, scope)


@pytest.mark.parametrize(
    ("status", "summary"),
    (
        ("Synthetic headline", {"succeeded": 1}),
        ("failed", {"Synthetic headline": 1}),
    ),
    ids=("status", "summary-key"),
)
def test_finish_run_rejects_free_form_text_channels(
    tmp_path: Path,
    status: str,
    summary: dict[str, object],
) -> None:
    with Catalog.open(tmp_path / "catalog.duckdb") as catalog:
        run_id = catalog.begin_run("run", "limit:1")
        with pytest.raises(CatalogError, match=r"invalid run|summary"):
            catalog.finish_run(run_id, status, summary)


@pytest.mark.parametrize(
    ("stage", "error_code", "extractor_version"),
    (
        (
            "Synthetic headline",
            "missing_publication_timestamp",
            "3",
        ),
        ("extract", "Synthetic_headline", "3"),
        (
            "extract",
            "missing_publication_timestamp",
            "Synthetic headline",
        ),
    ),
    ids=("stage", "error-code", "extractor-version"),
)
def test_replace_failure_rejects_free_form_text_channels(
    tmp_path: Path,
    stage: str,
    error_code: str,
    extractor_version: str,
) -> None:
    with Catalog.open(tmp_path / "catalog.duckdb") as catalog:
        run_id = catalog.begin_run("run", "limit:1")
        with pytest.raises(CatalogError, match="invalid failure"):
            catalog.replace_failure(
                run_id,
                "2024/broken.html",
                stage=stage,
                error_code=error_code,
                extractor_version=extractor_version,
            )


def test_replace_failure_uses_catalog_owned_message(tmp_path: Path) -> None:
    with Catalog.open(tmp_path / "catalog.duckdb") as catalog:
        run_id = catalog.begin_run("run", "limit:1")
        catalog.replace_failure(
            run_id,
            "2024/broken.html",
            stage="extract",
            error_code="missing_publication_timestamp",
            extractor_version="3",
        )
        row = catalog.connection.execute(
            "SELECT error_code, message FROM failures"
        ).fetchone()

    assert row == (
        "missing_publication_timestamp",
        "article has no publication timestamp",
    )


def test_context_manager_closes_connection(tmp_path: Path) -> None:
    catalog = Catalog.open(tmp_path / "catalog.duckdb")
    connection = catalog.connection

    with catalog:
        connection.execute("SELECT 1")

    with pytest.raises(duckdb.ConnectionException):
        connection.execute("SELECT 1")


def test_classify_compares_manifest_fingerprints_and_extractor_version(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    changed_image = SourceCandidate(
        relative_html_path=candidate.relative_html_path,
        html_size=candidate.html_size,
        html_mtime_ns=candidate.html_mtime_ns + 1,
        relative_header_image_path="2024/story_main_image.webp",
        header_image_size=999,
        header_image_mtime_ns=202020,
    )

    with Catalog.open(tmp_path / "catalog.duckdb") as catalog:
        assert catalog.classify(candidate, "3").kind == "new"
        catalog.replace_manifest(
            "run-1",
            candidate,
            extractor_version="3",
            status="processed",
            source_html_sha256="a" * 64,
            article_id="wsj:synthetic",
        )

        unchanged = catalog.classify(candidate, "3")
        stale = catalog.classify(candidate, "4")
        changed = catalog.classify(changed_image, "3")

    assert unchanged.kind == "unchanged"
    assert unchanged.old_html_sha256 == "a" * 64
    assert unchanged.old_article_id == "wsj:synthetic"
    assert stale.kind == "stale_extractor"
    assert changed.kind == "stat_changed"
    assert changed.image_changed is True


def test_failed_transaction_rolls_back_all_catalog_replacements(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    article = _article()

    with Catalog.open(tmp_path / "catalog.duckdb") as catalog:
        run_id = catalog.begin_run("run", "limit:1")

        with (
            pytest.raises(RuntimeError, match="synthetic interruption"),
            catalog.transaction(),
        ):
            catalog.replace_manifest(
                run_id,
                candidate,
                extractor_version=article.extractor_version,
                status="processed",
                source_html_sha256=article.source_html_sha256,
                article_id=article.article_id,
            )
            catalog.replace_candidate(run_id, article)
            catalog.replace_duplicates(
                run_id,
                article.article_id,
                candidate.relative_html_path,
                (
                    DuplicateRecord(
                        source_html_path="2024/duplicate.html",
                        duplicate_rank=2,
                        reason="lower_editorial_character_count",
                    ),
                ),
            )
            catalog.replace_failure(
                run_id,
                "2024/broken.html",
                stage="extract",
                error_code="missing_publication_timestamp",
                extractor_version=article.extractor_version,
            )
            catalog.replace_article(_article_record())
            in_transaction_counts = {
                table: catalog.connection.execute(
                    f"SELECT count(*) FROM {table}"
                ).fetchone()[0]
                for table in (
                    "source_manifest",
                    "candidates",
                    "duplicates",
                    "failures",
                    "articles",
                )
            }
            assert in_transaction_counts == {
                "source_manifest": 1,
                "candidates": 1,
                "duplicates": 1,
                "failures": 1,
                "articles": 1,
            }
            raise RuntimeError("synthetic interruption")

        counts = {
            table: catalog.connection.execute(
                f"SELECT count(*) FROM {table}"
            ).fetchone()[0]
            for table in (
                "source_manifest",
                "candidates",
                "duplicates",
                "failures",
                "articles",
            )
        }

    assert counts == {
        "source_manifest": 0,
        "candidates": 0,
        "duplicates": 0,
        "failures": 0,
        "articles": 0,
    }


def test_commit_failure_rolls_back_and_leaves_connection_usable(
    tmp_path: Path,
) -> None:
    class CommitFailureConnection:
        def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
            self.connection = connection
            self.fail_commit = True

        def execute(
            self,
            query: str,
            parameters: object = None,
        ) -> duckdb.DuckDBPyConnection:
            if query == "COMMIT" and self.fail_commit:
                self.fail_commit = False
                raise RuntimeError("synthetic commit failure")
            if parameters is None:
                return self.connection.execute(query)
            return self.connection.execute(query, parameters)

        def close(self) -> None:
            self.connection.close()

    catalog = Catalog.open(tmp_path / "catalog.duckdb")
    catalog.connection = CommitFailureConnection(catalog.connection)  # type: ignore[assignment]
    with catalog:
        with (
            pytest.raises(RuntimeError, match="synthetic commit failure"),
            catalog.transaction(),
        ):
            catalog.replace_article(_article_record())

        assert catalog.connection.execute("SELECT count(*) FROM articles").fetchone()[
            0
        ] == 0
        with catalog.transaction():
            catalog.replace_article(_article_record())
