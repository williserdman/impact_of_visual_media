"""Versioned DuckDB catalog and transactional pipeline state."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from uuid import uuid4

import duckdb

from wsj_pipeline.models import ExtractedArticle, SourceCandidate

CATALOG_SCHEMA_VERSION = "1"

_CATALOG_TABLES = {
    "articles",
    "candidates",
    "duplicates",
    "failures",
    "metadata",
    "runs",
    "source_manifest",
}
_ARTICLE_COLUMN_TYPES = {
    "article_id": "VARCHAR",
    "source_html_path": "VARCHAR",
    "cleaned_markdown_path": "VARCHAR",
    "header_image_path": "VARCHAR",
    "inline_image_urls": "VARCHAR[]",
    "published_at_utc": "TIMESTAMP WITH TIME ZONE",
    "publication_date_new_york": "DATE",
    "cleaned_markdown_sha256": "VARCHAR",
}


class CatalogError(RuntimeError):
    """Raised when the catalog is missing required compatible metadata."""


@dataclass(frozen=True, slots=True)
class CandidateStatus:
    """Classification of a discovered source against its manifest row."""

    kind: str
    old_html_sha256: str | None = None
    old_article_id: str | None = None
    image_changed: bool = False


@dataclass(frozen=True, slots=True)
class DuplicateRecord:
    """Content-free reason that one source lost canonical selection."""

    source_html_path: str
    duplicate_rank: int
    reason: str


@dataclass(frozen=True, slots=True)
class ArticleRecord:
    """Research-facing paths, image references, date, and integrity hash."""

    article_id: str
    source_html_path: str
    cleaned_markdown_path: str
    header_image_path: str | None
    inline_image_urls: tuple[str, ...]
    published_at_utc: datetime
    publication_date_new_york: date
    cleaned_markdown_sha256: str


class Catalog:
    """Own one compatible DuckDB connection and its SQL operations."""

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self.connection = connection

    @classmethod
    def open(cls, path: Path) -> Catalog:
        """Open ``path``, creating schema version 1 only for a fresh database."""

        path.parent.mkdir(parents=True, exist_ok=True)
        catalog = cls(duckdb.connect(str(path)))
        try:
            catalog._ensure_schema()
        except BaseException:
            catalog.connection.close()
            raise
        return catalog

    def __enter__(self) -> Catalog:
        return self

    def __exit__(self, *_args: object) -> None:
        self.connection.close()

    def _ensure_schema(self) -> None:
        tables = {
            row[0]
            for row in self.connection.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_catalog = current_database()
                  AND table_schema = 'main'
                """
            ).fetchall()
        }
        if "metadata" in tables:
            try:
                row = self.connection.execute(
                    "SELECT value FROM metadata WHERE key = 'schema_version'"
                ).fetchone()
            except duckdb.Error:
                row = None
            version = str(row[0]) if row is not None else "missing"
            self._refuse_unsupported_version(version)
            self._validate_existing_schema(tables)
            return
        elif tables:
            self._refuse_unsupported_version("missing")

        self.connection.execute("BEGIN TRANSACTION")
        try:
            self._create_tables()
            self.connection.execute(
                """
                INSERT INTO metadata (key, value)
                VALUES ('schema_version', ?)
                ON CONFLICT (key) DO NOTHING
                """,
                [CATALOG_SCHEMA_VERSION],
            )
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise
        else:
            self.connection.execute("COMMIT")

    @staticmethod
    def _refuse_unsupported_version(version: str) -> None:
        if version != CATALOG_SCHEMA_VERSION:
            raise CatalogError(
                f"unsupported catalog schema version {version}; "
                f"expected {CATALOG_SCHEMA_VERSION}; move derived output "
                "aside or choose a fresh output root"
            )

    def _validate_existing_schema(self, tables: set[str]) -> None:
        if tables != _CATALOG_TABLES:
            self._raise_malformed_schema()
        article_types = {
            row[1]: row[2]
            for row in self.connection.execute(
                "PRAGMA table_info('articles')"
            ).fetchall()
        }
        if article_types != _ARTICLE_COLUMN_TYPES:
            self._raise_malformed_schema()

    @staticmethod
    def _raise_malformed_schema() -> None:
        raise CatalogError(
            f"malformed catalog schema version {CATALOG_SCHEMA_VERSION}; "
            "move derived output aside or choose a fresh output root"
        )

    def _create_tables(self) -> None:
        statements = (
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key VARCHAR PRIMARY KEY,
                value VARCHAR NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id VARCHAR PRIMARY KEY,
                command VARCHAR NOT NULL,
                scope VARCHAR NOT NULL,
                status VARCHAR NOT NULL,
                started_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
                finished_at TIMESTAMPTZ,
                summary_json VARCHAR
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS source_manifest (
                source_html_path VARCHAR PRIMARY KEY,
                html_size BIGINT NOT NULL,
                html_mtime_ns BIGINT NOT NULL,
                header_image_path VARCHAR,
                header_image_size BIGINT,
                header_image_mtime_ns BIGINT,
                source_html_sha256 VARCHAR,
                extractor_version VARCHAR NOT NULL,
                status VARCHAR NOT NULL,
                article_id VARCHAR,
                last_run_id VARCHAR NOT NULL,
                last_seen_run_id VARCHAR NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS candidates (
                source_html_path VARCHAR PRIMARY KEY,
                article_id VARCHAR NOT NULL,
                source_html_sha256 VARCHAR NOT NULL,
                extractor_version VARCHAR NOT NULL,
                canonical_url VARCHAR,
                published_at_utc TIMESTAMPTZ NOT NULL,
                publication_date_new_york DATE NOT NULL,
                updated_at_utc TIMESTAMPTZ,
                editorial_character_count BIGINT NOT NULL,
                editorial_block_count INTEGER NOT NULL,
                metadata_completeness INTEGER NOT NULL,
                header_image_path VARCHAR,
                inline_image_urls VARCHAR[] NOT NULL,
                warnings VARCHAR[] NOT NULL,
                last_run_id VARCHAR NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS duplicates (
                article_id VARCHAR NOT NULL,
                source_html_path VARCHAR NOT NULL,
                winner_source_html_path VARCHAR NOT NULL,
                duplicate_rank INTEGER NOT NULL,
                reason VARCHAR NOT NULL,
                run_id VARCHAR NOT NULL,
                PRIMARY KEY (article_id, source_html_path)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS failures (
                run_id VARCHAR NOT NULL,
                source_html_path VARCHAR NOT NULL,
                stage VARCHAR NOT NULL,
                error_code VARCHAR NOT NULL,
                message VARCHAR NOT NULL,
                extractor_version VARCHAR NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
                PRIMARY KEY (run_id, source_html_path, stage)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS articles (
                article_id VARCHAR PRIMARY KEY,
                source_html_path VARCHAR NOT NULL UNIQUE,
                cleaned_markdown_path VARCHAR NOT NULL UNIQUE,
                header_image_path VARCHAR,
                inline_image_urls VARCHAR[] NOT NULL,
                published_at_utc TIMESTAMPTZ NOT NULL,
                publication_date_new_york DATE NOT NULL,
                cleaned_markdown_sha256 VARCHAR NOT NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS articles_publication_date_idx
            ON articles (
                publication_date_new_york,
                published_at_utc,
                article_id
            )
            """,
        )
        for statement in statements:
            self.connection.execute(statement)

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Commit all enclosed catalog operations or roll every one back."""

        self.connection.execute("BEGIN TRANSACTION")
        try:
            yield
            self.connection.execute("COMMIT")
        except BaseException:
            with suppress(duckdb.TransactionException):
                self.connection.execute("ROLLBACK")
            raise

    def begin_run(self, command: str, scope: str) -> str:
        run_id = str(uuid4())
        self.connection.execute(
            """
            INSERT INTO runs (run_id, command, scope, status)
            VALUES (?, ?, ?, 'running')
            """,
            [run_id, command, scope],
        )
        return run_id

    def finish_run(
        self,
        run_id: str,
        status: str,
        summary: Mapping[str, object],
    ) -> None:
        if not _is_content_free_summary(summary):
            raise CatalogError("run summary values must be content-free counts")
        summary_json = json.dumps(
            summary,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        self.connection.execute(
            """
            UPDATE runs
            SET status = ?, summary_json = ?, finished_at = current_timestamp
            WHERE run_id = ?
            """,
            [status, summary_json, run_id],
        )

    def classify(
        self,
        candidate: SourceCandidate,
        extractor_version: str,
    ) -> CandidateStatus:
        row = self.connection.execute(
            """
            SELECT html_size, html_mtime_ns, header_image_path,
                   header_image_size, header_image_mtime_ns,
                   source_html_sha256, extractor_version, article_id, status
            FROM source_manifest
            WHERE source_html_path = ?
            """,
            [candidate.relative_html_path],
        ).fetchone()
        if row is None or row[8] == "missing":
            return CandidateStatus("new")

        old_html_sha256 = row[5]
        old_article_id = row[7]
        if row[6] != extractor_version:
            return CandidateStatus(
                "stale_extractor",
                old_html_sha256,
                old_article_id,
            )

        current_fingerprint = (
            candidate.html_size,
            candidate.html_mtime_ns,
            candidate.relative_header_image_path,
            candidate.header_image_size,
            candidate.header_image_mtime_ns,
        )
        if tuple(row[:5]) == current_fingerprint:
            return CandidateStatus(
                "unchanged",
                old_html_sha256,
                old_article_id,
            )

        image_changed = tuple(row[2:5]) != current_fingerprint[2:5]
        return CandidateStatus(
            "stat_changed",
            old_html_sha256,
            old_article_id,
            image_changed,
        )

    def replace_manifest(
        self,
        run_id: str,
        candidate: SourceCandidate,
        *,
        extractor_version: str,
        status: str,
        source_html_sha256: str | None,
        article_id: str | None,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO source_manifest (
                source_html_path, html_size, html_mtime_ns,
                header_image_path, header_image_size, header_image_mtime_ns,
                source_html_sha256, extractor_version, status, article_id,
                last_run_id, last_seen_run_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (source_html_path) DO UPDATE SET
                html_size = excluded.html_size,
                html_mtime_ns = excluded.html_mtime_ns,
                header_image_path = excluded.header_image_path,
                header_image_size = excluded.header_image_size,
                header_image_mtime_ns = excluded.header_image_mtime_ns,
                source_html_sha256 = excluded.source_html_sha256,
                extractor_version = excluded.extractor_version,
                status = excluded.status,
                article_id = excluded.article_id,
                last_run_id = excluded.last_run_id,
                last_seen_run_id = excluded.last_seen_run_id
            """,
            [
                candidate.relative_html_path,
                candidate.html_size,
                candidate.html_mtime_ns,
                candidate.relative_header_image_path,
                candidate.header_image_size,
                candidate.header_image_mtime_ns,
                source_html_sha256,
                extractor_version,
                status,
                article_id,
                run_id,
                run_id,
            ],
        )

    def replace_candidate(self, run_id: str, article: ExtractedArticle) -> None:
        self.connection.execute(
            """
            INSERT INTO candidates (
                source_html_path, article_id, source_html_sha256,
                extractor_version, canonical_url, published_at_utc,
                publication_date_new_york, updated_at_utc,
                editorial_character_count, editorial_block_count,
                metadata_completeness, header_image_path, inline_image_urls,
                warnings, last_run_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (source_html_path) DO UPDATE SET
                article_id = excluded.article_id,
                source_html_sha256 = excluded.source_html_sha256,
                extractor_version = excluded.extractor_version,
                canonical_url = excluded.canonical_url,
                published_at_utc = excluded.published_at_utc,
                publication_date_new_york = excluded.publication_date_new_york,
                updated_at_utc = excluded.updated_at_utc,
                editorial_character_count = excluded.editorial_character_count,
                editorial_block_count = excluded.editorial_block_count,
                metadata_completeness = excluded.metadata_completeness,
                header_image_path = excluded.header_image_path,
                inline_image_urls = excluded.inline_image_urls,
                warnings = excluded.warnings,
                last_run_id = excluded.last_run_id
            """,
            [
                article.relative_html_path,
                article.article_id,
                article.source_html_sha256,
                article.extractor_version,
                article.canonical_url,
                article.published_at_utc,
                article.publication_date_new_york,
                article.updated_at_utc,
                article.editorial_character_count,
                article.editorial_block_count,
                article.metadata_completeness,
                article.relative_header_image_path,
                list(article.inline_image_urls),
                list(article.warnings),
                run_id,
            ],
        )

    def replace_duplicates(
        self,
        run_id: str,
        article_id: str,
        winner_source_html_path: str,
        duplicates: Iterable[DuplicateRecord],
    ) -> None:
        self.connection.execute(
            "DELETE FROM duplicates WHERE article_id = ?",
            [article_id],
        )
        for duplicate in duplicates:
            self.connection.execute(
                """
                INSERT INTO duplicates (
                    article_id, source_html_path, winner_source_html_path,
                    duplicate_rank, reason, run_id
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    article_id,
                    duplicate.source_html_path,
                    winner_source_html_path,
                    duplicate.duplicate_rank,
                    duplicate.reason,
                    run_id,
                ],
            )

    def replace_failure(
        self,
        run_id: str,
        source_html_path: str,
        *,
        stage: str,
        error_code: str,
        extractor_version: str,
        message: str | None = None,
    ) -> None:
        if message is not None:
            raise CatalogError(
                "failure messages are catalog-owned; pass only a stable error code"
            )
        self.connection.execute(
            """
            INSERT INTO failures (
                run_id, source_html_path, stage, error_code, message,
                extractor_version
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (run_id, source_html_path, stage) DO UPDATE SET
                error_code = excluded.error_code,
                message = excluded.message,
                extractor_version = excluded.extractor_version,
                created_at = now()
            """,
            [
                run_id,
                source_html_path,
                stage,
                error_code,
                error_code,
                extractor_version,
            ],
        )

    def replace_article(self, article: ArticleRecord) -> None:
        self.connection.execute(
            """
            INSERT INTO articles (
                article_id, source_html_path, cleaned_markdown_path,
                header_image_path, inline_image_urls, published_at_utc,
                publication_date_new_york, cleaned_markdown_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (article_id) DO UPDATE SET
                source_html_path = excluded.source_html_path,
                cleaned_markdown_path = excluded.cleaned_markdown_path,
                header_image_path = excluded.header_image_path,
                inline_image_urls = excluded.inline_image_urls,
                published_at_utc = excluded.published_at_utc,
                publication_date_new_york = excluded.publication_date_new_york,
                cleaned_markdown_sha256 = excluded.cleaned_markdown_sha256
            """,
            [
                article.article_id,
                article.source_html_path,
                article.cleaned_markdown_path,
                article.header_image_path,
                list(article.inline_image_urls),
                article.published_at_utc,
                article.publication_date_new_york,
                article.cleaned_markdown_sha256,
            ],
        )


def _is_content_free_summary(value: object) -> bool:
    if value is None or isinstance(value, bool | int):
        return True
    if isinstance(value, float):
        return value == value and value not in (float("inf"), float("-inf"))
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str) and _is_content_free_summary(item)
            for key, item in value.items()
        )
    if isinstance(value, tuple | list):
        return all(_is_content_free_summary(item) for item in value)
    return False
