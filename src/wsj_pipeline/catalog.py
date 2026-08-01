"""Versioned DuckDB catalog and transactional pipeline state."""

from __future__ import annotations

import json
import re
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
_TABLE_COLUMNS = {
    "metadata": (
        ("key", "VARCHAR", True, None, True),
        ("value", "VARCHAR", True, None, False),
    ),
    "runs": (
        ("run_id", "VARCHAR", True, None, True),
        ("command", "VARCHAR", True, None, False),
        ("scope", "VARCHAR", True, None, False),
        ("status", "VARCHAR", True, None, False),
        ("started_at", "TIMESTAMP WITH TIME ZONE", True, "current_timestamp", False),
        ("finished_at", "TIMESTAMP WITH TIME ZONE", False, None, False),
        ("summary_json", "VARCHAR", False, None, False),
    ),
    "source_manifest": (
        ("source_html_path", "VARCHAR", True, None, True),
        ("html_size", "BIGINT", True, None, False),
        ("html_mtime_ns", "BIGINT", True, None, False),
        ("header_image_path", "VARCHAR", False, None, False),
        ("header_image_size", "BIGINT", False, None, False),
        ("header_image_mtime_ns", "BIGINT", False, None, False),
        ("source_html_sha256", "VARCHAR", False, None, False),
        ("extractor_version", "VARCHAR", True, None, False),
        ("status", "VARCHAR", True, None, False),
        ("article_id", "VARCHAR", False, None, False),
        ("last_run_id", "VARCHAR", True, None, False),
        ("last_seen_run_id", "VARCHAR", True, None, False),
    ),
    "candidates": (
        ("source_html_path", "VARCHAR", True, None, True),
        ("article_id", "VARCHAR", True, None, False),
        ("source_html_sha256", "VARCHAR", True, None, False),
        ("extractor_version", "VARCHAR", True, None, False),
        ("canonical_url", "VARCHAR", False, None, False),
        ("published_at_utc", "TIMESTAMP WITH TIME ZONE", True, None, False),
        ("publication_date_new_york", "DATE", True, None, False),
        ("updated_at_utc", "TIMESTAMP WITH TIME ZONE", False, None, False),
        ("editorial_character_count", "BIGINT", True, None, False),
        ("editorial_block_count", "INTEGER", True, None, False),
        ("metadata_completeness", "INTEGER", True, None, False),
        ("header_image_path", "VARCHAR", False, None, False),
        ("inline_image_urls", "VARCHAR[]", True, None, False),
        ("warnings", "VARCHAR[]", True, None, False),
        ("last_run_id", "VARCHAR", True, None, False),
    ),
    "duplicates": (
        ("article_id", "VARCHAR", True, None, True),
        ("source_html_path", "VARCHAR", True, None, True),
        ("winner_source_html_path", "VARCHAR", True, None, False),
        ("duplicate_rank", "INTEGER", True, None, False),
        ("reason", "VARCHAR", True, None, False),
        ("run_id", "VARCHAR", True, None, False),
    ),
    "failures": (
        ("run_id", "VARCHAR", True, None, True),
        ("source_html_path", "VARCHAR", True, None, True),
        ("stage", "VARCHAR", True, None, True),
        ("error_code", "VARCHAR", True, None, False),
        ("message", "VARCHAR", True, None, False),
        ("extractor_version", "VARCHAR", True, None, False),
        ("created_at", "TIMESTAMP WITH TIME ZONE", True, "current_timestamp", False),
    ),
    "articles": tuple(
        (name, data_type, name != "header_image_path", None, name == "article_id")
        for name, data_type in _ARTICLE_COLUMN_TYPES.items()
    ),
}
_KEY_CONSTRAINTS = {
    ("metadata", "PRIMARY KEY", ("key",)),
    ("runs", "PRIMARY KEY", ("run_id",)),
    ("source_manifest", "PRIMARY KEY", ("source_html_path",)),
    ("candidates", "PRIMARY KEY", ("source_html_path",)),
    ("duplicates", "PRIMARY KEY", ("article_id", "source_html_path")),
    ("failures", "PRIMARY KEY", ("run_id", "source_html_path", "stage")),
    ("articles", "PRIMARY KEY", ("article_id",)),
    ("articles", "UNIQUE", ("source_html_path",)),
    ("articles", "UNIQUE", ("cleaned_markdown_path",)),
}
_RUN_COMMANDS = {"inventory", "run", "validate", "smoke"}
_RUN_SCOPE_PATTERN = re.compile(
    r"(?:limit:[1-9][0-9]*(?:,reprocess)?|full(?:,reprocess)?|fixture)"
)
_RUN_STATUSES = {"succeeded", "failed"}
_SUMMARY_COUNTERS = {
    "articles",
    "changed",
    "discovered",
    "duplicate_count",
    "duplicates",
    "failed",
    "failures",
    "metadata_only",
    "missing_images",
    "new",
    "removed",
    "reprocessed",
    "stale",
    "succeeded",
    "unchanged",
}
_FAILURE_STAGES = {"catalog", "discovery", "extract", "publish", "validate"}
_FAILURE_MESSAGES = {
    "invalid_timestamp": "timestamp is not valid ISO-8601",
    "missing_publication_timestamp": "article has no publication timestamp",
    "naive_timestamp": "timestamp has no timezone",
}
_EXTRACTOR_VERSION_PATTERN = re.compile(r"[1-9][0-9]*")


class CatalogError(RuntimeError):
    """Raised when the catalog is missing required compatible metadata."""


@dataclass(frozen=True, slots=True)
class CandidateStatus:
    """Classification of a discovered source against its manifest row."""

    kind: str
    old_html_sha256: str | None = None
    old_article_id: str | None = None
    image_changed: bool = False
    was_failed: bool = False


@dataclass(frozen=True, slots=True)
class DuplicateRecord:
    """Content-free reason that one source lost canonical selection."""

    source_html_path: str
    duplicate_rank: int
    reason: str


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    """Content-free quality metadata for one extracted source candidate."""

    source_html_path: str
    article_id: str
    source_html_sha256: str
    extractor_version: str
    canonical_url: str | None
    published_at_utc: datetime
    publication_date_new_york: date
    updated_at_utc: datetime | None
    editorial_character_count: int
    editorial_block_count: int
    metadata_completeness: int
    header_image_path: str | None
    inline_image_urls: tuple[str, ...]
    warnings: tuple[str, ...]


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
        table_types = {
            row[0]: row[1]
            for row in self.connection.execute(
                """
                SELECT table_name, table_type
                FROM information_schema.tables
                WHERE table_catalog = current_database()
                  AND table_schema = 'main'
                """
            ).fetchall()
        }
        if "metadata" in table_types:
            try:
                row = self.connection.execute(
                    "SELECT value FROM metadata WHERE key = 'schema_version'"
                ).fetchone()
            except duckdb.Error:
                row = None
            version = str(row[0]) if row is not None else "missing"
            self._refuse_unsupported_version(version)
            self._validate_existing_schema(table_types)
            return
        elif table_types:
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

    def _validate_existing_schema(self, table_types: dict[str, str]) -> None:
        if set(table_types) != _CATALOG_TABLES or set(table_types.values()) != {
            "BASE TABLE"
        }:
            self._raise_malformed_schema()
        table_columns = {
            table: tuple(
                (row[1], row[2], row[3], row[4], row[5])
                for row in self.connection.execute(
                    f"PRAGMA table_info('{table}')"
                ).fetchall()
            )
            for table in _CATALOG_TABLES
        }
        key_constraints = {
            (row[0], row[1], tuple(row[2]))
            for row in self.connection.execute(
                """
                SELECT table_name, constraint_type, constraint_column_names
                FROM duckdb_constraints()
                WHERE constraint_type IN ('PRIMARY KEY', 'UNIQUE')
                """
            ).fetchall()
        }
        indexes = self.connection.execute(
            """
            SELECT table_name, index_name, is_unique, expressions
            FROM duckdb_indexes()
            """
        ).fetchall()
        if (
            table_columns != _TABLE_COLUMNS
            or key_constraints != _KEY_CONSTRAINTS
            or indexes
            != [
                (
                    "articles",
                    "articles_publication_date_idx",
                    False,
                    "[publication_date_new_york, published_at_utc, article_id]",
                )
            ]
        ):
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
        if command not in _RUN_COMMANDS or _RUN_SCOPE_PATTERN.fullmatch(scope) is None:
            raise CatalogError("invalid run command or scope")
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
        if status not in _RUN_STATUSES:
            raise CatalogError("invalid run status")
        _validate_summary(summary)
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
        was_failed = row[8] == "failed"
        if row[6] != extractor_version:
            return CandidateStatus(
                "stale_extractor",
                old_html_sha256,
                old_article_id,
                was_failed=was_failed,
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
                was_failed=was_failed,
            )

        image_changed = tuple(row[2:5]) != current_fingerprint[2:5]
        return CandidateStatus(
            "stat_changed",
            old_html_sha256,
            old_article_id,
            image_changed,
            was_failed,
        )

    def mark_seen(self, run_id: str, source_html_path: str) -> None:
        """Mark a discovered row seen without changing its stored fingerprint."""

        result = self.connection.execute(
            """
            UPDATE source_manifest
            SET last_seen_run_id = ?
            WHERE source_html_path = ?
            """,
            [run_id, source_html_path],
        )
        if result.fetchone()[0] != 1:
            raise CatalogError("cannot mark an unknown source manifest row as seen")

    def reconcile_missing(self, run_id: str) -> tuple[str, ...]:
        """Mark unseen sources missing and discard their publishable candidates."""

        missing_paths = tuple(
            row[0]
            for row in self.connection.execute(
                """
                SELECT source_html_path
                FROM source_manifest
                WHERE last_seen_run_id <> ? AND status <> 'missing'
                ORDER BY source_html_path
                """,
                [run_id],
            ).fetchall()
        )
        if not missing_paths:
            return ()

        affected_article_ids = tuple(
            row[0]
            for row in self.connection.execute(
                """
                SELECT DISTINCT article_id
                FROM (
                    SELECT article_id
                    FROM source_manifest
                    WHERE source_html_path IN (
                        SELECT unnest(?)
                    )
                    UNION ALL
                    SELECT article_id
                    FROM candidates
                    WHERE source_html_path IN (
                        SELECT unnest(?)
                    )
                )
                WHERE article_id IS NOT NULL
                ORDER BY article_id
                """,
                [list(missing_paths), list(missing_paths)],
            ).fetchall()
        )
        self.connection.execute(
            """
            UPDATE source_manifest
            SET status = 'missing', last_run_id = ?
            WHERE source_html_path IN (SELECT unnest(?))
            """,
            [run_id, list(missing_paths)],
        )
        self.connection.execute(
            """
            DELETE FROM candidates
            WHERE source_html_path IN (SELECT unnest(?))
            """,
            [list(missing_paths)],
        )
        return affected_article_ids

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

    def candidates_for_article(
        self,
        article_id: str,
    ) -> tuple[CandidateRecord, ...]:
        """Return deterministic, content-free candidates for one identity."""

        rows = self.connection.execute(
            """
            SELECT source_html_path, article_id, source_html_sha256,
                   extractor_version, canonical_url, published_at_utc,
                   publication_date_new_york, updated_at_utc,
                   editorial_character_count, editorial_block_count,
                   metadata_completeness, header_image_path,
                   inline_image_urls, warnings
            FROM candidates
            WHERE article_id = ?
            ORDER BY source_html_path
            """,
            [article_id],
        ).fetchall()
        return tuple(
            CandidateRecord(
                source_html_path=row[0],
                article_id=row[1],
                source_html_sha256=row[2],
                extractor_version=row[3],
                canonical_url=row[4],
                published_at_utc=row[5],
                publication_date_new_york=row[6],
                updated_at_utc=row[7],
                editorial_character_count=row[8],
                editorial_block_count=row[9],
                metadata_completeness=row[10],
                header_image_path=row[11],
                inline_image_urls=tuple(row[12]),
                warnings=tuple(row[13]),
            )
            for row in rows
        )

    def remove_candidate(self, source_html_path: str) -> str | None:
        """Discard one invalid candidate and return its affected identity."""

        row = self.connection.execute(
            "SELECT article_id FROM candidates WHERE source_html_path = ?",
            [source_html_path],
        ).fetchone()
        if row is None:
            return None
        self.connection.execute(
            "DELETE FROM candidates WHERE source_html_path = ?",
            [source_html_path],
        )
        return str(row[0])

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
        if (
            stage not in _FAILURE_STAGES
            or error_code not in _FAILURE_MESSAGES
            or _EXTRACTOR_VERSION_PATTERN.fullmatch(extractor_version) is None
        ):
            raise CatalogError("invalid failure stage, code, or extractor version")
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
                _FAILURE_MESSAGES[error_code],
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

    def article_for_id(self, article_id: str) -> ArticleRecord | None:
        """Return the research record for one identity, if it exists."""

        row = self.connection.execute(
            """
            SELECT article_id, source_html_path, cleaned_markdown_path,
                   header_image_path, inline_image_urls, published_at_utc,
                   publication_date_new_york, cleaned_markdown_sha256
            FROM articles
            WHERE article_id = ?
            """,
            [article_id],
        ).fetchone()
        if row is None:
            return None
        return ArticleRecord(
            article_id=row[0],
            source_html_path=row[1],
            cleaned_markdown_path=row[2],
            header_image_path=row[3],
            inline_image_urls=tuple(row[4]),
            published_at_utc=row[5],
            publication_date_new_york=row[6],
            cleaned_markdown_sha256=row[7],
        )

    def article_count(self) -> int:
        """Return the number of current research article rows."""

        return int(
            self.connection.execute("SELECT count(*) FROM articles").fetchone()[0]
        )

    def duplicate_count(self) -> int:
        """Return the number of current non-winning source rows."""

        return int(
            self.connection.execute("SELECT count(*) FROM duplicates").fetchone()[0]
        )

    def remove_article(self, article_id: str) -> ArticleRecord | None:
        """Remove and return one research row inside the caller's transaction."""

        existing = self.article_for_id(article_id)
        if existing is not None:
            self.connection.execute(
                "DELETE FROM articles WHERE article_id = ?",
                [article_id],
            )
        return existing


def _validate_summary(summary: Mapping[str, object]) -> None:
    for key, value in summary.items():
        if key == "validation_ok" and isinstance(value, bool):
            continue
        if key not in _SUMMARY_COUNTERS or isinstance(value, bool) or not isinstance(
            value, int
        ) or value < 0:
            raise CatalogError(
                "run summary values must be content-free counts with allowlisted keys"
            )
