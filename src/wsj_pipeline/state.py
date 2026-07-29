"""DuckDB operational state and incremental source processing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import duckdb

from wsj_pipeline.config import PipelineConfig
from wsj_pipeline.extract import EXTRACTOR_VERSION, ExtractionError, extract_article
from wsj_pipeline.models import ExtractedArticle, SourceCandidate

STATE_SCHEMA_VERSION = "1"


class StateError(RuntimeError):
    """Raised when operational state is missing or incompatible."""


@dataclass(frozen=True, slots=True)
class CandidateStatus:
    """Classification of a discovered candidate against stored state."""

    kind: str
    old_html_sha256: str | None = None
    old_article_key: str | None = None


@dataclass(frozen=True, slots=True)
class ProcessSummary:
    """Observable results of one bounded extraction stage."""

    discovered: int = 0
    new: int = 0
    changed: int = 0
    metadata_only: int = 0
    unchanged: int = 0
    succeeded: int = 0
    failed: int = 0
    affected_article_keys: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class PipelineState:
    """Transactional access to the operational DuckDB database."""

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self.connection = connection

    @classmethod
    def open(cls, path: Path) -> PipelineState:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = cls(duckdb.connect(str(path)))
        state._ensure_schema()
        return state

    def __enter__(self) -> PipelineState:
        return self

    def __exit__(self, *_args: object) -> None:
        self.connection.close()

    def _ensure_schema(self) -> None:
        metadata_exists = self.connection.execute(
            """
            SELECT count(*)
            FROM information_schema.tables
            WHERE table_name = 'metadata'
            """
        ).fetchone()[0]
        if metadata_exists:
            row = self.connection.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            ).fetchone()
            version = row[0] if row else "missing"
            if version != STATE_SCHEMA_VERSION:
                self.connection.close()
                raise StateError(
                    f"unsupported operational schema version {version}; "
                    f"expected {STATE_SCHEMA_VERSION}"
                )
        self.connection.execute("BEGIN TRANSACTION")
        try:
            self._create_tables()
            self.connection.execute(
                """
                INSERT INTO metadata
                SELECT 'schema_version', ?
                WHERE NOT EXISTS (
                    SELECT 1 FROM metadata WHERE key = 'schema_version'
                )
                """,
                [STATE_SCHEMA_VERSION],
            )
        except Exception:
            self.connection.execute("ROLLBACK")
            raise
        else:
            self.connection.execute("COMMIT")

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
                source_path VARCHAR PRIMARY KEY,
                html_size BIGINT NOT NULL,
                html_mtime_ns BIGINT NOT NULL,
                image_path VARCHAR,
                image_size BIGINT,
                image_mtime_ns BIGINT,
                html_sha256 VARCHAR,
                extractor_version VARCHAR NOT NULL,
                status VARCHAR NOT NULL,
                article_key VARCHAR,
                last_run_id VARCHAR NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS extracted_sources (
                source_path VARCHAR PRIMARY KEY,
                article_key VARCHAR NOT NULL,
                payload_json VARCHAR NOT NULL,
                published_at_utc TIMESTAMPTZ NOT NULL,
                publication_year_ny INTEGER NOT NULL,
                publication_month_ny INTEGER NOT NULL,
                has_body BOOLEAN NOT NULL,
                body_word_count INTEGER NOT NULL,
                metadata_completeness INTEGER NOT NULL,
                updated_at_utc TIMESTAMPTZ,
                html_sha256 VARCHAR NOT NULL,
                extractor_version VARCHAR NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS failures (
                failure_id VARCHAR PRIMARY KEY,
                run_id VARCHAR NOT NULL,
                source_path VARCHAR NOT NULL,
                stage VARCHAR NOT NULL,
                error_code VARCHAR NOT NULL,
                message VARCHAR NOT NULL,
                extractor_version VARCHAR NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS canonical_sources (
                article_key VARCHAR PRIMARY KEY,
                source_path VARCHAR NOT NULL,
                publication_year_ny INTEGER NOT NULL,
                publication_month_ny INTEGER NOT NULL,
                updated_run_id VARCHAR NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS duplicates (
                article_key VARCHAR NOT NULL,
                source_path VARCHAR NOT NULL,
                winner_source_path VARCHAR NOT NULL,
                duplicate_rank INTEGER NOT NULL,
                reason VARCHAR NOT NULL,
                run_id VARCHAR NOT NULL,
                PRIMARY KEY (article_key, source_path)
            )
            """,
        )
        for statement in statements:
            self.connection.execute(statement)

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
        *,
        status: str,
        summary: Mapping[str, object],
    ) -> None:
        self.connection.execute(
            """
            UPDATE runs
            SET status = ?,
                summary_json = ?,
                finished_at = current_timestamp
            WHERE run_id = ?
            """,
            [
                status,
                json.dumps(
                    summary,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                run_id,
            ],
        )

    def classify(
        self,
        candidate: SourceCandidate,
        extractor_version: str,
    ) -> CandidateStatus:
        row = self.connection.execute(
            """
            SELECT html_size, html_mtime_ns, image_path, image_size,
                   image_mtime_ns, html_sha256, extractor_version, article_key
            FROM source_manifest
            WHERE source_path = ?
            """,
            [candidate.relative_html_path],
        ).fetchone()
        if row is None:
            return CandidateStatus("new")
        old_hash = row[5]
        old_key = row[7]
        if row[6] != extractor_version:
            return CandidateStatus("stale_extractor", old_hash, old_key)
        current_fingerprint = (
            candidate.html_size,
            candidate.html_mtime_ns,
            candidate.relative_image_path,
            candidate.image_size,
            candidate.image_mtime_ns,
        )
        if tuple(row[:5]) == current_fingerprint:
            return CandidateStatus("unchanged", old_hash, old_key)
        return CandidateStatus("stat_changed", old_hash, old_key)

    def store_success(
        self,
        run_id: str,
        candidate: SourceCandidate,
        article: ExtractedArticle,
    ) -> str:
        article_key = _provisional_article_key(article)
        payload_json = json.dumps(
            _json_compatible(asdict(article)),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self.connection.execute(
            """
            INSERT OR REPLACE INTO extracted_sources VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                candidate.relative_html_path,
                article_key,
                payload_json,
                article.published_at_utc,
                article.publication_year_ny,
                article.publication_month_ny,
                bool(article.body_text),
                article.body_word_count,
                article.metadata_completeness,
                article.updated_at_utc,
                article.source_html_sha256,
                article.extractor_version,
            ],
        )
        self._upsert_manifest(
            run_id,
            candidate,
            html_sha256=article.source_html_sha256,
            status="processed",
            article_key=article_key,
        )
        return article_key

    def store_failure(
        self,
        run_id: str,
        candidate: SourceCandidate,
        error: ExtractionError,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO failures (
                failure_id, run_id, source_path, stage, error_code,
                message, extractor_version
            ) VALUES (?, ?, ?, 'extract', ?, ?, ?)
            """,
            [
                str(uuid4()),
                run_id,
                candidate.relative_html_path,
                error.code,
                error.message,
                EXTRACTOR_VERSION,
            ],
        )
        self.connection.execute(
            "DELETE FROM extracted_sources WHERE source_path = ?",
            [candidate.relative_html_path],
        )
        self._upsert_manifest(
            run_id,
            candidate,
            html_sha256=None,
            status="failed",
            article_key=None,
        )

    def update_metadata_only(
        self,
        run_id: str,
        candidate: SourceCandidate,
        *,
        html_sha256: str,
        article_key: str | None,
    ) -> None:
        self._upsert_manifest(
            run_id,
            candidate,
            html_sha256=html_sha256,
            status="processed",
            article_key=article_key,
        )

    def _upsert_manifest(
        self,
        run_id: str,
        candidate: SourceCandidate,
        *,
        html_sha256: str | None,
        status: str,
        article_key: str | None,
    ) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO source_manifest VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                candidate.relative_html_path,
                candidate.html_size,
                candidate.html_mtime_ns,
                candidate.relative_image_path,
                candidate.image_size,
                candidate.image_mtime_ns,
                html_sha256,
                EXTRACTOR_VERSION,
                status,
                article_key,
                run_id,
            ],
        )


def process_candidates(
    config: PipelineConfig,
    candidates: Iterable[SourceCandidate],
    run_id: str,
) -> ProcessSummary:
    """Extract candidates incrementally in bounded committed batches."""

    counters = {
        "discovered": 0,
        "new": 0,
        "changed": 0,
        "metadata_only": 0,
        "unchanged": 0,
        "succeeded": 0,
        "failed": 0,
    }
    affected: set[str] = set()
    iterator = iter(candidates)
    with PipelineState.open(config.state_db) as state:
        while batch := tuple(_take(iterator, config.batch_size)):
            state.connection.execute("BEGIN TRANSACTION")
            try:
                for candidate in batch:
                    counters["discovered"] += 1
                    status = state.classify(candidate, EXTRACTOR_VERSION)
                    if status.kind == "unchanged":
                        counters["unchanged"] += 1
                        continue
                    if status.kind == "stale_extractor":
                        counters["unchanged"] += 1
                        continue
                    if status.kind == "new":
                        counters["new"] += 1
                    try:
                        article = extract_article(candidate, config.source_root)
                    except ExtractionError as error:
                        if status.old_article_key:
                            affected.add(status.old_article_key)
                        state.store_failure(run_id, candidate, error)
                        counters["failed"] += 1
                        continue

                    if (
                        status.old_html_sha256
                        and article.source_html_sha256 == status.old_html_sha256
                    ):
                        state.update_metadata_only(
                            run_id,
                            candidate,
                            html_sha256=article.source_html_sha256,
                            article_key=status.old_article_key,
                        )
                        counters["metadata_only"] += 1
                        continue

                    if status.kind == "stat_changed":
                        counters["changed"] += 1
                    if status.old_article_key:
                        affected.add(status.old_article_key)
                    affected.add(state.store_success(run_id, candidate, article))
                    counters["succeeded"] += 1
            except Exception:
                state.connection.execute("ROLLBACK")
                raise
            else:
                state.connection.execute("COMMIT")

    return ProcessSummary(
        **counters,
        affected_article_keys=tuple(sorted(affected)),
    )


def _take(iterator: Iterator[SourceCandidate], count: int) -> Iterator[SourceCandidate]:
    for _ in range(count):
        try:
            yield next(iterator)
        except StopIteration:
            return


def _provisional_article_key(article: ExtractedArticle) -> str:
    if article.wsj_article_id:
        return f"wsj:{article.wsj_article_id.strip()}"
    if article.canonical_url:
        digest = hashlib.sha256(article.canonical_url.encode()).hexdigest()
        return f"url:{digest}"
    return f"html:{article.source_html_sha256}"


def _json_compatible(value: Any) -> Any:
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_compatible(item) for item in value]
    return value
