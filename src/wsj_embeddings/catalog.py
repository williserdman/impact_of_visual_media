"""Minimal versioned DuckDB catalog for article-text embeddings."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import duckdb

from wsj_embeddings.models import EmbeddingProfile

EMBEDDING_CATALOG_SCHEMA_VERSION = "1"

_EMBEDDING_CATALOG_TABLES = {
    "embedding_configurations",
    "embeddings",
    "metadata",
    "runs",
}
_TABLE_COLUMNS = {
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


class EmbeddingCatalogError(RuntimeError):
    """Raised when the embedding catalog cannot be safely published."""


def configuration_id(profile: EmbeddingProfile) -> str:
    """Return the stable identity for every public profile field."""

    canonical = json.dumps(
        asdict(profile),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


class EmbeddingCatalog:
    """Own the transaction used to publish one bounded embedding batch."""

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self.connection = connection

    @classmethod
    @contextmanager
    def open(cls, path: Path) -> Iterator[EmbeddingCatalog]:
        """Open a new catalog or reject every incompatible existing file."""

        is_existing = path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = duckdb.connect(str(path))
        catalog = cls(connection)
        try:
            if is_existing:
                catalog.validate_schema()
            else:
                catalog.ensure_schema()
            yield catalog
        finally:
            connection.close()

    @classmethod
    @contextmanager
    def read_only(cls, path: Path) -> Iterator[EmbeddingCatalog]:
        """Open only a compatible existing catalog without mutating it."""

        if not path.is_file():
            cls._raise_unsupported_schema()
        connection = duckdb.connect(str(path), read_only=True)
        catalog = cls(connection)
        try:
            catalog.validate_schema()
            yield catalog
        finally:
            connection.close()

    def ensure_schema(self) -> None:
        """Install the initial exact tables for a fresh downstream catalog."""

        self.connection.execute("BEGIN TRANSACTION")
        try:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key VARCHAR PRIMARY KEY,
                    value VARCHAR NOT NULL
                )
                """
            )
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_configurations (
                    configuration_id VARCHAR PRIMARY KEY,
                    model VARCHAR NOT NULL,
                    task VARCHAR NOT NULL,
                    dimensions INTEGER NOT NULL,
                    output_type VARCHAR NOT NULL,
                    normalization VARCHAR NOT NULL
                )
                """
            )
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id VARCHAR PRIMARY KEY,
                    configuration_id VARCHAR NOT NULL,
                    articles INTEGER NOT NULL,
                    embeddings INTEGER NOT NULL,
                    started_at TIMESTAMP WITH TIME ZONE NOT NULL
                )
                """
            )
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    article_id VARCHAR NOT NULL,
                    modality VARCHAR NOT NULL,
                    configuration_id VARCHAR NOT NULL,
                    published_at_utc TIMESTAMP WITH TIME ZONE NOT NULL,
                    publication_date_new_york DATE NOT NULL,
                    dimensions INTEGER NOT NULL,
                    input_sha256 VARCHAR NOT NULL,
                    stored_vector_sha256 VARCHAR NOT NULL,
                    vector FLOAT[2048] NOT NULL,
                    PRIMARY KEY (article_id, modality, configuration_id)
                )
                """
            )
            self.connection.execute(
                """
                INSERT INTO metadata (key, value)
                VALUES ('schema_version', ?)
                ON CONFLICT (key) DO NOTHING
                """,
                [EMBEDDING_CATALOG_SCHEMA_VERSION],
            )
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise
        else:
            self.connection.execute("COMMIT")

    def validate_schema(self) -> None:
        """Reject every existing catalog outside the supported exact contract."""

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
        if set(table_types) != _EMBEDDING_CATALOG_TABLES or set(
            table_types.values()
        ) != {"BASE TABLE"}:
            self._raise_unsupported_schema()
        table_columns = {
            table: tuple(
                (row[1], row[2], row[3], row[4], row[5])
                for row in self.connection.execute(
                    f"PRAGMA table_info('{table}')"
                ).fetchall()
            )
            for table in _EMBEDDING_CATALOG_TABLES
        }
        if table_columns != _TABLE_COLUMNS:
            self._raise_unsupported_schema()
        try:
            row = self.connection.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            ).fetchone()
        except duckdb.Error:
            row = None
        if row is None or row[0] != EMBEDDING_CATALOG_SCHEMA_VERSION:
            self._raise_unsupported_schema()

    @staticmethod
    def _raise_unsupported_schema() -> None:
        raise EmbeddingCatalogError(
            "unsupported embedding catalog schema; move derived output aside "
            "or choose a fresh output root"
        )

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Commit or roll back one bounded coordinator publication."""

        self.connection.execute("BEGIN TRANSACTION")
        try:
            yield
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise
        else:
            self.connection.execute("COMMIT")
