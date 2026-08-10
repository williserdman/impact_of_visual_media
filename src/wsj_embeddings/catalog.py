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
        """Create the parent and open the first embedding catalog version."""

        path.parent.mkdir(parents=True, exist_ok=True)
        connection = duckdb.connect(str(path))
        catalog = cls(connection)
        try:
            catalog.ensure_schema()
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
