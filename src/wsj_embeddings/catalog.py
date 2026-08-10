"""Minimal versioned DuckDB catalog for article-text embeddings."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

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
_KEY_CONSTRAINTS = {
    ("metadata", "PRIMARY KEY", ("key",)),
    ("embedding_configurations", "PRIMARY KEY", ("configuration_id",)),
    ("runs", "PRIMARY KEY", ("run_id",)),
    (
        "embeddings",
        "PRIMARY KEY",
        ("article_id", "modality", "configuration_id"),
    ),
}
_EXPECTED_CONSTRAINTS = Counter(
    (table, constraint_type, columns, None, None, ())
    for table, constraint_type, columns in _KEY_CONSTRAINTS
)
_EXPECTED_CONSTRAINTS.update(
    (
        table,
        "NOT NULL",
        (name,),
        None,
        None,
        (),
    )
    for table, columns in _TABLE_COLUMNS.items()
    for name, _data_type, not_null, _default, _primary_key in columns
    if not_null
)
_CATALOG_DIRECTORY_FLAGS = (
    os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
)
_CATALOG_READ_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
_CATALOG_WRITE_FLAGS = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)


class EmbeddingCatalogError(RuntimeError):
    """Raised when the embedding catalog cannot be safely published."""


def _raise_unsafe_catalog_path() -> None:
    raise EmbeddingCatalogError(
        "unsafe embedding catalog path; move derived output aside "
        "or choose a fresh output root"
    )


def _prepare_catalog_path(
    path: Path,
    *,
    create: bool,
) -> tuple[int, tuple[int, int] | None]:
    """Anchor one safe embedding-catalog leaf below its parent descriptor."""

    absolute_path = Path(os.path.abspath(path))
    if (
        not absolute_path.name
        or absolute_path.name in {".", ".."}
        or Path(absolute_path.name).name != absolute_path.name
    ):
        _raise_unsafe_catalog_path()
    if create:
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        parent_descriptor = os.open(absolute_path.parent, _CATALOG_DIRECTORY_FLAGS)
    except OSError:
        _raise_unsafe_catalog_path()
    try:
        try:
            leaf_stat = os.stat(
                absolute_path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            if not create:
                _raise_unsafe_catalog_path()
            return parent_descriptor, None
        if not stat.S_ISREG(leaf_stat.st_mode) or leaf_stat.st_nlink != 1:
            _raise_unsafe_catalog_path()
        return parent_descriptor, (leaf_stat.st_dev, leaf_stat.st_ino)
    except BaseException:
        os.close(parent_descriptor)
        raise


def _verify_catalog_leaf(
    parent_descriptor: int,
    leaf_name: str,
    identity: tuple[int, int],
) -> None:
    """Reject a catalog leaf replaced while DuckDB opens it."""

    try:
        leaf_stat = os.stat(
            leaf_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except OSError:
        _raise_unsafe_catalog_path()
    if (
        not stat.S_ISREG(leaf_stat.st_mode)
        or leaf_stat.st_nlink != 1
        or (leaf_stat.st_dev, leaf_stat.st_ino) != identity
    ):
        _raise_unsafe_catalog_path()


def _open_catalog_leaf_descriptor(
    parent_descriptor: int,
    leaf_name: str,
    identity: tuple[int, int],
    *,
    read_only: bool,
) -> int:
    """Open the verified catalog inode without following a leaf symlink."""

    try:
        descriptor = os.open(
            leaf_name,
            _CATALOG_READ_FLAGS if read_only else _CATALOG_WRITE_FLAGS,
            dir_fd=parent_descriptor,
        )
    except OSError:
        _raise_unsafe_catalog_path()
    leaf_stat = os.fstat(descriptor)
    if (
        not stat.S_ISREG(leaf_stat.st_mode)
        or leaf_stat.st_nlink != 1
        or (leaf_stat.st_dev, leaf_stat.st_ino) != identity
    ):
        os.close(descriptor)
        _raise_unsafe_catalog_path()
    return descriptor


def _create_fresh_catalog_leaf(
    parent_descriptor: int,
    leaf_name: str,
) -> tuple[int, int]:
    """Create a fresh catalog atomically without overwriting a raced-in leaf."""

    temporary_name = f".{leaf_name}.{uuid4().hex}.tmp"
    temporary_path = f"/proc/self/fd/{parent_descriptor}/{temporary_name}"
    try:
        duckdb.connect(temporary_path).close()
        temporary_stat = os.stat(
            temporary_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(temporary_stat.st_mode) or temporary_stat.st_nlink != 1:
            _raise_unsafe_catalog_path()
        try:
            os.link(
                temporary_name,
                leaf_name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError:
            _raise_unsafe_catalog_path()
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name, dir_fd=parent_descriptor)
    leaf_stat = os.stat(leaf_name, dir_fd=parent_descriptor, follow_symlinks=False)
    if not stat.S_ISREG(leaf_stat.st_mode) or leaf_stat.st_nlink != 1:
        _raise_unsafe_catalog_path()
    return leaf_stat.st_dev, leaf_stat.st_ino


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

    def __init__(
        self,
        connection: duckdb.DuckDBPyConnection,
        parent_descriptor: int | None = None,
        leaf_descriptor: int | None = None,
    ) -> None:
        self.connection = connection
        self._parent_descriptor = parent_descriptor
        self._leaf_descriptor = leaf_descriptor

    @classmethod
    @contextmanager
    def open(cls, path: Path) -> Iterator[EmbeddingCatalog]:
        """Open a new catalog or reject every incompatible existing file."""

        parent_descriptor, identity = _prepare_catalog_path(path, create=True)
        is_fresh = identity is None
        connection: duckdb.DuckDBPyConnection | None = None
        leaf_descriptor: int | None = None
        try:
            if identity is None:
                identity = _create_fresh_catalog_leaf(parent_descriptor, path.name)
            else:
                with cls._read_only_anchored(
                    parent_descriptor,
                    path.name,
                    identity,
                ):
                    pass
                _verify_catalog_leaf(parent_descriptor, path.name, identity)
            leaf_descriptor = _open_catalog_leaf_descriptor(
                parent_descriptor,
                path.name,
                identity,
                read_only=False,
            )
            connection = duckdb.connect(f"/proc/self/fd/{leaf_descriptor}")
            _verify_catalog_leaf(parent_descriptor, path.name, identity)
        except BaseException:
            if connection is not None:
                connection.close()
            if leaf_descriptor is not None:
                os.close(leaf_descriptor)
            os.close(parent_descriptor)
            raise
        assert connection is not None
        catalog = cls(connection, parent_descriptor, leaf_descriptor)
        try:
            if is_fresh:
                catalog.ensure_schema()
            else:
                catalog.validate_schema()
            yield catalog
        finally:
            catalog.close()

    @classmethod
    @contextmanager
    def _read_only_anchored(
        cls,
        parent_descriptor: int,
        leaf_name: str,
        identity: tuple[int, int],
    ) -> Iterator[EmbeddingCatalog]:
        """Inspect one existing catalog through its original path anchor."""

        connection: duckdb.DuckDBPyConnection | None = None
        leaf_descriptor: int | None = None
        try:
            leaf_descriptor = _open_catalog_leaf_descriptor(
                parent_descriptor,
                leaf_name,
                identity,
                read_only=True,
            )
            connection = duckdb.connect(
                f"/proc/self/fd/{leaf_descriptor}", read_only=True
            )
            _verify_catalog_leaf(parent_descriptor, leaf_name, identity)
        except BaseException:
            if connection is not None:
                connection.close()
            if leaf_descriptor is not None:
                os.close(leaf_descriptor)
            raise
        assert connection is not None
        catalog = cls(connection, leaf_descriptor=leaf_descriptor)
        try:
            catalog.validate_schema()
            yield catalog
        finally:
            catalog.close()

    @classmethod
    @contextmanager
    def read_only(cls, path: Path) -> Iterator[EmbeddingCatalog]:
        """Open only a compatible existing catalog without mutating it."""

        parent_descriptor, identity = _prepare_catalog_path(path, create=False)
        try:
            assert identity is not None
            with cls._read_only_anchored(
                parent_descriptor,
                path.name,
                identity,
            ) as catalog:
                yield catalog
        finally:
            os.close(parent_descriptor)

    def close(self) -> None:
        """Close the catalog and its anchored filesystem descriptors."""

        self.connection.close()
        if self._leaf_descriptor is not None:
            os.close(self._leaf_descriptor)
            self._leaf_descriptor = None
        if self._parent_descriptor is not None:
            os.close(self._parent_descriptor)
            self._parent_descriptor = None

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
        constraints = Counter(
            (
                row[0],
                row[1],
                tuple(row[2] or ()),
                row[3],
                row[4],
                tuple(row[5] or ()),
            )
            for row in self.connection.execute(
                """
                SELECT table_name, constraint_type, constraint_column_names,
                       expression, referenced_table, referenced_column_names
                FROM duckdb_constraints()
                WHERE database_name = current_database()
                  AND schema_name = 'main'
                """
            ).fetchall()
        )
        indexes = self.connection.execute(
            """
            SELECT table_name, index_name, is_unique, expressions
            FROM duckdb_indexes()
            """
        ).fetchall()
        if (
            table_columns != _TABLE_COLUMNS
            or constraints != _EXPECTED_CONSTRAINTS
            or indexes
        ):
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
