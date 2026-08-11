"""Versioned DuckDB catalog for source and derived article embeddings."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import duckdb

from wsj_embeddings.long_text import LongTextPart
from wsj_embeddings.models import (
    EmbeddingModality,
    EmbeddingProfile,
    EmbeddingRunResult,
    SupersessionReason,
    WorkState,
)

EMBEDDING_CATALOG_SCHEMA_VERSION = "9"

_EMBEDDING_CATALOG_TABLES = {
    "article_text_aggregation_provenance",
    "embedding_configurations",
    "embedding_generation_history",
    "embedding_work_items",
    "embeddings",
    "long_text_parts",
    "long_text_part_generations",
    "metadata",
    "multimodal_embedding_provenance",
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
_KEY_CONSTRAINTS = {
    ("metadata", "PRIMARY KEY", ("key",)),
    ("embedding_configurations", "PRIMARY KEY", ("configuration_id",)),
    ("runs", "PRIMARY KEY", ("run_id",)),
    (
        "embedding_work_items",
        "PRIMARY KEY",
        ("article_id", "modality", "configuration_id"),
    ),
    (
        "embeddings",
        "PRIMARY KEY",
        ("article_id", "modality", "configuration_id"),
    ),
    (
        "embedding_generation_history",
        "PRIMARY KEY",
        ("article_id", "modality", "configuration_id", "generation_run_id"),
    ),
    (
        "multimodal_embedding_provenance",
        "PRIMARY KEY",
        ("article_id", "configuration_id", "generation_run_id"),
    ),
    (
        "long_text_parts",
        "PRIMARY KEY",
        (
            "article_id",
            "configuration_id",
            "article_input_sha256",
            "part_index",
        ),
    ),
    (
        "long_text_part_generations",
        "PRIMARY KEY",
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
        "PRIMARY KEY",
        ("article_id", "configuration_id", "generation_run_id", "part_index"),
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
_RUN_COUNT_COLUMNS = frozenset(
    {
        "reused",
        "attempted",
        "succeeded",
        "retryable",
        "terminal",
        "interrupted",
        "header_absent",
        "header_failed",
    }
)


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
    if create:
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        parent_descriptor = os.open(absolute_path.parent, _CATALOG_DIRECTORY_FLAGS)
    except OSError:
        _raise_unsafe_catalog_path()
    try:
        identity = _prepare_catalog_leaf(
            parent_descriptor,
            absolute_path.name,
            create=create,
        )
        return parent_descriptor, identity
    except BaseException:
        os.close(parent_descriptor)
        raise


def _prepare_catalog_leaf(
    parent_descriptor: int,
    leaf_name: str,
    *,
    create: bool,
) -> tuple[int, int] | None:
    """Inspect one catalog leaf below an already anchored output directory."""

    if (
        not leaf_name
        or leaf_name in {".", ".."}
        or Path(leaf_name).name != leaf_name
        or not stat.S_ISDIR(os.fstat(parent_descriptor).st_mode)
    ):
        _raise_unsafe_catalog_path()
    try:
        leaf_stat = os.stat(
            leaf_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        if not create:
            _raise_unsafe_catalog_path()
        return None
    if not stat.S_ISREG(leaf_stat.st_mode) or leaf_stat.st_nlink != 1:
        _raise_unsafe_catalog_path()
    return leaf_stat.st_dev, leaf_stat.st_ino


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
        with cls._open_anchored_parent(
            parent_descriptor,
            path.name,
            identity,
        ) as catalog:
            yield catalog

    @classmethod
    @contextmanager
    def open_in(
        cls,
        output_directory_descriptor: int,
        leaf_name: str = "catalog.duckdb",
    ) -> Iterator[EmbeddingCatalog]:
        """Open a catalog below an already anchored embedding output root."""

        parent_descriptor = os.dup(output_directory_descriptor)
        try:
            identity = _prepare_catalog_leaf(
                parent_descriptor,
                leaf_name,
                create=True,
            )
        except BaseException:
            os.close(parent_descriptor)
            raise
        with cls._open_anchored_parent(
            parent_descriptor,
            leaf_name,
            identity,
        ) as catalog:
            yield catalog

    @classmethod
    @contextmanager
    def _open_anchored_parent(
        cls,
        parent_descriptor: int,
        leaf_name: str,
        identity: tuple[int, int] | None,
    ) -> Iterator[EmbeddingCatalog]:
        """Open one writable catalog while owning its parent descriptor."""

        is_fresh = identity is None
        connection: duckdb.DuckDBPyConnection | None = None
        leaf_descriptor: int | None = None
        try:
            if identity is None:
                identity = _create_fresh_catalog_leaf(parent_descriptor, leaf_name)
            else:
                with cls._read_only_anchored(
                    parent_descriptor,
                    leaf_name,
                    identity,
                ):
                    pass
                _verify_catalog_leaf(parent_descriptor, leaf_name, identity)
            leaf_descriptor = _open_catalog_leaf_descriptor(
                parent_descriptor,
                leaf_name,
                identity,
                read_only=False,
            )
            connection = duckdb.connect(f"/proc/self/fd/{leaf_descriptor}")
            _verify_catalog_leaf(parent_descriptor, leaf_name, identity)
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
                    observed_model VARCHAR NOT NULL,
                    observed_api_version VARCHAR NOT NULL,
                    task VARCHAR NOT NULL,
                    dimensions INTEGER NOT NULL,
                    output_type VARCHAR NOT NULL,
                    normalization VARCHAR NOT NULL,
                    tokenizer_revision VARCHAR NOT NULL,
                    tokenizer_engine VARCHAR NOT NULL,
                    context_token_limit INTEGER NOT NULL,
                    context_rules VARCHAR NOT NULL,
                    long_text_aggregation VARCHAR NOT NULL,
                    long_text_part_attempt_limit INTEGER NOT NULL,
                    image_input_rules VARCHAR NOT NULL,
                    image_transform VARCHAR NOT NULL,
                    multimodal_formula VARCHAR NOT NULL,
                    client_configuration_version VARCHAR NOT NULL
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
                    reused INTEGER NOT NULL,
                    attempted INTEGER NOT NULL,
                    succeeded INTEGER NOT NULL,
                    retryable INTEGER NOT NULL,
                    terminal INTEGER NOT NULL,
                    interrupted INTEGER NOT NULL,
                    header_absent INTEGER NOT NULL,
                    header_failed INTEGER NOT NULL,
                    started_at TIMESTAMP WITH TIME ZONE NOT NULL
                )
                """
            )
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_work_items (
                    article_id VARCHAR NOT NULL,
                    modality VARCHAR NOT NULL,
                    configuration_id VARCHAR NOT NULL,
                    source_relative_path VARCHAR,
                    input_sha256 VARCHAR,
                    state VARCHAR NOT NULL,
                    attempt_count INTEGER NOT NULL,
                    error_code VARCHAR,
                    status_code INTEGER,
                    retry_after_seconds DOUBLE,
                    last_run_id VARCHAR NOT NULL,
                    generation_run_id VARCHAR,
                    updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    PRIMARY KEY (article_id, modality, configuration_id)
                )
                """
            )
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    article_id VARCHAR NOT NULL,
                    modality VARCHAR NOT NULL,
                    configuration_id VARCHAR NOT NULL,
                    source_relative_path VARCHAR,
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
                CREATE TABLE IF NOT EXISTS embedding_generation_history (
                    article_id VARCHAR NOT NULL,
                    modality VARCHAR NOT NULL,
                    configuration_id VARCHAR NOT NULL,
                    generation_run_id VARCHAR NOT NULL,
                    source_relative_path VARCHAR,
                    input_sha256 VARCHAR NOT NULL,
                    stored_vector_sha256 VARCHAR NOT NULL,
                    superseded_run_id VARCHAR NOT NULL,
                    superseded_reason VARCHAR NOT NULL,
                    superseded_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    PRIMARY KEY (
                        article_id, modality, configuration_id, generation_run_id
                    )
                )
                """
            )
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS multimodal_embedding_provenance (
                    article_id VARCHAR NOT NULL,
                    configuration_id VARCHAR NOT NULL,
                    generation_run_id VARCHAR NOT NULL,
                    text_generation_run_id VARCHAR NOT NULL,
                    text_stored_vector_sha256 VARCHAR NOT NULL,
                    header_image_generation_run_id VARCHAR NOT NULL,
                    header_image_stored_vector_sha256 VARCHAR NOT NULL,
                    formula_version VARCHAR NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    PRIMARY KEY (
                        article_id, configuration_id, generation_run_id
                    )
                )
                """
            )
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS long_text_parts (
                    article_id VARCHAR NOT NULL,
                    configuration_id VARCHAR NOT NULL,
                    article_input_sha256 VARCHAR NOT NULL,
                    part_index INTEGER NOT NULL,
                    part_count INTEGER NOT NULL,
                    char_start BIGINT NOT NULL,
                    char_end BIGINT NOT NULL,
                    byte_start BIGINT NOT NULL,
                    byte_end BIGINT NOT NULL,
                    part_input_sha256 VARCHAR NOT NULL,
                    token_count INTEGER NOT NULL,
                    state VARCHAR NOT NULL,
                    attempt_count INTEGER NOT NULL,
                    error_code VARCHAR,
                    status_code INTEGER,
                    retry_after_seconds DOUBLE,
                    last_run_id VARCHAR NOT NULL,
                    generation_run_id VARCHAR,
                    stored_vector_sha256 VARCHAR,
                    vector FLOAT[2048],
                    updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    PRIMARY KEY (
                        article_id, configuration_id, article_input_sha256,
                        part_index
                    )
                )
                """
            )
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS long_text_part_generations (
                    article_id VARCHAR NOT NULL,
                    configuration_id VARCHAR NOT NULL,
                    article_input_sha256 VARCHAR NOT NULL,
                    part_index INTEGER NOT NULL,
                    generation_run_id VARCHAR NOT NULL,
                    part_count INTEGER NOT NULL,
                    char_start BIGINT NOT NULL,
                    char_end BIGINT NOT NULL,
                    byte_start BIGINT NOT NULL,
                    byte_end BIGINT NOT NULL,
                    part_input_sha256 VARCHAR NOT NULL,
                    token_count INTEGER NOT NULL,
                    stored_vector_sha256 VARCHAR NOT NULL,
                    vector FLOAT[2048] NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    PRIMARY KEY (
                        article_id, configuration_id, article_input_sha256,
                        part_index, generation_run_id
                    )
                )
                """
            )
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS article_text_aggregation_provenance (
                    article_id VARCHAR NOT NULL,
                    configuration_id VARCHAR NOT NULL,
                    generation_run_id VARCHAR NOT NULL,
                    part_index INTEGER NOT NULL,
                    article_input_sha256 VARCHAR NOT NULL,
                    part_count INTEGER NOT NULL,
                    part_generation_run_id VARCHAR NOT NULL,
                    part_input_sha256 VARCHAR NOT NULL,
                    token_count INTEGER NOT NULL,
                    part_stored_vector_sha256 VARCHAR NOT NULL,
                    aggregation_version VARCHAR NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    PRIMARY KEY (
                        article_id, configuration_id, generation_run_id,
                        part_index
                    )
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

    def begin_run(
        self,
        run_id: str,
        configuration_identifier: str,
        profile: EmbeddingProfile,
        article_count: int,
    ) -> None:
        """Persist one immutable configuration and content-free run record."""

        self.connection.execute(
            """
            INSERT INTO embedding_configurations (
                configuration_id, model, observed_model, observed_api_version,
                task, dimensions, output_type, normalization,
                tokenizer_revision, tokenizer_engine, context_token_limit,
                context_rules, long_text_aggregation,
                long_text_part_attempt_limit, image_input_rules, image_transform,
                multimodal_formula, client_configuration_version
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (configuration_id) DO NOTHING
            """,
            [
                configuration_identifier,
                profile.model,
                profile.observed_model,
                profile.observed_api_version,
                profile.task,
                profile.dimensions,
                profile.output_type,
                profile.normalization,
                profile.tokenizer_revision,
                profile.tokenizer_engine,
                profile.context_token_limit,
                profile.context_rules,
                profile.long_text_aggregation,
                profile.long_text_part_attempt_limit,
                profile.image_input_rules,
                profile.image_transform,
                profile.multimodal_formula,
                profile.client_configuration_version,
            ],
        )
        self.connection.execute(
            """
            INSERT INTO runs (
                run_id, configuration_id, articles, embeddings, reused,
                attempted, succeeded, retryable, terminal, interrupted,
                header_absent, header_failed, started_at
            )
            VALUES (?, ?, ?, 0, 0, 0, 0, 0, 0, 0, 0, 0, current_timestamp)
            """,
            [run_id, configuration_identifier, article_count],
        )

    def register_work(
        self,
        *,
        run_id: str,
        article_id: str,
        modality: str,
        configuration_identifier: str,
        source_relative_path: str | None,
        input_sha256: str,
        published_at_utc: object,
        publication_date_new_york: object,
        reprocess: bool,
    ) -> WorkState:
        """Register or recover one bounded item and invalidate only its vector."""

        row = self.connection.execute(
            """
            SELECT state, input_sha256
            FROM embedding_work_items
            WHERE article_id = ? AND modality = ? AND configuration_id = ?
            """,
            [article_id, modality, configuration_identifier],
        ).fetchone()
        if row is None:
            self.connection.execute(
                """
                INSERT INTO embedding_work_items
                VALUES (?, ?, ?, ?, ?, ?, 0, NULL, NULL, NULL, ?, NULL,
                        current_timestamp)
                """,
                [
                    article_id,
                    modality,
                    configuration_identifier,
                    source_relative_path,
                    input_sha256,
                    WorkState.QUEUED.value,
                    run_id,
                ],
            )
            return WorkState.QUEUED
        state = WorkState(str(row[0]))
        prior_input_sha256 = str(row[1])
        if reprocess or prior_input_sha256 != input_sha256:
            reason = (
                SupersessionReason.EXPLICIT_REPROCESS
                if reprocess
                else SupersessionReason.INPUT_CHANGED
            )
            self._invalidate_work_generation(
                run_id=run_id,
                article_id=article_id,
                modality=modality,
                configuration_identifier=configuration_identifier,
                reason=reason,
                replacement_source_relative_path=source_relative_path,
                replacement_input_sha256=input_sha256,
            )
            if modality in {
                EmbeddingModality.ARTICLE_TEXT.value,
                EmbeddingModality.HEADER_IMAGE.value,
            }:
                self._invalidate_work_generation(
                    run_id=run_id,
                    article_id=article_id,
                    modality=EmbeddingModality.MULTIMODAL_ARTICLE.value,
                    configuration_identifier=configuration_identifier,
                    reason=reason,
                    replacement_source_relative_path=None,
                    replacement_input_sha256=None,
                )
            return WorkState.QUEUED
        proven_success = state.is_reusable_success and bool(
            self.connection.execute(
                """
                SELECT count(*)
                FROM embeddings
                WHERE article_id = ? AND modality = ? AND configuration_id = ?
                  AND input_sha256 = ?
                """,
                [article_id, modality, configuration_identifier, input_sha256],
            ).fetchone()[0]
        )
        if proven_success:
            self.connection.execute(
                """
                UPDATE embeddings
                SET source_relative_path = ?, published_at_utc = ?,
                    publication_date_new_york = ?
                WHERE article_id = ? AND modality = ? AND configuration_id = ?
                """,
                [
                    source_relative_path,
                    published_at_utc,
                    publication_date_new_york,
                    article_id,
                    modality,
                    configuration_identifier,
                ],
            )
            self.connection.execute(
                """
                UPDATE embedding_work_items
                SET source_relative_path = ?, last_run_id = ?,
                    updated_at = current_timestamp
                WHERE article_id = ? AND modality = ? AND configuration_id = ?
                """,
                [
                    source_relative_path,
                    run_id,
                    article_id,
                    modality,
                    configuration_identifier,
                ],
            )
            return WorkState.SUCCEEDED
        if state is WorkState.IN_PROGRESS or state.is_reusable_success:
            if state.is_reusable_success and modality in {
                EmbeddingModality.ARTICLE_TEXT.value,
                EmbeddingModality.HEADER_IMAGE.value,
            }:
                self._invalidate_work_generation(
                    run_id=run_id,
                    article_id=article_id,
                    modality=EmbeddingModality.MULTIMODAL_ARTICLE.value,
                    configuration_identifier=configuration_identifier,
                    reason=SupersessionReason.INPUT_CHANGED,
                    replacement_source_relative_path=None,
                    replacement_input_sha256=None,
                )
            self.connection.execute(
                """
                UPDATE embedding_work_items
                SET state = ?, last_run_id = ?, generation_run_id = NULL,
                    updated_at = current_timestamp
                WHERE article_id = ? AND modality = ? AND configuration_id = ?
                """,
                [
                    WorkState.INTERRUPTED.value,
                    run_id,
                    article_id,
                    modality,
                    configuration_identifier,
                ],
            )
            self.increment_run(run_id, "interrupted")
            return WorkState.INTERRUPTED
        if state is WorkState.TERMINAL:
            self.connection.execute(
                """
                UPDATE embedding_work_items
                SET last_run_id = ?, updated_at = current_timestamp
                WHERE article_id = ? AND modality = ? AND configuration_id = ?
                """,
                [run_id, article_id, modality, configuration_identifier],
            )
            self.increment_run(run_id, "terminal")
            if modality == EmbeddingModality.HEADER_IMAGE.value:
                self.increment_run(run_id, "header_failed")
        return state

    def register_not_applicable(
        self,
        *,
        run_id: str,
        article_id: str,
        modality: str,
        configuration_identifier: str,
    ) -> WorkState:
        """Persist an optional modality that has no canonical local input."""

        row = self.connection.execute(
            """
            SELECT state
            FROM embedding_work_items
            WHERE article_id = ? AND modality = ? AND configuration_id = ?
            """,
            [article_id, modality, configuration_identifier],
        ).fetchone()
        if row is None:
            self.connection.execute(
                """
                INSERT INTO embedding_work_items
                VALUES (?, ?, ?, NULL, NULL, ?, 0, NULL, NULL, NULL, ?, NULL,
                        current_timestamp)
                """,
                [
                    article_id,
                    modality,
                    configuration_identifier,
                    WorkState.NOT_APPLICABLE.value,
                    run_id,
                ],
            )
            self.increment_run(run_id, "header_absent")
            return WorkState.NOT_APPLICABLE
        state = WorkState(str(row[0]))
        if state is not WorkState.NOT_APPLICABLE:
            modalities = [modality]
            if modality == EmbeddingModality.HEADER_IMAGE.value:
                modalities.append(EmbeddingModality.MULTIMODAL_ARTICLE.value)
            for invalidated_modality in modalities:
                self._invalidate_work_generation(
                    run_id=run_id,
                    article_id=article_id,
                    modality=invalidated_modality,
                    configuration_identifier=configuration_identifier,
                    reason=SupersessionReason.INPUT_CHANGED,
                    replacement_source_relative_path=None,
                    replacement_input_sha256=None,
                )
        self.connection.execute(
            """
            UPDATE embedding_work_items
            SET source_relative_path = NULL, input_sha256 = NULL, state = ?,
                attempt_count = 0, error_code = NULL, status_code = NULL,
                retry_after_seconds = NULL, last_run_id = ?,
                generation_run_id = NULL,
                updated_at = current_timestamp
            WHERE article_id = ? AND modality = ? AND configuration_id = ?
            """,
            [
                WorkState.NOT_APPLICABLE.value,
                run_id,
                article_id,
                modality,
                configuration_identifier,
            ],
        )
        self.increment_run(run_id, "header_absent")
        return WorkState.NOT_APPLICABLE

    def register_missing_header(
        self,
        *,
        run_id: str,
        article_id: str,
        configuration_identifier: str,
        source_relative_path: str,
    ) -> WorkState:
        """Persist the narrow retryable disposition for one unavailable image."""

        row = self.connection.execute(
            """
            SELECT state, source_relative_path, input_sha256, error_code
            FROM embedding_work_items
            WHERE article_id = ? AND modality = ? AND configuration_id = ?
            """,
            [
                article_id,
                EmbeddingModality.HEADER_IMAGE.value,
                configuration_identifier,
            ],
        ).fetchone()
        already_missing = row == (
            WorkState.RETRYABLE.value,
            source_relative_path,
            None,
            "missing_header_image",
        )
        if row is None:
            self.connection.execute(
                """
                INSERT INTO embedding_work_items
                VALUES (?, ?, ?, ?, NULL, ?, 0, ?, NULL, NULL, ?, NULL,
                        current_timestamp)
                """,
                [
                    article_id,
                    EmbeddingModality.HEADER_IMAGE.value,
                    configuration_identifier,
                    source_relative_path,
                    WorkState.RETRYABLE.value,
                    "missing_header_image",
                    run_id,
                ],
            )
        else:
            if not already_missing:
                for modality in (
                    EmbeddingModality.HEADER_IMAGE.value,
                    EmbeddingModality.MULTIMODAL_ARTICLE.value,
                ):
                    self._invalidate_work_generation(
                        run_id=run_id,
                        article_id=article_id,
                        modality=modality,
                        configuration_identifier=configuration_identifier,
                        reason=SupersessionReason.INPUT_CHANGED,
                        replacement_source_relative_path=(
                            source_relative_path
                            if modality == EmbeddingModality.HEADER_IMAGE.value
                            else None
                        ),
                        replacement_input_sha256=None,
                    )
            self.connection.execute(
                """
                UPDATE embedding_work_items
                SET source_relative_path = ?, input_sha256 = NULL, state = ?,
                    attempt_count = 0, error_code = ?, status_code = NULL,
                    retry_after_seconds = NULL, last_run_id = ?,
                    generation_run_id = NULL, updated_at = current_timestamp
                WHERE article_id = ? AND modality = ? AND configuration_id = ?
                """,
                [
                    source_relative_path,
                    WorkState.RETRYABLE.value,
                    "missing_header_image",
                    run_id,
                    article_id,
                    EmbeddingModality.HEADER_IMAGE.value,
                    configuration_identifier,
                ],
            )
        self.increment_run(run_id, "retryable")
        self.increment_run(run_id, "header_failed")
        return WorkState.RETRYABLE

    def _invalidate_work_generation(
        self,
        *,
        run_id: str,
        article_id: str,
        modality: str,
        configuration_identifier: str,
        reason: SupersessionReason,
        replacement_source_relative_path: str | None,
        replacement_input_sha256: str | None,
    ) -> None:
        """Archive, remove, and queue one exact existing generation."""

        self._archive_active_generation(
            run_id=run_id,
            article_id=article_id,
            modality=modality,
            configuration_identifier=configuration_identifier,
            reason=reason,
        )
        self.connection.execute(
            """
            DELETE FROM embeddings
            WHERE article_id = ? AND modality = ? AND configuration_id = ?
            """,
            [article_id, modality, configuration_identifier],
        )
        self.connection.execute(
            """
            UPDATE embedding_work_items
            SET source_relative_path = ?,
                input_sha256 = COALESCE(?, input_sha256), state = ?,
                attempt_count = 0, error_code = NULL, status_code = NULL,
                retry_after_seconds = NULL, last_run_id = ?,
                generation_run_id = NULL,
                updated_at = current_timestamp
            WHERE article_id = ? AND modality = ? AND configuration_id = ?
            """,
            [
                replacement_source_relative_path,
                replacement_input_sha256,
                WorkState.QUEUED.value,
                run_id,
                article_id,
                modality,
                configuration_identifier,
            ],
        )

    def _archive_active_generation(
        self,
        *,
        run_id: str,
        article_id: str,
        modality: str,
        configuration_identifier: str,
        reason: SupersessionReason,
    ) -> None:
        row = self.connection.execute(
            """
            SELECT w.generation_run_id, e.source_relative_path, e.input_sha256,
                   e.stored_vector_sha256
            FROM embeddings AS e
            JOIN embedding_work_items AS w
              USING (article_id, modality, configuration_id)
            WHERE e.article_id = ? AND e.modality = ?
              AND e.configuration_id = ?
            """,
            [article_id, modality, configuration_identifier],
        ).fetchone()
        if row is None:
            return
        (
            generation_run_id,
            source_relative_path,
            input_sha256,
            stored_vector_sha256,
        ) = row
        self.connection.execute(
            """
            INSERT INTO embedding_generation_history
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, current_timestamp)
            ON CONFLICT (
                article_id, modality, configuration_id, generation_run_id
            ) DO NOTHING
            """,
            [
                article_id,
                modality,
                configuration_identifier,
                generation_run_id,
                source_relative_path,
                input_sha256,
                stored_vector_sha256,
                run_id,
                reason.value,
            ],
        )

    def start_attempt(
        self,
        *,
        run_id: str,
        article_id: str,
        modality: str,
        configuration_identifier: str,
    ) -> None:
        """Checkpoint in-progress before any content read or adapter request."""

        self.connection.execute(
            """
            UPDATE embedding_work_items
            SET state = ?, attempt_count = attempt_count + 1,
                error_code = NULL, status_code = NULL,
                retry_after_seconds = NULL, last_run_id = ?,
                generation_run_id = NULL,
                updated_at = current_timestamp
            WHERE article_id = ? AND modality = ? AND configuration_id = ?
            """,
            [
                WorkState.IN_PROGRESS.value,
                run_id,
                article_id,
                modality,
                configuration_identifier,
            ],
        )
        self.increment_run(run_id, "attempted")

    def register_long_text_parts(
        self,
        *,
        run_id: str,
        article_id: str,
        configuration_identifier: str,
        article_input_sha256: str,
        parts: Sequence[LongTextPart],
        reprocess: bool,
    ) -> tuple[WorkState, ...]:
        """Register exact operational parts without creating observations."""

        states: list[WorkState] = []
        part_count = len(parts)
        for part in parts:
            row = self.connection.execute(
                """
                SELECT part_count, char_start, char_end, byte_start, byte_end,
                       part_input_sha256, token_count, state, generation_run_id,
                       stored_vector_sha256, vector
                FROM long_text_parts
                WHERE article_id = ? AND configuration_id = ?
                  AND article_input_sha256 = ? AND part_index = ?
                """,
                [
                    article_id,
                    configuration_identifier,
                    article_input_sha256,
                    part.index,
                ],
            ).fetchone()
            if row is None:
                self.connection.execute(
                    """
                    INSERT INTO long_text_parts (
                        article_id, configuration_id, article_input_sha256,
                        part_index, part_count, char_start, char_end,
                        byte_start, byte_end, part_input_sha256, token_count,
                        state, attempt_count, error_code, status_code,
                        retry_after_seconds, last_run_id, generation_run_id,
                        stored_vector_sha256, vector, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL,
                              NULL, NULL, ?, NULL, NULL, NULL, current_timestamp)
                    """,
                    [
                        article_id,
                        configuration_identifier,
                        article_input_sha256,
                        part.index,
                        part_count,
                        part.char_start,
                        part.char_end,
                        part.byte_start,
                        part.byte_end,
                        part.input_sha256,
                        part.token_count,
                        WorkState.QUEUED.value,
                        run_id,
                    ],
                )
                states.append(WorkState.QUEUED)
                continue
            if tuple(row[:7]) != (
                part_count,
                part.char_start,
                part.char_end,
                part.byte_start,
                part.byte_end,
                part.input_sha256,
                part.token_count,
            ):
                raise ValueError("long-text part identity changed within configuration")
            state = WorkState(str(row[7]))
            if reprocess:
                self.connection.execute(
                    """
                    UPDATE long_text_parts
                    SET state = ?, attempt_count = 0, error_code = NULL,
                        status_code = NULL, retry_after_seconds = NULL,
                        last_run_id = ?, generation_run_id = NULL,
                        stored_vector_sha256 = NULL, vector = NULL,
                        updated_at = current_timestamp
                    WHERE article_id = ? AND configuration_id = ?
                      AND article_input_sha256 = ? AND part_index = ?
                    """,
                    [
                        WorkState.QUEUED.value,
                        run_id,
                        article_id,
                        configuration_identifier,
                        article_input_sha256,
                        part.index,
                    ],
                )
                states.append(WorkState.QUEUED)
                continue
            proven_success = (
                state is WorkState.SUCCEEDED
                and row[8] is not None
                and row[9] is not None
                and row[10] is not None
            )
            if proven_success:
                self.connection.execute(
                    """
                    UPDATE long_text_parts
                    SET last_run_id = ?, updated_at = current_timestamp
                    WHERE article_id = ? AND configuration_id = ?
                      AND article_input_sha256 = ? AND part_index = ?
                    """,
                    [
                        run_id,
                        article_id,
                        configuration_identifier,
                        article_input_sha256,
                        part.index,
                    ],
                )
                states.append(WorkState.SUCCEEDED)
                continue
            if state in {WorkState.IN_PROGRESS, WorkState.SUCCEEDED}:
                state = WorkState.INTERRUPTED
                self.connection.execute(
                    """
                    UPDATE long_text_parts
                    SET state = ?, last_run_id = ?, generation_run_id = NULL,
                        stored_vector_sha256 = NULL, vector = NULL,
                        updated_at = current_timestamp
                    WHERE article_id = ? AND configuration_id = ?
                      AND article_input_sha256 = ? AND part_index = ?
                    """,
                    [
                        state.value,
                        run_id,
                        article_id,
                        configuration_identifier,
                        article_input_sha256,
                        part.index,
                    ],
                )
            else:
                self.connection.execute(
                    """
                    UPDATE long_text_parts
                    SET last_run_id = ?, updated_at = current_timestamp
                    WHERE article_id = ? AND configuration_id = ?
                      AND article_input_sha256 = ? AND part_index = ?
                    """,
                    [
                        run_id,
                        article_id,
                        configuration_identifier,
                        article_input_sha256,
                        part.index,
                    ],
                )
            states.append(state)
        return tuple(states)

    def start_long_text_part_attempt(
        self,
        *,
        run_id: str,
        article_id: str,
        configuration_identifier: str,
        article_input_sha256: str,
        part_index: int,
    ) -> None:
        """Checkpoint one part before its hosted request."""

        self.connection.execute(
            """
            UPDATE long_text_parts
            SET state = ?, attempt_count = attempt_count + 1,
                error_code = NULL, status_code = NULL,
                retry_after_seconds = NULL, last_run_id = ?,
                generation_run_id = NULL, stored_vector_sha256 = NULL,
                vector = NULL, updated_at = current_timestamp
            WHERE article_id = ? AND configuration_id = ?
              AND article_input_sha256 = ? AND part_index = ?
            """,
            [
                WorkState.IN_PROGRESS.value,
                run_id,
                article_id,
                configuration_identifier,
                article_input_sha256,
                part_index,
            ],
        )

    def record_long_text_part_failure(
        self,
        *,
        run_id: str,
        article_id: str,
        configuration_identifier: str,
        article_input_sha256: str,
        part_index: int,
        state: WorkState,
        error_code: str,
        status_code: int | None,
        retry_after_seconds: float | None,
    ) -> None:
        """Commit one content-free part failure."""

        if not state.is_failure:
            raise ValueError("part failure requires a failure state")
        self.connection.execute(
            """
            UPDATE long_text_parts
            SET state = ?, error_code = ?, status_code = ?,
                retry_after_seconds = ?, last_run_id = ?,
                generation_run_id = NULL, stored_vector_sha256 = NULL,
                vector = NULL, updated_at = current_timestamp
            WHERE article_id = ? AND configuration_id = ?
              AND article_input_sha256 = ? AND part_index = ?
            """,
            [
                state.value,
                error_code,
                status_code,
                retry_after_seconds,
                run_id,
                article_id,
                configuration_identifier,
                article_input_sha256,
                part_index,
            ],
        )

    def publish_long_text_part_success(
        self,
        *,
        run_id: str,
        article_id: str,
        configuration_identifier: str,
        article_input_sha256: str,
        part_index: int,
        stored_vector_sha256: str,
        vector: Sequence[float],
    ) -> None:
        """Commit one validated operational vector as its durable checkpoint."""

        self.connection.execute(
            """
            INSERT INTO long_text_part_generations
            SELECT article_id, configuration_id, article_input_sha256,
                   part_index, ?, part_count, char_start, char_end,
                   byte_start, byte_end, part_input_sha256, token_count,
                   ?, ?, current_timestamp
            FROM long_text_parts
            WHERE article_id = ? AND configuration_id = ?
              AND article_input_sha256 = ? AND part_index = ?
            """,
            [
                run_id,
                stored_vector_sha256,
                list(vector),
                article_id,
                configuration_identifier,
                article_input_sha256,
                part_index,
            ],
        )
        self.connection.execute(
            """
            UPDATE long_text_parts
            SET state = ?, error_code = NULL, status_code = NULL,
                retry_after_seconds = NULL, last_run_id = ?,
                generation_run_id = ?, stored_vector_sha256 = ?, vector = ?,
                updated_at = current_timestamp
            WHERE article_id = ? AND configuration_id = ?
              AND article_input_sha256 = ? AND part_index = ?
            """,
            [
                WorkState.SUCCEEDED.value,
                run_id,
                run_id,
                stored_vector_sha256,
                list(vector),
                article_id,
                configuration_identifier,
                article_input_sha256,
                part_index,
            ],
        )

    def long_text_part_generation(
        self,
        *,
        article_id: str,
        configuration_identifier: str,
        article_input_sha256: str,
        part_index: int,
    ) -> tuple[object, ...]:
        """Return one proven current part generation for aggregation."""

        row = self.connection.execute(
            """
            SELECT g.generation_run_id, g.stored_vector_sha256, g.vector
            FROM long_text_parts AS p
            JOIN long_text_part_generations AS g
              USING (article_id, configuration_id, article_input_sha256,
                     part_index, generation_run_id)
            WHERE p.article_id = ? AND p.configuration_id = ?
              AND p.article_input_sha256 = ? AND p.part_index = ?
              AND p.state = ?
            """,
            [
                article_id,
                configuration_identifier,
                article_input_sha256,
                part_index,
                WorkState.SUCCEEDED.value,
            ],
        ).fetchone()
        if row is None or any(value is None for value in row):
            raise ValueError("long-text part lacks a proven generation")
        return tuple(row)

    def long_text_part_attempt_count(
        self,
        *,
        article_id: str,
        configuration_identifier: str,
        article_input_sha256: str,
        part_index: int,
    ) -> int:
        """Return the committed request count for one exact operational part."""

        row = self.connection.execute(
            """
            SELECT attempt_count
            FROM long_text_parts
            WHERE article_id = ? AND configuration_id = ?
              AND article_input_sha256 = ? AND part_index = ?
            """,
            [
                article_id,
                configuration_identifier,
                article_input_sha256,
                part_index,
            ],
        ).fetchone()
        if row is None:
            raise ValueError("long-text part is not registered")
        return int(row[0])

    def has_terminal_long_text_part(
        self,
        *,
        article_id: str,
        configuration_identifier: str,
        article_input_sha256: str,
    ) -> bool:
        """Return whether an exact article has a terminal operational part."""

        return bool(
            self.connection.execute(
                """
                SELECT count(*)
                FROM long_text_parts
                WHERE article_id = ? AND configuration_id = ?
                  AND article_input_sha256 = ? AND state = ?
                """,
                [
                    article_id,
                    configuration_identifier,
                    article_input_sha256,
                    WorkState.TERMINAL.value,
                ],
            ).fetchone()[0]
        )

    def record_failure(
        self,
        *,
        run_id: str,
        article_id: str,
        modality: str,
        configuration_identifier: str,
        state: WorkState,
        error_code: str,
        status_code: int | None,
        retry_after_seconds: float | None,
    ) -> None:
        """Commit one content-free retryable or terminal checkpoint."""

        if not state.is_failure:
            raise ValueError("failure checkpoint requires a failure state")
        self.connection.execute(
            """
            UPDATE embedding_work_items
            SET state = ?, error_code = ?, status_code = ?,
                retry_after_seconds = ?, last_run_id = ?,
                generation_run_id = NULL,
                updated_at = current_timestamp
            WHERE article_id = ? AND modality = ? AND configuration_id = ?
            """,
            [
                state.value,
                error_code,
                status_code,
                retry_after_seconds,
                run_id,
                article_id,
                modality,
                configuration_identifier,
            ],
        )
        self.increment_run(run_id, state.value)

    def attempt_count(
        self,
        *,
        article_id: str,
        modality: str,
        configuration_identifier: str,
    ) -> int:
        """Return the committed bounded attempt count for one exact item."""

        row = self.connection.execute(
            """
            SELECT attempt_count
            FROM embedding_work_items
            WHERE article_id = ? AND modality = ? AND configuration_id = ?
            """,
            [article_id, modality, configuration_identifier],
        ).fetchone()
        assert row is not None
        return int(row[0])

    def publish_success(
        self,
        *,
        run_id: str,
        article_id: str,
        modality: str,
        configuration_identifier: str,
        source_relative_path: str | None,
        published_at_utc: object,
        publication_date_new_york: object,
        dimensions: int,
        input_sha256: str,
        stored_vector_sha256: str,
        vector: Sequence[float],
    ) -> None:
        """Publish one vector and its success checkpoint atomically."""

        self.connection.execute(
            """
            INSERT INTO embeddings
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (article_id, modality, configuration_id)
            DO UPDATE SET
                source_relative_path = excluded.source_relative_path,
                published_at_utc = excluded.published_at_utc,
                publication_date_new_york = excluded.publication_date_new_york,
                dimensions = excluded.dimensions,
                input_sha256 = excluded.input_sha256,
                stored_vector_sha256 = excluded.stored_vector_sha256,
                vector = excluded.vector
            """,
            [
                article_id,
                modality,
                configuration_identifier,
                source_relative_path,
                published_at_utc,
                publication_date_new_york,
                dimensions,
                input_sha256,
                stored_vector_sha256,
                list(vector),
            ],
        )
        self.connection.execute(
            """
            UPDATE embedding_work_items
            SET state = ?, source_relative_path = ?, input_sha256 = ?,
                error_code = NULL,
                status_code = NULL, retry_after_seconds = NULL,
                last_run_id = ?, generation_run_id = ?,
                updated_at = current_timestamp
            WHERE article_id = ? AND modality = ? AND configuration_id = ?
            """,
            [
                WorkState.SUCCEEDED.value,
                source_relative_path,
                input_sha256,
                run_id,
                run_id,
                article_id,
                modality,
                configuration_identifier,
            ],
        )
        self.connection.execute(
            """
            UPDATE runs
            SET embeddings = embeddings + 1, succeeded = succeeded + 1
            WHERE run_id = ?
            """,
            [run_id],
        )

    def publish_long_text_success(
        self,
        *,
        run_id: str,
        article_id: str,
        configuration_identifier: str,
        published_at_utc: object,
        publication_date_new_york: object,
        dimensions: int,
        input_sha256: str,
        stored_vector_sha256: str,
        vector: Sequence[float],
        aggregation_version: str,
        parts: Sequence[tuple[int, str, int, str, str]],
    ) -> None:
        """Atomically publish one aggregate and every content-free part link."""

        self.publish_success(
            run_id=run_id,
            article_id=article_id,
            modality=EmbeddingModality.ARTICLE_TEXT.value,
            configuration_identifier=configuration_identifier,
            source_relative_path=None,
            published_at_utc=published_at_utc,
            publication_date_new_york=publication_date_new_york,
            dimensions=dimensions,
            input_sha256=input_sha256,
            stored_vector_sha256=stored_vector_sha256,
            vector=vector,
        )
        part_count = len(parts)
        self.connection.executemany(
            """
            INSERT INTO article_text_aggregation_provenance
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, current_timestamp)
            """,
            [
                [
                    article_id,
                    configuration_identifier,
                    run_id,
                    part_index,
                    input_sha256,
                    part_count,
                    part_generation_run_id,
                    part_input_sha256,
                    token_count,
                    part_vector_sha256,
                    aggregation_version,
                ]
                for (
                    part_index,
                    part_input_sha256,
                    token_count,
                    part_generation_run_id,
                    part_vector_sha256,
                ) in parts
            ],
        )

    def multimodal_sources(
        self,
        *,
        article_id: str,
        configuration_identifier: str,
    ) -> tuple[object, ...] | None:
        """Return two current successful source generations under one config."""

        row = self.connection.execute(
            """
            SELECT text_work.generation_run_id,
                   text_vector.stored_vector_sha256, text_vector.vector,
                   image_work.generation_run_id,
                   image_vector.stored_vector_sha256, image_vector.vector
            FROM embedding_work_items AS text_work
            JOIN embeddings AS text_vector
              ON text_vector.article_id = text_work.article_id
             AND text_vector.modality = text_work.modality
             AND text_vector.configuration_id = text_work.configuration_id
             AND text_vector.input_sha256 = text_work.input_sha256
            JOIN embedding_work_items AS image_work
              ON image_work.article_id = text_work.article_id
             AND image_work.configuration_id = text_work.configuration_id
             AND image_work.modality = ?
             AND image_work.state = ?
             AND image_work.generation_run_id IS NOT NULL
            JOIN embeddings AS image_vector
              ON image_vector.article_id = image_work.article_id
             AND image_vector.modality = image_work.modality
             AND image_vector.configuration_id = image_work.configuration_id
             AND image_vector.input_sha256 = image_work.input_sha256
            WHERE text_work.article_id = ? AND text_work.modality = ?
              AND text_work.configuration_id = ? AND text_work.state = ?
              AND text_work.generation_run_id IS NOT NULL
            """,
            [
                EmbeddingModality.HEADER_IMAGE.value,
                WorkState.SUCCEEDED.value,
                article_id,
                EmbeddingModality.ARTICLE_TEXT.value,
                configuration_identifier,
                WorkState.SUCCEEDED.value,
            ],
        ).fetchone()
        return None if row is None else tuple(row)

    def multimodal_provenance_matches(
        self,
        *,
        article_id: str,
        configuration_identifier: str,
        generation_run_id: str,
        text_generation_run_id: str,
        text_stored_vector_sha256: str,
        header_image_generation_run_id: str,
        header_image_stored_vector_sha256: str,
        formula_version: str,
    ) -> bool:
        """Check the immutable provenance for one reusable composite."""

        row = self.connection.execute(
            """
            SELECT count(*)
            FROM multimodal_embedding_provenance
            WHERE article_id = ? AND configuration_id = ?
              AND generation_run_id = ? AND text_generation_run_id = ?
              AND text_stored_vector_sha256 = ?
              AND header_image_generation_run_id = ?
              AND header_image_stored_vector_sha256 = ?
              AND formula_version = ?
            """,
            [
                article_id,
                configuration_identifier,
                generation_run_id,
                text_generation_run_id,
                text_stored_vector_sha256,
                header_image_generation_run_id,
                header_image_stored_vector_sha256,
                formula_version,
            ],
        ).fetchone()
        return bool(row[0])

    def publish_multimodal_success(
        self,
        *,
        run_id: str,
        article_id: str,
        configuration_identifier: str,
        published_at_utc: object,
        publication_date_new_york: object,
        dimensions: int,
        input_sha256: str,
        stored_vector_sha256: str,
        vector: Sequence[float],
        text_generation_run_id: str,
        text_stored_vector_sha256: str,
        header_image_generation_run_id: str,
        header_image_stored_vector_sha256: str,
        formula_version: str,
    ) -> None:
        """Atomically publish a derived vector and immutable source linkage."""

        self.publish_success(
            run_id=run_id,
            article_id=article_id,
            modality=EmbeddingModality.MULTIMODAL_ARTICLE.value,
            configuration_identifier=configuration_identifier,
            source_relative_path=None,
            published_at_utc=published_at_utc,
            publication_date_new_york=publication_date_new_york,
            dimensions=dimensions,
            input_sha256=input_sha256,
            stored_vector_sha256=stored_vector_sha256,
            vector=vector,
        )
        self.connection.execute(
            """
            INSERT INTO multimodal_embedding_provenance
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, current_timestamp)
            """,
            [
                article_id,
                configuration_identifier,
                run_id,
                text_generation_run_id,
                text_stored_vector_sha256,
                header_image_generation_run_id,
                header_image_stored_vector_sha256,
                formula_version,
            ],
        )

    def increment_run(self, run_id: str, column: str) -> None:
        """Increment one whitelisted content-free run counter."""

        if column not in _RUN_COUNT_COLUMNS:
            raise AssertionError("unsupported run count")
        self.connection.execute(
            f"UPDATE runs SET {column} = {column} + 1 WHERE run_id = ?",
            [run_id],
        )

    def run_result(self, run_id: str) -> EmbeddingRunResult:
        """Read one completed or interrupted run's stable counters."""

        row = self.connection.execute(
            """
            SELECT articles, embeddings, reused, attempted, succeeded,
                   retryable, terminal, interrupted, header_absent, header_failed
            FROM runs
            WHERE run_id = ?
            """,
            [run_id],
        ).fetchone()
        assert row is not None
        return EmbeddingRunResult(*(int(value) for value in row))

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
