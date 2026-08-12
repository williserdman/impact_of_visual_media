"""Versioned DuckDB catalog for source and derived article embeddings."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import duckdb

from wsj_embeddings.long_text import LongTextPart
from wsj_embeddings.models import (
    EmbeddingModality,
    EmbeddingProfile,
    EmbeddingRunResult,
    SupersessionReason,
    VectorDerivationKind,
    WorkState,
)
from wsj_embeddings.run_metrics import normalize_safe_usage

EMBEDDING_CATALOG_SCHEMA_VERSION = "17"

_EMBEDDING_CATALOG_TYPES = {
    "article_text_aggregation_provenance",
    "embedding_configurations",
    "embedding_generation_history",
    "embedding_work_storage",
    "embedding_storage",
    "full_run_articles",
    "hosted_request_reservations",
    "image_input_provenance",
    "long_text_parts",
    "long_text_part_generations",
    "metadata",
    "multimodal_embedding_provenance",
    "reconciliation_actions",
    "runs",
}
_EMBEDDING_CATALOG_TYPES = {
    name: "BASE TABLE" for name in _EMBEDDING_CATALOG_TYPES
} | {
    "embedding_work_items": "VIEW",
    "embeddings": "VIEW",
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
        ("client_api_contract_version", "VARCHAR", True, None, False),
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
        ("batch_max_items", "INTEGER", True, None, False),
        ("batch_max_estimated_tokens", "INTEGER", True, None, False),
        ("batch_max_encoded_bytes", "BIGINT", True, None, False),
        ("batch_max_response_bytes", "BIGINT", True, None, False),
        ("batch_max_concurrency", "INTEGER", True, None, False),
        ("batch_max_attempts", "INTEGER", True, None, False),
        ("batch_initial_backoff_seconds", "DOUBLE", True, None, False),
        ("batch_max_backoff_seconds", "DOUBLE", True, None, False),
        ("quota_max_requests", "INTEGER", True, None, False),
        ("quota_max_estimated_tokens", "INTEGER", True, None, False),
        ("quota_window_seconds", "INTEGER", True, None, False),
        ("image_input_rules", "VARCHAR", True, None, False),
        ("image_transform", "VARCHAR", True, None, False),
        ("multimodal_formula", "VARCHAR", True, None, False),
        ("client_configuration_version", "VARCHAR", True, None, False),
    ),
    "runs": (
        ("run_id", "VARCHAR", True, None, True),
        ("configuration_id", "VARCHAR", True, None, False),
        ("scope", "VARCHAR", True, None, False),
        ("status", "VARCHAR", True, None, False),
        ("discovery_complete", "BOOLEAN", True, None, False),
        ("reconciliation_complete", "BOOLEAN", True, None, False),
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
        ("hosted_requests", "INTEGER", True, None, False),
        ("hosted_retries", "INTEGER", True, None, False),
        ("usage_json", "VARCHAR", True, None, False),
        ("throttles", "INTEGER", True, None, False),
        ("elapsed_seconds", "DOUBLE", True, None, False),
        ("started_at", "TIMESTAMP WITH TIME ZONE", True, None, False),
        ("finished_at", "TIMESTAMP WITH TIME ZONE", False, None, False),
    ),
    "full_run_articles": (
        ("run_id", "VARCHAR", True, None, True),
        ("article_id", "VARCHAR", True, None, True),
        ("header_image_path", "VARCHAR", False, None, False),
    ),
    "hosted_request_reservations": (
        ("reservation_id", "VARCHAR", True, None, True),
        ("run_id", "VARCHAR", True, None, False),
        ("configuration_id", "VARCHAR", True, None, False),
        ("reserved_at", "TIMESTAMP WITH TIME ZONE", True, None, False),
        ("estimated_tokens", "INTEGER", True, None, False),
        ("observed_input_tokens", "INTEGER", False, None, False),
    ),
    "embedding_work_storage": (
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
    "embedding_storage": (
        ("article_id", "VARCHAR", True, None, True),
        ("modality", "VARCHAR", True, None, True),
        ("configuration_id", "VARCHAR", True, None, True),
        ("source_relative_path", "VARCHAR", False, None, False),
        ("published_at_utc", "TIMESTAMP WITH TIME ZONE", True, None, False),
        ("publication_date_new_york", "DATE", True, None, False),
        ("dimensions", "INTEGER", True, None, False),
        ("input_sha256", "VARCHAR", True, None, False),
        ("derivation_kind", "VARCHAR", True, None, False),
        ("raw_response_sha256", "VARCHAR", False, None, False),
        ("response_model", "VARCHAR", False, None, False),
        ("stored_vector_sha256", "VARCHAR", True, None, False),
        ("vector", "FLOAT[2048]", True, None, False),
    ),
    "reconciliation_actions": (
        ("run_id", "VARCHAR", True, None, True),
        ("article_id", "VARCHAR", True, None, True),
        ("modality", "VARCHAR", True, None, True),
        ("configuration_id", "VARCHAR", True, None, True),
        ("target_source_relative_path", "VARCHAR", False, None, False),
        ("target_state", "VARCHAR", True, None, False),
        ("expected_source_relative_path", "VARCHAR", False, None, False),
        ("expected_input_sha256", "VARCHAR", False, None, False),
        ("expected_state", "VARCHAR", True, None, False),
        ("expected_generation_run_id", "VARCHAR", False, None, False),
        ("expected_vector_source_relative_path", "VARCHAR", False, None, False),
        ("expected_vector_input_sha256", "VARCHAR", False, None, False),
        ("expected_derivation_kind", "VARCHAR", False, None, False),
        ("expected_raw_response_sha256", "VARCHAR", False, None, False),
        ("expected_response_model", "VARCHAR", False, None, False),
        ("expected_stored_vector_sha256", "VARCHAR", False, None, False),
        ("staged_at", "TIMESTAMP WITH TIME ZONE", True, None, False),
    ),
    "embedding_generation_history": (
        ("article_id", "VARCHAR", True, None, True),
        ("modality", "VARCHAR", True, None, True),
        ("configuration_id", "VARCHAR", True, None, True),
        ("generation_run_id", "VARCHAR", True, None, True),
        ("source_relative_path", "VARCHAR", False, None, False),
        ("input_sha256", "VARCHAR", True, None, False),
        ("derivation_kind", "VARCHAR", True, None, False),
        ("raw_response_sha256", "VARCHAR", False, None, False),
        ("response_model", "VARCHAR", False, None, False),
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
    "image_input_provenance": (
        ("article_id", "VARCHAR", True, None, True),
        ("configuration_id", "VARCHAR", True, None, True),
        ("generation_run_id", "VARCHAR", True, None, True),
        ("source_sha256", "VARCHAR", True, None, False),
        ("source_format", "VARCHAR", True, None, False),
        ("source_bytes", "BIGINT", True, None, False),
        ("source_width", "INTEGER", True, None, False),
        ("source_height", "INTEGER", True, None, False),
        ("source_frames", "INTEGER", True, None, False),
        ("embedded_input_sha256", "VARCHAR", True, None, False),
        ("embedded_format", "VARCHAR", True, None, False),
        ("embedded_bytes", "BIGINT", True, None, False),
        ("embedded_width", "INTEGER", True, None, False),
        ("embedded_height", "INTEGER", True, None, False),
        ("embedded_frames", "INTEGER", True, None, False),
        ("transform_id", "VARCHAR", True, None, False),
        ("rendition_relative_path", "VARCHAR", False, None, False),
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
        ("derivation_kind", "VARCHAR", False, None, False),
        ("raw_response_sha256", "VARCHAR", False, None, False),
        ("response_model", "VARCHAR", False, None, False),
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
        ("derivation_kind", "VARCHAR", True, None, False),
        ("raw_response_sha256", "VARCHAR", False, None, False),
        ("response_model", "VARCHAR", False, None, False),
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
_VIEW_COLUMNS = {
    "embedding_work_items": tuple(
        (name, data_type, False, default, False)
        for name, data_type, _not_null, default, _primary_key in _TABLE_COLUMNS[
            "embedding_work_storage"
        ]
    ),
    "embeddings": tuple(
        (name, data_type, False, default, False)
        for name, data_type, _not_null, default, _primary_key in _TABLE_COLUMNS[
            "embedding_storage"
        ]
    ),
}
_VIEW_SQL_FINGERPRINTS = {
    "embedding_work_items": (
        "b7e0432cc58fac7135221ff6a6fa06405ab953d02d62a78dc416fc99411019bf"
    ),
    "embeddings": (
        "a1e0f2b7089d7e41ff015280ba6da384b92bd67d29c7b416cbb71822f7abe159"
    ),
}
_KEY_CONSTRAINTS = {
    ("metadata", "PRIMARY KEY", ("key",)),
    ("embedding_configurations", "PRIMARY KEY", ("configuration_id",)),
    ("runs", "PRIMARY KEY", ("run_id",)),
    ("full_run_articles", "PRIMARY KEY", ("run_id", "article_id")),
    (
        "hosted_request_reservations",
        "PRIMARY KEY",
        ("reservation_id",),
    ),
    (
        "embedding_work_storage",
        "PRIMARY KEY",
        ("article_id", "modality", "configuration_id"),
    ),
    (
        "embedding_storage",
        "PRIMARY KEY",
        ("article_id", "modality", "configuration_id"),
    ),
    (
        "reconciliation_actions",
        "PRIMARY KEY",
        ("run_id", "article_id", "modality", "configuration_id"),
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
        "image_input_provenance",
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


@dataclass(frozen=True, slots=True)
class QuotaReservationDecision:
    """One atomic hosted-wave admission decision."""

    reservation_ids: tuple[str, ...]
    retry_after_seconds: float


def _normalize_view_sql(sql: str) -> str:
    """Normalize non-behavioral formatting before fingerprinting view SQL."""

    return " ".join(sql.strip().removesuffix(";").split())


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
            catalog.connection.execute("BEGIN TRANSACTION")
            catalog.validate_schema()
            yield catalog
        finally:
            with suppress(duckdb.Error):
                catalog.connection.execute("ROLLBACK")
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
                    client_api_contract_version VARCHAR NOT NULL,
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
                    batch_max_items INTEGER NOT NULL,
                    batch_max_estimated_tokens INTEGER NOT NULL,
                    batch_max_encoded_bytes BIGINT NOT NULL,
                    batch_max_response_bytes BIGINT NOT NULL,
                    batch_max_concurrency INTEGER NOT NULL,
                    batch_max_attempts INTEGER NOT NULL,
                    batch_initial_backoff_seconds DOUBLE NOT NULL,
                    batch_max_backoff_seconds DOUBLE NOT NULL,
                    quota_max_requests INTEGER NOT NULL,
                    quota_max_estimated_tokens INTEGER NOT NULL,
                    quota_window_seconds INTEGER NOT NULL,
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
                    scope VARCHAR NOT NULL,
                    status VARCHAR NOT NULL,
                    discovery_complete BOOLEAN NOT NULL,
                    reconciliation_complete BOOLEAN NOT NULL,
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
                    hosted_requests INTEGER NOT NULL,
                    hosted_retries INTEGER NOT NULL,
                    usage_json VARCHAR NOT NULL,
                    throttles INTEGER NOT NULL,
                    elapsed_seconds DOUBLE NOT NULL,
                    started_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    finished_at TIMESTAMP WITH TIME ZONE
                )
                """
            )
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS full_run_articles (
                    run_id VARCHAR NOT NULL,
                    article_id VARCHAR NOT NULL,
                    header_image_path VARCHAR,
                    PRIMARY KEY (run_id, article_id)
                )
                """
            )
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS hosted_request_reservations (
                    reservation_id VARCHAR PRIMARY KEY,
                    run_id VARCHAR NOT NULL,
                    configuration_id VARCHAR NOT NULL,
                    reserved_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    estimated_tokens INTEGER NOT NULL,
                    observed_input_tokens INTEGER
                )
                """
            )
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_work_storage (
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
                CREATE TABLE IF NOT EXISTS embedding_storage (
                    article_id VARCHAR NOT NULL,
                    modality VARCHAR NOT NULL,
                    configuration_id VARCHAR NOT NULL,
                    source_relative_path VARCHAR,
                    published_at_utc TIMESTAMP WITH TIME ZONE NOT NULL,
                    publication_date_new_york DATE NOT NULL,
                    dimensions INTEGER NOT NULL,
                    input_sha256 VARCHAR NOT NULL,
                    derivation_kind VARCHAR NOT NULL,
                    raw_response_sha256 VARCHAR,
                    response_model VARCHAR,
                    stored_vector_sha256 VARCHAR NOT NULL,
                    vector FLOAT[2048] NOT NULL,
                    PRIMARY KEY (article_id, modality, configuration_id)
                )
                """
            )
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS reconciliation_actions (
                    run_id VARCHAR NOT NULL,
                    article_id VARCHAR NOT NULL,
                    modality VARCHAR NOT NULL,
                    configuration_id VARCHAR NOT NULL,
                    target_source_relative_path VARCHAR,
                    target_state VARCHAR NOT NULL,
                    expected_source_relative_path VARCHAR,
                    expected_input_sha256 VARCHAR,
                    expected_state VARCHAR NOT NULL,
                    expected_generation_run_id VARCHAR,
                    expected_vector_source_relative_path VARCHAR,
                    expected_vector_input_sha256 VARCHAR,
                    expected_derivation_kind VARCHAR,
                    expected_raw_response_sha256 VARCHAR,
                    expected_response_model VARCHAR,
                    expected_stored_vector_sha256 VARCHAR,
                    staged_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    PRIMARY KEY (
                        run_id, article_id, modality, configuration_id
                    )
                )
                """
            )
            self.connection.execute(
                """
                CREATE VIEW embeddings AS
                SELECT storage.* FROM embedding_storage AS storage
                WHERE NOT EXISTS (
                    SELECT 1 FROM reconciliation_actions AS action
                    JOIN runs ON runs.run_id = action.run_id
                    WHERE runs.reconciliation_complete
                      AND action.article_id = storage.article_id
                      AND action.modality = storage.modality
                      AND action.configuration_id = storage.configuration_id
                )
                """
            )
            self.connection.execute(
                """
                CREATE VIEW embedding_work_items AS
                SELECT storage.* FROM embedding_work_storage AS storage
                WHERE NOT EXISTS (
                    SELECT 1 FROM reconciliation_actions AS action
                    JOIN runs ON runs.run_id = action.run_id
                    WHERE runs.reconciliation_complete
                      AND action.article_id = storage.article_id
                      AND action.modality = storage.modality
                      AND action.configuration_id = storage.configuration_id
                )
                UNION ALL
                SELECT action.article_id, action.modality,
                       action.configuration_id,
                       action.target_source_relative_path,
                       NULL AS input_sha256, action.target_state,
                       0 AS attempt_count, NULL AS error_code,
                       NULL AS status_code, NULL AS retry_after_seconds,
                       action.run_id AS last_run_id,
                       NULL AS generation_run_id, action.staged_at AS updated_at
                FROM reconciliation_actions AS action
                JOIN runs ON runs.run_id = action.run_id
                WHERE runs.reconciliation_complete
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
                    derivation_kind VARCHAR NOT NULL,
                    raw_response_sha256 VARCHAR,
                    response_model VARCHAR,
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
                CREATE TABLE IF NOT EXISTS image_input_provenance (
                    article_id VARCHAR NOT NULL,
                    configuration_id VARCHAR NOT NULL,
                    generation_run_id VARCHAR NOT NULL,
                    source_sha256 VARCHAR NOT NULL,
                    source_format VARCHAR NOT NULL,
                    source_bytes BIGINT NOT NULL,
                    source_width INTEGER NOT NULL,
                    source_height INTEGER NOT NULL,
                    source_frames INTEGER NOT NULL,
                    embedded_input_sha256 VARCHAR NOT NULL,
                    embedded_format VARCHAR NOT NULL,
                    embedded_bytes BIGINT NOT NULL,
                    embedded_width INTEGER NOT NULL,
                    embedded_height INTEGER NOT NULL,
                    embedded_frames INTEGER NOT NULL,
                    transform_id VARCHAR NOT NULL,
                    rendition_relative_path VARCHAR,
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
                    derivation_kind VARCHAR,
                    raw_response_sha256 VARCHAR,
                    response_model VARCHAR,
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
                    derivation_kind VARCHAR NOT NULL,
                    raw_response_sha256 VARCHAR,
                    response_model VARCHAR,
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
        if table_types != _EMBEDDING_CATALOG_TYPES:
            self._raise_unsupported_schema()
        table_columns = {
            table: tuple(
                (row[1], row[2], row[3], row[4], row[5])
                for row in self.connection.execute(
                    f"PRAGMA table_info('{table}')"
                ).fetchall()
            )
            for table, object_type in _EMBEDDING_CATALOG_TYPES.items()
            if object_type == "BASE TABLE"
        }
        view_columns = {
            view: tuple(
                (row[1], row[2], row[3], row[4], row[5])
                for row in self.connection.execute(
                    f"PRAGMA table_info('{view}')"
                ).fetchall()
            )
            for view in _VIEW_COLUMNS
        }
        view_sql_fingerprints = {
            str(view_name): hashlib.sha256(
                _normalize_view_sql(str(sql)).encode("utf-8")
            ).hexdigest()
            for view_name, sql in self.connection.execute(
                """
                SELECT view_name, sql FROM duckdb_views()
                WHERE database_name = current_database()
                  AND schema_name = 'main'
                  AND view_name IN ('embeddings', 'embedding_work_items')
                """
            ).fetchall()
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
            or view_columns != _VIEW_COLUMNS
            or view_sql_fingerprints != _VIEW_SQL_FINGERPRINTS
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
        *,
        scope: str = "limited",
    ) -> None:
        """Persist one immutable configuration and content-free run record."""

        self.connection.execute(
            """
            INSERT INTO embedding_configurations (
                configuration_id, model, observed_model,
                client_api_contract_version,
                task, dimensions, output_type, normalization,
                tokenizer_revision, tokenizer_engine, context_token_limit,
                context_rules, long_text_aggregation,
                long_text_part_attempt_limit, batch_max_items,
                batch_max_estimated_tokens, batch_max_encoded_bytes,
                batch_max_response_bytes,
                batch_max_concurrency, batch_max_attempts,
                batch_initial_backoff_seconds, batch_max_backoff_seconds,
                quota_max_requests, quota_max_estimated_tokens,
                quota_window_seconds,
                image_input_rules, image_transform,
                multimodal_formula, client_configuration_version
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (configuration_id) DO NOTHING
            """,
            [
                configuration_identifier,
                profile.model,
                profile.observed_model,
                profile.client_api_contract_version,
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
                profile.batch_max_items,
                profile.batch_max_estimated_tokens,
                profile.batch_max_encoded_bytes,
                profile.batch_max_response_bytes,
                profile.batch_max_concurrency,
                profile.batch_max_attempts,
                profile.batch_initial_backoff_seconds,
                profile.batch_max_backoff_seconds,
                profile.quota_max_requests,
                profile.quota_max_estimated_tokens,
                profile.quota_window_seconds,
                profile.image_input_rules,
                profile.image_transform,
                profile.multimodal_formula,
                profile.client_configuration_version,
            ],
        )
        self.connection.execute(
            """
            INSERT INTO runs (
                run_id, configuration_id, scope, status, discovery_complete,
                reconciliation_complete, articles, embeddings, reused,
                attempted, succeeded, retryable, terminal, interrupted,
                header_absent, header_failed, hosted_requests, hosted_retries,
                usage_json, throttles, elapsed_seconds, started_at, finished_at
            )
            VALUES (?, ?, ?, 'running', false, false, ?, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, '{}', 0, 0.0, current_timestamp, NULL)
            """,
            [run_id, configuration_identifier, scope, article_count],
        )

    def reserve_hosted_requests(
        self,
        *,
        run_id: str,
        configuration_identifier: str,
        estimated_tokens: Sequence[int],
        reserved_at: datetime,
        profile: EmbeddingProfile,
    ) -> QuotaReservationDecision:
        """Atomically reserve one bounded hosted wave or return its safe delay."""

        costs = tuple(estimated_tokens)
        if (
            not costs
            or len(costs) > profile.batch_max_concurrency
            or any(
                isinstance(cost, bool)
                or not isinstance(cost, int)
                or cost < 0
                or cost > profile.batch_max_estimated_tokens
                for cost in costs
            )
            or profile.quota_max_requests < 1
            or profile.quota_max_estimated_tokens < 1
            or profile.quota_window_seconds < 1
            or profile.batch_max_concurrency < 1
            or profile.batch_max_estimated_tokens < 1
        ):
            raise ValueError("hosted quota reservation is invalid")
        if configuration_identifier != configuration_id(profile):
            raise ValueError("hosted quota configuration does not match profile")
        if reserved_at.tzinfo is None or reserved_at.utcoffset() is None:
            raise ValueError("hosted quota timestamp must be timezone-aware")
        run = self.connection.execute(
            "SELECT configuration_id FROM runs WHERE run_id = ?",
            [run_id],
        ).fetchone()
        if run != (configuration_identifier,):
            raise ValueError("hosted quota run does not match configuration")

        window_start = reserved_at - timedelta(
            seconds=profile.quota_window_seconds
        )
        rows = self.connection.execute(
            """
            SELECT reservation_id, reserved_at,
                   greatest(estimated_tokens,
                            coalesce(observed_input_tokens, estimated_tokens))
            FROM hosted_request_reservations
            WHERE reserved_at > ? AND reserved_at <= ?
            ORDER BY reserved_at, reservation_id
            LIMIT ?
            """,
            [window_start, reserved_at, profile.quota_max_requests + 1],
        ).fetchall()
        if len(rows) > profile.quota_max_requests or any(
            int(row[2]) < 0 for row in rows
        ):
            raise EmbeddingCatalogError("invalid hosted quota reservation state")

        proposed_requests = len(costs)
        proposed_tokens = sum(costs)
        active_requests = len(rows)
        active_tokens = sum(int(row[2]) for row in rows)
        if (
            proposed_requests > profile.quota_max_requests
            or proposed_tokens > profile.quota_max_estimated_tokens
        ):
            raise ValueError("hosted wave exceeds quota policy")
        if (
            active_requests + proposed_requests <= profile.quota_max_requests
            and active_tokens + proposed_tokens
            <= profile.quota_max_estimated_tokens
        ):
            reservation_ids = tuple(uuid4().hex for _ in costs)
            self.connection.executemany(
                """
                INSERT INTO hosted_request_reservations (
                    reservation_id, run_id, configuration_id, reserved_at,
                    estimated_tokens, observed_input_tokens
                ) VALUES (?, ?, ?, ?, ?, NULL)
                """,
                [
                    (
                        reservation_id,
                        run_id,
                        configuration_identifier,
                        reserved_at,
                        cost,
                    )
                    for reservation_id, cost in zip(
                        reservation_ids, costs, strict=True
                    )
                ],
            )
            return QuotaReservationDecision(reservation_ids, 0.0)

        remaining_requests = active_requests
        remaining_tokens = active_tokens
        for _reservation_id, row_reserved_at, effective_tokens in rows:
            remaining_requests -= 1
            remaining_tokens -= int(effective_tokens)
            if (
                remaining_requests + proposed_requests
                <= profile.quota_max_requests
                and remaining_tokens + proposed_tokens
                <= profile.quota_max_estimated_tokens
            ):
                retry_at = row_reserved_at + timedelta(
                    seconds=profile.quota_window_seconds
                )
                return QuotaReservationDecision(
                    (), max(0.0, (retry_at - reserved_at).total_seconds())
                )
        raise EmbeddingCatalogError("invalid hosted quota reservation state")

    def reconcile_hosted_request_usage(
        self,
        *,
        reservation_ids: Sequence[str],
        observed_input_tokens: int,
    ) -> None:
        """Increase one wave's durable token debt from safe provider usage."""

        identifiers = tuple(reservation_ids)
        if (
            not identifiers
            or len(set(identifiers)) != len(identifiers)
            or isinstance(observed_input_tokens, bool)
            or not isinstance(observed_input_tokens, int)
            or observed_input_tokens < 0
            or observed_input_tokens > 1_000_000_000
        ):
            raise ValueError("hosted quota usage observation is invalid")
        rows = self.connection.execute(
            """
            SELECT reservation_id, estimated_tokens, observed_input_tokens
            FROM hosted_request_reservations
            WHERE reservation_id IN (SELECT unnest(?))
            ORDER BY reservation_id
            """,
            [list(identifiers)],
        ).fetchall()
        if len(rows) != len(identifiers):
            raise ValueError("hosted quota reservation is missing")
        # Provider usage is batch-wide. Assign any excess to the first stable
        # reservation while retaining every request's original estimate.
        estimated_total = sum(int(row[1]) for row in rows)
        effective_total = max(estimated_total, observed_input_tokens)
        excess = effective_total - estimated_total
        first_identifier = min(identifiers)
        for reservation_id, estimated, prior_observed in rows:
            observed = int(estimated) + (
                excess if reservation_id == first_identifier else 0
            )
            if prior_observed is not None:
                observed = max(observed, int(prior_observed))
            self.connection.execute(
                """
                UPDATE hosted_request_reservations
                SET observed_input_tokens = ?
                WHERE reservation_id = ?
                """,
                [observed, reservation_id],
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
                INSERT INTO embedding_work_storage
                VALUES (?, ?, ?, ?, ?, ?, 0, NULL, NULL, NULL, ?, NULL,
                        current_timestamp)
                """,
                [
                    article_id,
                    modality,
                    configuration_identifier,
                    source_relative_path,
                    input_sha256,
                    WorkState.ELIGIBLE.value,
                    run_id,
                ],
            )
            return WorkState.ELIGIBLE
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
            return WorkState.ELIGIBLE
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
                UPDATE embedding_storage
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
                UPDATE embedding_work_storage
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
                UPDATE embedding_work_storage
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
                UPDATE embedding_work_storage
                SET last_run_id = ?, updated_at = current_timestamp
                WHERE article_id = ? AND modality = ? AND configuration_id = ?
                """,
                [run_id, article_id, modality, configuration_identifier],
            )
            self.increment_run(run_id, "terminal")
            if modality == EmbeddingModality.HEADER_IMAGE.value:
                self.increment_run(run_id, "header_failed")
        return state

    def record_full_discovery_page(
        self,
        run_id: str,
        articles: Sequence[object],
    ) -> None:
        """Persist one bounded, content-free page of canonical identities."""

        self.connection.executemany(
            """
            INSERT INTO full_run_articles VALUES (?, ?, ?)
            ON CONFLICT (run_id, article_id) DO NOTHING
            """,
            [
                [run_id, article.article_id, article.header_image_path]
                for article in articles
            ],
        )
        self.connection.execute(
            """
            UPDATE runs
            SET articles = (
                SELECT count(*) FROM full_run_articles WHERE run_id = ?
            )
            WHERE run_id = ?
            """,
            [run_id, run_id],
        )

    def mark_full_discovery_complete(self, run_id: str) -> None:
        """Record successful exhaustion of the canonical lexical traversal."""

        self.connection.execute(
            "UPDATE runs SET discovery_complete = true WHERE run_id = ?",
            [run_id],
        )

    def full_run_article_ids(
        self,
        run_id: str,
        *,
        after_article_id: str | None,
        page_size: int,
    ) -> tuple[str, ...]:
        """Return one bounded lexical page from a completed full inventory."""

        if page_size < 1:
            raise ValueError("page size must be positive")
        if after_article_id is None:
            rows = self.connection.execute(
                """
                SELECT article_id FROM full_run_articles
                WHERE run_id = ? ORDER BY article_id LIMIT ?
                """,
                [run_id, page_size],
            ).fetchmany(page_size)
        else:
            rows = self.connection.execute(
                """
                SELECT article_id FROM full_run_articles
                WHERE run_id = ? AND article_id > ?
                ORDER BY article_id LIMIT ?
                """,
                [run_id, after_article_id, page_size],
            ).fetchmany(page_size)
        return tuple(str(row[0]) for row in rows)

    def stage_reconciliation_page(
        self,
        run_id: str,
        *,
        after_action_key: tuple[str, str, str] | None,
        page_size: int,
        before_commit: Callable[[int], None] | None = None,
    ) -> tuple[int, tuple[str, str, str] | None]:
        """Stage one bounded page of exact, still-invisible work actions."""

        if page_size < 1:
            raise ValueError("page size must be positive")
        run = self.connection.execute(
            """
            SELECT scope, discovery_complete FROM runs WHERE run_id = ?
            """,
            [run_id],
        ).fetchone()
        if run != ("full", True):
            raise EmbeddingCatalogError(
                "full reconciliation requires completed discovery"
            )
        cursor_clause = ""
        parameters: list[object] = [
            EmbeddingModality.HEADER_IMAGE.value,
            EmbeddingModality.HEADER_IMAGE.value,
            WorkState.NOT_APPLICABLE.value,
            WorkState.STALE_INPUT.value,
            run_id,
            EmbeddingModality.HEADER_IMAGE.value,
            EmbeddingModality.MULTIMODAL_ARTICLE.value,
            EmbeddingModality.HEADER_IMAGE.value,
        ]
        if after_action_key is not None:
            cursor_clause = """
                AND (work.article_id, work.modality, work.configuration_id)
                    > (?, ?, ?)
            """
            parameters.extend(after_action_key)
        parameters.append(page_size)
        rows = self.connection.execute(
            f"""
            SELECT work.article_id, work.modality, work.configuration_id,
                   CASE
                     WHEN seen.article_id IS NULL THEN NULL
                     WHEN work.modality = ? THEN seen.header_image_path
                     ELSE NULL
                   END AS target_source_relative_path,
                   CASE
                     WHEN seen.article_id IS NOT NULL
                      AND work.modality = ?
                      AND seen.header_image_path IS NULL
                     THEN ? ELSE ?
                   END AS target_state,
                   work.source_relative_path, work.input_sha256, work.state,
                   work.generation_run_id, vector.source_relative_path,
                   vector.input_sha256, vector.derivation_kind,
                   vector.raw_response_sha256, vector.response_model,
                   vector.stored_vector_sha256
            FROM embedding_work_storage AS work
            LEFT JOIN full_run_articles AS seen
              ON seen.run_id = ? AND seen.article_id = work.article_id
            LEFT JOIN embedding_storage AS vector
              ON vector.article_id = work.article_id
             AND vector.modality = work.modality
             AND vector.configuration_id = work.configuration_id
            WHERE (
                seen.article_id IS NULL
                OR (
                    work.modality = ?
                    AND work.source_relative_path IS DISTINCT FROM
                        seen.header_image_path
                )
                OR (
                    work.modality = ? AND EXISTS (
                        SELECT 1 FROM embedding_work_storage AS header
                        WHERE header.article_id = work.article_id
                          AND header.configuration_id = work.configuration_id
                          AND header.modality = ?
                          AND header.source_relative_path IS DISTINCT FROM
                              seen.header_image_path
                    )
                )
            )
            {cursor_clause}
            ORDER BY work.article_id, work.modality, work.configuration_id
            LIMIT ?
            """,
            parameters,
        ).fetchmany(page_size)
        if rows:
            self.connection.executemany(
                """
                INSERT INTO reconciliation_actions
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        current_timestamp)
                """,
                [[run_id, *row] for row in rows],
            )
        if before_commit is not None and rows:
            before_commit(len(rows))
        next_cursor = (
            tuple(str(value) for value in rows[-1][:3]) if rows else after_action_key
        )
        return len(rows), next_cursor

    def commit_reconciliation_visibility(self, run_id: str) -> None:
        """Atomically expose one fully staged, identity-bound action set."""

        mismatch = self.connection.execute(
            """
            SELECT count(*)
            FROM reconciliation_actions AS action
            LEFT JOIN embedding_work_storage AS work
              USING (article_id, modality, configuration_id)
            LEFT JOIN embedding_storage AS vector
              USING (article_id, modality, configuration_id)
            WHERE action.run_id = ? AND (
                work.article_id IS NULL
                OR work.source_relative_path IS DISTINCT FROM
                    action.expected_source_relative_path
                OR work.input_sha256 IS DISTINCT FROM action.expected_input_sha256
                OR work.state IS DISTINCT FROM action.expected_state
                OR work.generation_run_id IS DISTINCT FROM
                    action.expected_generation_run_id
                OR vector.source_relative_path IS DISTINCT FROM
                    action.expected_vector_source_relative_path
                OR vector.input_sha256 IS DISTINCT FROM
                    action.expected_vector_input_sha256
                OR vector.derivation_kind IS DISTINCT FROM
                    action.expected_derivation_kind
                OR vector.raw_response_sha256 IS DISTINCT FROM
                    action.expected_raw_response_sha256
                OR vector.response_model IS DISTINCT FROM
                    action.expected_response_model
                OR vector.stored_vector_sha256 IS DISTINCT FROM
                    action.expected_stored_vector_sha256
            )
            """,
            [run_id],
        ).fetchone()[0]
        if mismatch:
            raise EmbeddingCatalogError("reconciliation base identity changed")
        self.finish_run(run_id, reconciled=True)

    def compact_reconciliation_page(
        self,
        *,
        after_action_key: tuple[str, str, str, str] | None,
        page_size: int,
        before_commit: Callable[[int], None] | None = None,
    ) -> tuple[int, tuple[str, str, str, str] | None]:
        """Materialize one exact action-row page without changing public state."""

        if page_size < 1:
            raise ValueError("page size must be positive")
        cursor_clause = ""
        parameters: list[object] = []
        if after_action_key is not None:
            cursor_clause = """
              AND (action.run_id, action.article_id, action.modality,
                   action.configuration_id) > (?, ?, ?, ?)
            """
            parameters.extend(after_action_key)
        parameters.append(page_size)
        actions = self.connection.execute(
            f"""
            SELECT action.run_id, action.article_id, action.modality,
                   action.configuration_id,
                   action.target_source_relative_path, action.target_state,
                   action.expected_source_relative_path,
                   action.expected_input_sha256, action.expected_state,
                   action.expected_generation_run_id,
                   action.expected_vector_source_relative_path,
                   action.expected_vector_input_sha256,
                   action.expected_derivation_kind,
                   action.expected_raw_response_sha256,
                   action.expected_response_model,
                   action.expected_stored_vector_sha256
            FROM reconciliation_actions AS action
            JOIN runs ON runs.run_id = action.run_id
            WHERE runs.reconciliation_complete
            {cursor_clause}
            ORDER BY action.run_id, action.article_id, action.modality,
                     action.configuration_id
            LIMIT ?
            """,
            parameters,
        ).fetchmany(page_size)
        for action in actions:
            (
                action_run_id,
                article_id,
                modality,
                configuration_identifier,
                target_source_relative_path,
                target_state,
                expected_source_relative_path,
                expected_input_sha256,
                expected_state,
                expected_generation_run_id,
                expected_vector_source_relative_path,
                expected_vector_input_sha256,
                expected_derivation_kind,
                expected_raw_response_sha256,
                expected_response_model,
                expected_stored_vector_sha256,
            ) = action
            identity = self.connection.execute(
                """
                SELECT work.source_relative_path, work.input_sha256, work.state,
                       work.generation_run_id, vector.source_relative_path,
                       vector.input_sha256, vector.derivation_kind,
                       vector.raw_response_sha256, vector.response_model,
                       vector.stored_vector_sha256
                FROM embedding_work_storage AS work
                LEFT JOIN embedding_storage AS vector
                  USING (article_id, modality, configuration_id)
                WHERE work.article_id = ? AND work.modality = ?
                  AND work.configuration_id = ?
                """,
                [article_id, modality, configuration_identifier],
            ).fetchone()
            expected = (
                expected_source_relative_path,
                expected_input_sha256,
                expected_state,
                expected_generation_run_id,
                expected_vector_source_relative_path,
                expected_vector_input_sha256,
                expected_derivation_kind,
                expected_raw_response_sha256,
                expected_response_model,
                expected_stored_vector_sha256,
            )
            if identity != expected:
                raise EmbeddingCatalogError("reconciliation base identity changed")
            self._archive_active_generation_from_storage(
                run_id=str(action_run_id),
                article_id=str(article_id),
                modality=str(modality),
                configuration_identifier=str(configuration_identifier),
            )
            self.connection.execute(
                """
                DELETE FROM embedding_storage
                WHERE article_id = ? AND modality = ? AND configuration_id = ?
                """,
                [article_id, modality, configuration_identifier],
            )
            self.connection.execute(
                """
                UPDATE embedding_work_storage
                SET source_relative_path = ?, input_sha256 = NULL, state = ?,
                    attempt_count = 0, error_code = NULL, status_code = NULL,
                    retry_after_seconds = NULL, last_run_id = ?,
                    generation_run_id = NULL, updated_at = current_timestamp
                WHERE article_id = ? AND modality = ? AND configuration_id = ?
                """,
                [
                    target_source_relative_path,
                    target_state,
                    action_run_id,
                    article_id,
                    modality,
                    configuration_identifier,
                ],
            )
            self.connection.execute(
                """
                DELETE FROM reconciliation_actions
                WHERE run_id = ? AND article_id = ? AND modality = ?
                  AND configuration_id = ?
                """,
                [action_run_id, article_id, modality, configuration_identifier],
            )
        if before_commit is not None and actions:
            before_commit(len(actions))
        next_cursor = (
            tuple(str(value) for value in actions[-1][:4])
            if actions
            else after_action_key
        )
        return len(actions), next_cursor

    def discard_invisible_reconciliation_page(self, *, page_size: int) -> int:
        """Drop one bounded page staged by terminal non-visible runs."""

        if page_size < 1:
            raise ValueError("page size must be positive")
        keys = self.connection.execute(
            """
            SELECT action.run_id, action.article_id, action.modality,
                   action.configuration_id
            FROM reconciliation_actions AS action
            JOIN runs ON runs.run_id = action.run_id
            WHERE NOT runs.reconciliation_complete
              AND runs.status IN ('failed', 'interrupted')
            ORDER BY action.run_id, action.article_id, action.modality,
                     action.configuration_id
            LIMIT ?
            """,
            [page_size],
        ).fetchmany(page_size)
        if keys:
            self.connection.executemany(
                """
                DELETE FROM reconciliation_actions
                WHERE run_id = ? AND article_id = ? AND modality = ?
                  AND configuration_id = ?
                """,
                keys,
            )
        return len(keys)

    def interrupt_abandoned_runs(self) -> int:
        """Terminalize prior process-owned run records after lock reacquisition."""

        rows = self.connection.execute(
            """
            UPDATE runs
            SET status = 'interrupted', finished_at = current_timestamp
            WHERE status = 'running'
            RETURNING run_id
            """
        ).fetchall()
        return len(rows)

    def finish_run(self, run_id: str, *, reconciled: bool) -> None:
        """Mark one normally returned run terminal after all required work."""

        self.connection.execute(
            """
            UPDATE runs SET status = 'succeeded',
                reconciliation_complete = ?, finished_at = current_timestamp
            WHERE run_id = ?
            """,
            [reconciled, run_id],
        )

    def interrupt_run(self, run_id: str) -> None:
        """Record a caught interruption without asserting discovery/removal."""

        self.connection.execute(
            """
            UPDATE runs SET status = 'interrupted', finished_at = current_timestamp
            WHERE run_id = ? AND status = 'running'
            """,
            [run_id],
        )

    def fail_run(self, run_id: str) -> None:
        """Record an operational failure without claiming traversal completion."""

        self.connection.execute(
            """
            UPDATE runs SET status = 'failed', finished_at = current_timestamp
            WHERE run_id = ? AND status = 'running'
            """,
            [run_id],
        )

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
                INSERT INTO embedding_work_storage
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
            if modality == EmbeddingModality.HEADER_IMAGE.value:
                self.connection.execute(
                    """
                    UPDATE embedding_work_storage
                    SET source_relative_path = NULL, input_sha256 = NULL,
                        state = ?
                    WHERE article_id = ? AND modality = ?
                      AND configuration_id = ?
                    """,
                    [
                        WorkState.STALE_INPUT.value,
                        article_id,
                        EmbeddingModality.MULTIMODAL_ARTICLE.value,
                        configuration_identifier,
                    ],
                )
        self.connection.execute(
            """
            UPDATE embedding_work_storage
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
                INSERT INTO embedding_work_storage
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
                UPDATE embedding_work_storage
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
            DELETE FROM embedding_storage
            WHERE article_id = ? AND modality = ? AND configuration_id = ?
            """,
            [article_id, modality, configuration_identifier],
        )
        self.connection.execute(
            """
            UPDATE embedding_work_storage
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
                WorkState.ELIGIBLE.value,
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
                   e.derivation_kind, e.raw_response_sha256, e.response_model,
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
            derivation_kind,
            raw_response_sha256,
            response_model,
            stored_vector_sha256,
        ) = row
        self.connection.execute(
            """
            INSERT INTO embedding_generation_history
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, current_timestamp)
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
                derivation_kind,
                raw_response_sha256,
                response_model,
                stored_vector_sha256,
                run_id,
                reason.value,
            ],
        )

    def _archive_active_generation_from_storage(
        self,
        *,
        run_id: str,
        article_id: str,
        modality: str,
        configuration_identifier: str,
    ) -> None:
        """Archive the exact physical generation hidden by a visible action."""

        row = self.connection.execute(
            """
            SELECT work.generation_run_id, vector.source_relative_path,
                   vector.input_sha256, vector.derivation_kind,
                   vector.raw_response_sha256, vector.response_model,
                   vector.stored_vector_sha256
            FROM embedding_storage AS vector
            JOIN embedding_work_storage AS work
              USING (article_id, modality, configuration_id)
            WHERE vector.article_id = ? AND vector.modality = ?
              AND vector.configuration_id = ?
            """,
            [article_id, modality, configuration_identifier],
        ).fetchone()
        if row is None:
            return
        self.connection.execute(
            """
            INSERT INTO embedding_generation_history
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, current_timestamp)
            ON CONFLICT (
                article_id, modality, configuration_id, generation_run_id
            ) DO NOTHING
            """,
            [
                article_id,
                modality,
                configuration_identifier,
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
                row[5],
                row[6],
                run_id,
                SupersessionReason.INPUT_CHANGED.value,
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

        row = self.connection.execute(
            """
            UPDATE embedding_work_storage
            SET state = ?, attempt_count = attempt_count + 1,
                error_code = NULL, status_code = NULL,
                retry_after_seconds = NULL, last_run_id = ?,
                generation_run_id = NULL,
                updated_at = current_timestamp
            WHERE article_id = ? AND modality = ? AND configuration_id = ?
              AND state = ?
            RETURNING attempt_count
            """,
            [
                WorkState.IN_PROGRESS.value,
                run_id,
                article_id,
                modality,
                configuration_identifier,
                WorkState.QUEUED.value,
            ],
        ).fetchone()
        if row is None:
            raise ValueError("embedding work is not queued")
        self.increment_run(run_id, "attempted")

    def admit_work(
        self,
        *,
        run_id: str,
        article_id: str,
        modality: str,
        configuration_identifier: str,
    ) -> WorkState:
        """Durably queue one attemptable work item without starting an attempt."""

        row = self.connection.execute(
            """
            UPDATE embedding_work_storage
            SET state = ?, error_code = NULL, status_code = NULL,
                retry_after_seconds = NULL, last_run_id = ?,
                updated_at = current_timestamp
            WHERE article_id = ? AND modality = ? AND configuration_id = ?
              AND state IN (?, ?, ?)
            RETURNING state
            """,
            [
                WorkState.QUEUED.value,
                run_id,
                article_id,
                modality,
                configuration_identifier,
                WorkState.ELIGIBLE.value,
                WorkState.INTERRUPTED.value,
                WorkState.RETRYABLE.value,
            ],
        ).fetchone()
        if row is None:
            current = self.connection.execute(
                """
                SELECT state FROM embedding_work_items
                WHERE article_id = ? AND modality = ? AND configuration_id = ?
                """,
                [article_id, modality, configuration_identifier],
            ).fetchone()
            if current is None:
                raise ValueError("embedding work is not registered")
            return WorkState(str(current[0]))
        return WorkState.QUEUED

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
                       derivation_kind, raw_response_sha256, response_model,
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
                        derivation_kind, raw_response_sha256, response_model,
                        stored_vector_sha256, vector, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL,
                              NULL, NULL, ?, NULL, NULL, NULL, NULL, NULL,
                              NULL, current_timestamp)
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
                        WorkState.ELIGIBLE.value,
                        run_id,
                    ],
                )
                states.append(WorkState.ELIGIBLE)
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
                        derivation_kind = NULL, raw_response_sha256 = NULL,
                        response_model = NULL,
                        stored_vector_sha256 = NULL, vector = NULL,
                        updated_at = current_timestamp
                    WHERE article_id = ? AND configuration_id = ?
                      AND article_input_sha256 = ? AND part_index = ?
                    """,
                    [
                        WorkState.ELIGIBLE.value,
                        run_id,
                        article_id,
                        configuration_identifier,
                        article_input_sha256,
                        part.index,
                    ],
                )
                states.append(WorkState.ELIGIBLE)
                continue
            proven_success = (
                state is WorkState.SUCCEEDED
                and row[8] is not None
                and row[9] is not None
                and row[12] is not None
                and row[13] is not None
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
                        derivation_kind = NULL, raw_response_sha256 = NULL,
                        response_model = NULL,
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
    ) -> int:
        """Checkpoint one part before its hosted request."""

        row = self.connection.execute(
            """
            UPDATE long_text_parts
            SET state = ?, attempt_count = attempt_count + 1,
                error_code = NULL, status_code = NULL,
                retry_after_seconds = NULL, last_run_id = ?,
                generation_run_id = NULL, derivation_kind = NULL,
                raw_response_sha256 = NULL, response_model = NULL,
                stored_vector_sha256 = NULL,
                vector = NULL, updated_at = current_timestamp
            WHERE article_id = ? AND configuration_id = ?
              AND article_input_sha256 = ? AND part_index = ?
              AND state = ?
            RETURNING attempt_count
            """,
            [
                WorkState.IN_PROGRESS.value,
                run_id,
                article_id,
                configuration_identifier,
                article_input_sha256,
                part_index,
                WorkState.QUEUED.value,
            ],
        ).fetchone()
        if row is None:
            raise ValueError("long-text part is not registered")
        return int(row[0])

    def admit_long_text_part(
        self,
        *,
        run_id: str,
        article_id: str,
        configuration_identifier: str,
        article_input_sha256: str,
        part_index: int,
    ) -> WorkState:
        """Durably queue one attemptable long-text part."""

        row = self.connection.execute(
            """
            UPDATE long_text_parts
            SET state = ?, error_code = NULL, status_code = NULL,
                retry_after_seconds = NULL, last_run_id = ?,
                updated_at = current_timestamp
            WHERE article_id = ? AND configuration_id = ?
              AND article_input_sha256 = ? AND part_index = ?
              AND state IN (?, ?, ?)
            RETURNING state
            """,
            [
                WorkState.QUEUED.value,
                run_id,
                article_id,
                configuration_identifier,
                article_input_sha256,
                part_index,
                WorkState.ELIGIBLE.value,
                WorkState.INTERRUPTED.value,
                WorkState.RETRYABLE.value,
            ],
        ).fetchone()
        if row is None:
            current = self.connection.execute(
                """
                SELECT state FROM long_text_parts
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
            if current is None:
                raise ValueError("long-text part is not registered")
            return WorkState(str(current[0]))
        return WorkState.QUEUED

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
                generation_run_id = NULL, derivation_kind = NULL,
                raw_response_sha256 = NULL, response_model = NULL,
                stored_vector_sha256 = NULL,
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

    def record_terminal_long_text_part_failure(
        self,
        *,
        run_id: str,
        article_id: str,
        configuration_identifier: str,
        article_input_sha256: str,
        part_index: int,
        error_code: str,
        status_code: int | None,
        retry_after_seconds: float | None,
    ) -> None:
        """Atomically terminalize one part, its parent work, and run count."""

        parent = self.connection.execute(
            """
            SELECT state
            FROM embedding_work_items
            WHERE article_id = ? AND modality = ? AND configuration_id = ?
            """,
            [
                article_id,
                EmbeddingModality.ARTICLE_TEXT.value,
                configuration_identifier,
            ],
        ).fetchone()
        if parent is None:
            raise ValueError("article-text work is not registered")
        self.record_long_text_part_failure(
            run_id=run_id,
            article_id=article_id,
            configuration_identifier=configuration_identifier,
            article_input_sha256=article_input_sha256,
            part_index=part_index,
            state=WorkState.TERMINAL,
            error_code=error_code,
            status_code=status_code,
            retry_after_seconds=retry_after_seconds,
        )
        if WorkState(str(parent[0])) is WorkState.TERMINAL:
            return
        self.connection.execute(
            """
            UPDATE embedding_work_storage
            SET state = ?, error_code = ?, status_code = ?,
                retry_after_seconds = ?, last_run_id = ?,
                generation_run_id = NULL, updated_at = current_timestamp
            WHERE article_id = ? AND modality = ? AND configuration_id = ?
            """,
            [
                WorkState.TERMINAL.value,
                error_code,
                status_code,
                retry_after_seconds,
                run_id,
                article_id,
                EmbeddingModality.ARTICLE_TEXT.value,
                configuration_identifier,
            ],
        )
        self.increment_run(run_id, "terminal")

    def terminalize_exhausted_long_text_work(
        self,
        *,
        run_id: str,
        article_id: str,
        configuration_identifier: str,
        article_input_sha256: str,
        attempt_limit: int,
    ) -> bool:
        """Repair durable terminal/exhausted parts before another hosted call."""

        row = self.connection.execute(
            """
            SELECT part_index, state, error_code, status_code,
                   retry_after_seconds
            FROM long_text_parts
            WHERE article_id = ? AND configuration_id = ?
              AND article_input_sha256 = ?
              AND (
                  state = ?
                  OR (state != ? AND attempt_count >= ?)
              )
            ORDER BY part_index
            LIMIT 1
            """,
            [
                article_id,
                configuration_identifier,
                article_input_sha256,
                WorkState.TERMINAL.value,
                WorkState.SUCCEEDED.value,
                attempt_limit,
            ],
        ).fetchone()
        if row is None:
            return False
        part_index, state, error_code, status_code, retry_after_seconds = row
        if WorkState(str(state)) is not WorkState.TERMINAL:
            error_code = "attempt_limit_exhausted"
            status_code = None
            retry_after_seconds = None
        self.record_terminal_long_text_part_failure(
            run_id=run_id,
            article_id=article_id,
            configuration_identifier=configuration_identifier,
            article_input_sha256=article_input_sha256,
            part_index=int(part_index),
            error_code=str(error_code),
            status_code=status_code,
            retry_after_seconds=retry_after_seconds,
        )
        return True

    def publish_long_text_part_success(
        self,
        *,
        run_id: str,
        article_id: str,
        configuration_identifier: str,
        article_input_sha256: str,
        part_index: int,
        derivation_kind: str,
        raw_response_sha256: str | None,
        response_model: str | None,
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
                   ?, ?, ?, ?, ?, current_timestamp
            FROM long_text_parts
            WHERE article_id = ? AND configuration_id = ?
              AND article_input_sha256 = ? AND part_index = ?
            """,
            [
                run_id,
                derivation_kind,
                raw_response_sha256,
                response_model,
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
                generation_run_id = ?, derivation_kind = ?,
                raw_response_sha256 = ?, response_model = ?,
                stored_vector_sha256 = ?, vector = ?,
                updated_at = current_timestamp
            WHERE article_id = ? AND configuration_id = ?
              AND article_input_sha256 = ? AND part_index = ?
            """,
            [
                WorkState.SUCCEEDED.value,
                run_id,
                run_id,
                derivation_kind,
                raw_response_sha256,
                response_model,
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
            SELECT g.generation_run_id, g.stored_vector_sha256, g.vector,
                   g.derivation_kind, g.raw_response_sha256, g.response_model
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
        if row is None or any(row[index] is None for index in (0, 1, 2, 3)):
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
        count_run: bool = True,
    ) -> None:
        """Commit one content-free retryable or terminal checkpoint."""

        if not state.is_failure:
            raise ValueError("failure checkpoint requires a failure state")
        self.connection.execute(
            """
            UPDATE embedding_work_storage
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
        if count_run:
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
        derivation_kind: str,
        raw_response_sha256: str | None,
        response_model: str | None,
        stored_vector_sha256: str,
        vector: Sequence[float],
    ) -> None:
        """Publish one vector and its success checkpoint atomically."""

        self.connection.execute(
            """
            INSERT INTO embedding_storage
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (article_id, modality, configuration_id)
            DO UPDATE SET
                source_relative_path = excluded.source_relative_path,
                published_at_utc = excluded.published_at_utc,
                publication_date_new_york = excluded.publication_date_new_york,
                dimensions = excluded.dimensions,
                input_sha256 = excluded.input_sha256,
                derivation_kind = excluded.derivation_kind,
                raw_response_sha256 = excluded.raw_response_sha256,
                response_model = excluded.response_model,
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
                derivation_kind,
                raw_response_sha256,
                response_model,
                stored_vector_sha256,
                list(vector),
            ],
        )
        self.connection.execute(
            """
            UPDATE embedding_work_storage
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
            derivation_kind=VectorDerivationKind.LONG_TEXT_AGGREGATE.value,
            raw_response_sha256=None,
            response_model=None,
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

    def publish_image_success(
        self,
        *,
        run_id: str,
        article_id: str,
        configuration_identifier: str,
        source_relative_path: str,
        published_at_utc: object,
        publication_date_new_york: object,
        dimensions: int,
        source_sha256: str,
        source_format: str,
        source_bytes: int,
        source_width: int,
        source_height: int,
        source_frames: int,
        embedded_input_sha256: str,
        embedded_format: str,
        embedded_bytes: int,
        embedded_width: int,
        embedded_height: int,
        embedded_frames: int,
        transform_id: str,
        rendition_relative_path: str | None,
        derivation_kind: str,
        raw_response_sha256: str | None,
        response_model: str | None,
        stored_vector_sha256: str,
        vector: Sequence[float],
    ) -> None:
        """Atomically publish an image vector and its exact input provenance."""

        self.publish_success(
            run_id=run_id,
            article_id=article_id,
            modality=EmbeddingModality.HEADER_IMAGE.value,
            configuration_identifier=configuration_identifier,
            source_relative_path=source_relative_path,
            published_at_utc=published_at_utc,
            publication_date_new_york=publication_date_new_york,
            dimensions=dimensions,
            input_sha256=source_sha256,
            derivation_kind=derivation_kind,
            raw_response_sha256=raw_response_sha256,
            response_model=response_model,
            stored_vector_sha256=stored_vector_sha256,
            vector=vector,
        )
        self.connection.execute(
            """
            INSERT INTO image_input_provenance
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    current_timestamp)
            """,
            [
                article_id,
                configuration_identifier,
                run_id,
                source_sha256,
                source_format,
                source_bytes,
                source_width,
                source_height,
                source_frames,
                embedded_input_sha256,
                embedded_format,
                embedded_bytes,
                embedded_width,
                embedded_height,
                embedded_frames,
                transform_id,
                rendition_relative_path,
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
            derivation_kind=VectorDerivationKind.MULTIMODAL_MIDPOINT.value,
            raw_response_sha256=None,
            response_model=None,
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

    def finish_run_metrics(
        self,
        run_id: str,
        *,
        hosted_requests: int,
        hosted_retries: int,
        usage: dict[str, int | float],
        throttles: int,
        elapsed_seconds: float,
    ) -> None:
        """Persist safe aggregate hosted observations for one returned summary."""

        counters = (hosted_requests, hosted_retries, throttles)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counters
        ):
            raise ValueError("hosted run counters must be nonnegative integers")
        if (
            isinstance(elapsed_seconds, bool)
            or not isinstance(elapsed_seconds, (int, float))
            or not math.isfinite(float(elapsed_seconds))
            or elapsed_seconds < 0
        ):
            raise ValueError("hosted run elapsed time must be finite and nonnegative")
        usage_json = json.dumps(
            normalize_safe_usage(usage), sort_keys=True, separators=(",", ":")
        )
        self.connection.execute(
            """
            UPDATE runs
            SET hosted_requests = ?, hosted_retries = ?, usage_json = ?,
                throttles = ?, elapsed_seconds = ?
            WHERE run_id = ?
            """,
            [
                hosted_requests,
                hosted_retries,
                usage_json,
                throttles,
                elapsed_seconds,
                run_id,
            ],
        )

    def rendition_is_referenced(self, relative_path: str) -> bool:
        """Check one candidate against current and historical provenance."""

        return (
            self.connection.execute(
                """
                SELECT 1 FROM image_input_provenance
                WHERE rendition_relative_path = ? LIMIT 1
                """,
                [relative_path],
            ).fetchone()
            is not None
        )

    def run_result(self, run_id: str) -> EmbeddingRunResult:
        """Read one completed or interrupted run's stable counters."""

        row = self.connection.execute(
            """
            SELECT articles, embeddings, reused, attempted, succeeded,
                   retryable, terminal, interrupted, header_absent, header_failed,
                   hosted_requests, hosted_retries, usage_json, throttles,
                   elapsed_seconds
            FROM runs
            WHERE run_id = ?
            """,
            [run_id],
        ).fetchone()
        assert row is not None
        return EmbeddingRunResult(
            *(int(value) for value in row[:12]),
            usage=json.loads(row[12]),
            throttles=int(row[13]),
            elapsed_seconds=float(row[14]),
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
