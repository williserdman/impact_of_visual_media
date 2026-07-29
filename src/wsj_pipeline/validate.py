"""Cross-artifact validation for canonical Parquet and DuckDB indexes."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from wsj_pipeline.config import PipelineConfig
from wsj_pipeline.extract import derive_publication_dates
from wsj_pipeline.publish import ARTICLE_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One stable validation failure without licensed article content."""

    code: str
    message: str


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Complete validation result across published artifacts."""

    issues: tuple[ValidationIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues


def validate_outputs(config: PipelineConfig) -> ValidationReport:
    """Return all detected invariant violations in deterministic order."""

    issues: list[ValidationIssue] = []
    article_rows: list[dict[str, object]] = []
    article_files = sorted((config.parquet_root / "articles").rglob("articles.parquet"))
    if not article_files:
        issues.append(
            ValidationIssue(
                "missing_article_partitions",
                "no canonical article partitions were found",
            )
        )
        return ValidationReport(tuple(issues))

    for path in article_files:
        table = pq.read_table(path, partitioning=None)
        rows = table.to_pylist()
        article_rows.extend(rows)
        directory_year, directory_month = _partition_from_path(path)
        if any(
            row["publication_year_ny"] != directory_year
            or row["publication_month_ny"] != directory_month
            or row["publication_date_new_york"].year != directory_year
            or row["publication_date_new_york"].month != directory_month
            for row in rows
        ):
            issues.append(
                ValidationIssue(
                    "partition_date_mismatch",
                    f"{path.name} contains rows outside its year/month partition",
                )
            )
        if any(row["schema_version"] != ARTICLE_SCHEMA_VERSION for row in rows):
            issues.append(
                ValidationIssue(
                    "schema_version_mismatch",
                    f"{path.name} contains an unsupported article schema",
                )
            )

    keys = [str(row["article_key"]) for row in article_rows]
    duplicate_keys = [key for key, count in Counter(keys).items() if count > 1]
    if duplicate_keys:
        issues.append(
            ValidationIssue(
                "duplicate_article_key",
                f"canonical Parquet contains {len(duplicate_keys)} duplicate keys",
            )
        )
    if any(not key for key in keys):
        issues.append(
            ValidationIssue(
                "null_article_key",
                "canonical Parquet contains a null or empty article key",
            )
        )

    if any(not _date_fields_match(row) for row in article_rows):
        issues.append(
            ValidationIssue(
                "date_derivation_mismatch",
                "stored UTC/New York date fields are inconsistent",
            )
        )

    _validate_duplicate_references(config, article_rows, issues)
    _validate_index_keys(config, keys, issues)
    return ValidationReport(
        tuple(sorted(issues, key=lambda issue: (issue.code, issue.message)))
    )


def _partition_from_path(path: Path) -> tuple[int, int]:
    year_part = path.parent.parent.name
    month_part = path.parent.name
    try:
        year = int(year_part.removeprefix("publication_year_ny="))
        month = int(month_part.removeprefix("publication_month_ny="))
    except ValueError:
        return (-1, -1)
    return year, month


def _date_fields_match(row: dict[str, object]) -> bool:
    derived = derive_publication_dates(row["published_at_utc"])
    return (
        derived.publication_date_utc == row["publication_date_utc"]
        and derived.publication_date_new_york == row["publication_date_new_york"]
        and derived.publication_year_ny == row["publication_year_ny"]
        and derived.publication_month_ny == row["publication_month_ny"]
    )


def _validate_duplicate_references(
    config: PipelineConfig,
    article_rows: list[dict[str, object]],
    issues: list[ValidationIssue],
) -> None:
    path = config.parquet_root / "audit" / "duplicates.parquet"
    duplicate_rows = pq.read_table(path).to_pylist()
    canonical_paths = {row["source_path"] for row in article_rows}
    invalid = [
        row
        for row in duplicate_rows
        if row["winner_source_path"] not in canonical_paths
    ]
    if invalid:
        issues.append(
            ValidationIssue(
                "invalid_duplicate_winner",
                f"duplicate audit contains {len(invalid)} invalid winner references",
            )
        )


def _validate_index_keys(
    config: PipelineConfig,
    parquet_keys: list[str],
    issues: list[ValidationIssue],
) -> None:
    with duckdb.connect(str(config.index_db), read_only=True) as connection:
        index_keys = [
            row[0]
            for row in connection.execute(
                "SELECT article_key FROM publication_index"
            ).fetchall()
        ]
    if set(parquet_keys) != set(index_keys) or len(parquet_keys) != len(index_keys):
        issues.append(
            ValidationIssue(
                "index_key_mismatch",
                "canonical Parquet and publication_index keys differ",
            )
        )
