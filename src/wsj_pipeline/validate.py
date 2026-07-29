"""Cross-artifact validation for canonical Parquet and DuckDB indexes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from wsj_pipeline.config import PipelineConfig
from wsj_pipeline.extract import derive_publication_dates
from wsj_pipeline.publish import ARTICLE_SCHEMA_VERSION

ARTICLE_VALIDATION_COLUMNS = (
    "schema_version",
    "article_key",
    "source_path",
    "published_at_utc",
    "publication_date_utc",
    "published_at_new_york",
    "publication_date_new_york",
    "publication_year_ny",
    "publication_month_ny",
)


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
        directory_year, directory_month = _partition_from_path(path)
        partition_mismatch = False
        schema_mismatch = False
        date_mismatch = False
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(
            columns=list(ARTICLE_VALIDATION_COLUMNS),
            batch_size=65_536,
        ):
            rows = batch.to_pylist()
            partition_mismatch = partition_mismatch or any(
                not _row_matches_partition(
                    row,
                    directory_year,
                    directory_month,
                )
                for row in rows
            )
            schema_mismatch = schema_mismatch or any(
                row["schema_version"] != ARTICLE_SCHEMA_VERSION for row in rows
            )
            date_mismatch = date_mismatch or any(
                not _date_fields_match(row) for row in rows
            )
        if partition_mismatch:
            issues.append(
                ValidationIssue(
                    "partition_date_mismatch",
                    f"{path.name} contains rows outside its year/month partition",
                )
            )
        if schema_mismatch:
            issues.append(
                ValidationIssue(
                    "schema_version_mismatch",
                    f"{path.name} contains an unsupported article schema",
                )
            )
        if date_mismatch:
            issues.append(
                ValidationIssue(
                    "date_derivation_mismatch",
                    f"{path.name} contains inconsistent UTC/New York date fields",
                )
            )

    _validate_global_relations(config, issues)
    _validate_state_queues(config, issues)
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
    try:
        derived = derive_publication_dates(row["published_at_utc"])
    except (AttributeError, TypeError, ValueError):
        return False
    return (
        derived.publication_date_utc == row["publication_date_utc"]
        and derived.published_at_new_york == row["published_at_new_york"]
        and derived.publication_date_new_york == row["publication_date_new_york"]
        and derived.publication_year_ny == row["publication_year_ny"]
        and derived.publication_month_ny == row["publication_month_ny"]
    )


def _row_matches_partition(
    row: dict[str, object],
    directory_year: int,
    directory_month: int,
) -> bool:
    publication_date = row["publication_date_new_york"]
    return (
        isinstance(publication_date, date)
        and row["publication_year_ny"] == directory_year
        and row["publication_month_ny"] == directory_month
        and publication_date.year == directory_year
        and publication_date.month == directory_month
    )


def _validate_global_relations(
    config: PipelineConfig,
    issues: list[ValidationIssue],
) -> None:
    article_glob = (
        config.parquet_root
        / "articles"
        / "publication_year_ny=*"
        / "publication_month_ny=*"
        / "articles.parquet"
    )
    duplicate_path = config.parquet_root / "audit" / "duplicates.parquet"
    columns = (
        "article_key",
        "source_path",
        "published_at_utc",
        "publication_date_utc",
        "published_at_new_york",
        "publication_date_new_york",
    )
    with duckdb.connect(str(config.index_db), read_only=True) as connection:
        connection.execute(
            f"""
            CREATE TEMP VIEW validation_articles AS
            SELECT {", ".join(ARTICLE_VALIDATION_COLUMNS)}
            FROM read_parquet(
                '{_sql_path(article_glob)}',
                hive_partitioning = false,
                union_by_name = true
            )
            """
        )
        null_keys, duplicate_keys = connection.execute(
            """
            SELECT
                count(*) FILTER (
                    WHERE article_key IS NULL OR article_key = ''
                ),
                (
                    SELECT count(*)
                    FROM (
                        SELECT article_key
                        FROM validation_articles
                        WHERE article_key IS NOT NULL
                          AND article_key != ''
                        GROUP BY article_key
                        HAVING count(*) > 1
                    )
                )
            FROM validation_articles
            """
        ).fetchone()
        invalid_winners = connection.execute(
            f"""
            SELECT count(*)
            FROM read_parquet(
                '{_sql_path(duplicate_path)}'
            ) AS duplicate
            LEFT JOIN validation_articles AS article
              ON article.article_key = duplicate.article_key
             AND article.source_path = duplicate.winner_source_path
            WHERE article.article_key IS NULL
            """
        ).fetchone()[0]
        key_differences = _difference_count(
            connection,
            ("article_key",),
        )
        value_differences = (
            _difference_count(connection, columns)
            if not key_differences
            else 0
        )

    if duplicate_keys:
        issues.append(
            ValidationIssue(
                "duplicate_article_key",
                f"canonical Parquet contains {duplicate_keys} duplicate keys",
            )
        )
    if null_keys:
        issues.append(
            ValidationIssue(
                "null_article_key",
                "canonical Parquet contains a null or empty article key",
            )
        )
    if invalid_winners:
        issues.append(
            ValidationIssue(
                "invalid_duplicate_winner",
                (
                    "duplicate audit contains "
                    f"{invalid_winners} invalid winner references"
                ),
            )
        )
    if key_differences:
        issues.append(
            ValidationIssue(
                "index_key_mismatch",
                "canonical Parquet and publication_index keys differ",
            )
        )
    elif value_differences:
        issues.append(
            ValidationIssue(
                "index_value_mismatch",
                "canonical Parquet and publication_index values differ",
            )
        )


def _difference_count(
    connection: duckdb.DuckDBPyConnection,
    columns: tuple[str, ...],
) -> int:
    selection = ", ".join(columns)
    return connection.execute(
        f"""
        SELECT count(*)
        FROM (
            (
                SELECT {selection}
                FROM validation_articles
                EXCEPT ALL
                SELECT {selection}
                FROM publication_index
            )
            UNION ALL
            (
                SELECT {selection}
                FROM publication_index
                EXCEPT ALL
                SELECT {selection}
                FROM validation_articles
            )
        )
        """
    ).fetchone()[0]


def _validate_state_queues(
    config: PipelineConfig,
    issues: list[ValidationIssue],
) -> None:
    with duckdb.connect(str(config.state_db), read_only=True) as connection:
        pending, dirty = connection.execute(
            """
            SELECT
                (SELECT count(*) FROM pending_article_keys),
                (SELECT count(*) FROM dirty_partitions)
            """
        ).fetchone()
    if pending or dirty:
        issues.append(
            ValidationIssue(
                "unpublished_state_changes",
                (
                    f"operational state has {pending} pending keys "
                    f"and {dirty} dirty partitions"
                ),
            )
        )


def _sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")
