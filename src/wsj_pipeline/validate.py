"""Content-free validation of the DuckDB catalog and generated files."""

from __future__ import annotations

import errno
import hashlib
import os
import stat
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlsplit

import duckdb

from wsj_pipeline.catalog import CATALOG_SCHEMA_VERSION, Catalog, CatalogError
from wsj_pipeline.config import PipelineConfig
from wsj_pipeline.extract import EXTRACTOR_VERSION, derive_publication_date_new_york

_READ_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY_FLAGS = _READ_FLAGS | os.O_DIRECTORY
_HASH_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One stable validation failure without licensed article content."""

    code: str
    message: str


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Complete validation result across the catalog and generated tree."""

    issues: tuple[ValidationIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues


@dataclass(frozen=True, slots=True)
class _CatalogFile:
    article_id: str
    cleaned_markdown_path: str
    header_image_path: str | None
    inline_image_urls: tuple[str, ...]
    published_at_utc: datetime
    publication_date_new_york: date
    cleaned_markdown_sha256: str


def validate_outputs(config: PipelineConfig) -> ValidationReport:
    """Return all catalog and filesystem violations in deterministic order."""

    catalog_status = _catalog_file_status(config.catalog_db)
    if catalog_status != "ok":
        code = (
            "missing_catalog" if catalog_status == "missing" else "unsafe_catalog_path"
        )
        return ValidationReport(
            (ValidationIssue(code, f"catalog path is {catalog_status}"),)
        )

    issues: list[ValidationIssue] = []
    try:
        with duckdb.connect(str(config.catalog_db), read_only=True) as connection:
            try:
                Catalog.validate_schema(connection)
            except CatalogError as error:
                messages = {
                    "malformed_catalog_schema": (
                        "catalog schema does not match the supported contract"
                    ),
                    "unsupported_catalog_version": (
                        "catalog schema version is not supported"
                    ),
                }
                return ValidationReport(
                    (
                        ValidationIssue(
                            error.code,
                            messages.get(
                                error.code,
                                "catalog could not be queried with the "
                                "supported schema",
                            ),
                        ),
                    )
                )
            _validate_relations(connection, issues)
            for item in _iter_catalog_files(connection):
                _validate_catalog_file(config, item, issues)
            _validate_orphans(config, connection, issues)
    except duckdb.Error:
        issues.append(
            ValidationIssue(
                "invalid_catalog",
                "catalog could not be queried with the supported schema",
            )
        )

    return ValidationReport(
        tuple(sorted(issues, key=lambda issue: (issue.code, issue.message)))
    )


def _catalog_file_status(path: Path) -> str:
    try:
        path_stat = path.lstat()
    except FileNotFoundError:
        return "missing"
    if not stat.S_ISREG(path_stat.st_mode) or path_stat.st_nlink != 1:
        return "unsafe"
    return "ok"


def _validate_relations(
    connection: duckdb.DuckDBPyConnection,
    issues: list[ValidationIssue],
) -> None:
    schema_version = connection.execute(
        "SELECT value FROM metadata WHERE key = 'schema_version'"
    ).fetchone()
    if schema_version is None or schema_version[0] != CATALOG_SCHEMA_VERSION:
        issues.append(
            ValidationIssue(
                "unsupported_catalog_version",
                "catalog schema version is not supported",
            )
        )

    unsupported_extractors = _count(
        connection,
        """
        SELECT count(*)
        FROM (
            SELECT source_html_path, extractor_version FROM source_manifest
            UNION ALL
            SELECT source_html_path, extractor_version FROM candidates
            UNION ALL
            SELECT source_html_path, extractor_version FROM failures
        ) AS versions
        WHERE extractor_version <> ?
        """,
        [EXTRACTOR_VERSION],
    )
    _append_count_issue(
        issues,
        "unsupported_extractor_version",
        unsupported_extractors,
        "catalog rows use an unsupported extractor version",
    )

    duplicate_checks = (
        ("article_id", "duplicate_article_id"),
        ("source_html_path", "duplicate_source_html_path"),
        ("cleaned_markdown_path", "duplicate_markdown_path"),
    )
    for column, code in duplicate_checks:
        duplicate_groups = _count(
            connection,
            f"""
            SELECT count(*)
            FROM (
                SELECT {column}
                FROM articles
                GROUP BY {column}
                HAVING count(*) > 1
            ) AS duplicates
            """,
        )
        _append_count_issue(
            issues,
            code,
            duplicate_groups,
            f"catalog contains duplicate {column} groups",
        )

    running_runs = _count(
        connection,
        "SELECT count(*) FROM runs WHERE status = 'running'",
    )
    _append_count_issue(
        issues,
        "running_run",
        running_runs,
        "catalog contains unfinished running runs",
    )

    inconsistent_manifest_winners = _count(
        connection,
        """
        SELECT count(*)
        FROM articles AS article
        LEFT JOIN source_manifest AS manifest
          ON manifest.source_html_path = article.source_html_path
        WHERE manifest.source_html_path IS NULL
           OR manifest.article_id IS DISTINCT FROM article.article_id
        """,
    )
    _append_count_issue(
        issues,
        "inconsistent_manifest_winner",
        inconsistent_manifest_winners,
        "article winners disagree with source manifests",
    )

    failed_or_missing_winners = _count(
        connection,
        """
        SELECT count(*)
        FROM articles AS article
        JOIN source_manifest AS manifest
          ON manifest.source_html_path = article.source_html_path
        WHERE manifest.status IN ('failed', 'missing')
        """,
    )
    _append_count_issue(
        issues,
        "failed_or_missing_winner",
        failed_or_missing_winners,
        "articles reference failed or missing winners",
    )

    inconsistent_article_candidates = _count(
        connection,
        """
        SELECT count(*)
        FROM articles AS article
        LEFT JOIN candidates AS candidate
          ON candidate.source_html_path = article.source_html_path
         AND candidate.article_id = article.article_id
        WHERE candidate.source_html_path IS NULL
           OR candidate.header_image_path
                IS DISTINCT FROM article.header_image_path
           OR candidate.inline_image_urls
                IS DISTINCT FROM article.inline_image_urls
           OR candidate.published_at_utc
                IS DISTINCT FROM article.published_at_utc
           OR candidate.publication_date_new_york
                IS DISTINCT FROM article.publication_date_new_york
        """,
    )
    _append_count_issue(
        issues,
        "inconsistent_article_candidate",
        inconsistent_article_candidates,
        "article winners disagree with their candidates",
    )

    inconsistent_candidate_manifests = _count(
        connection,
        """
        SELECT count(*)
        FROM candidates AS candidate
        LEFT JOIN source_manifest AS manifest
          ON manifest.source_html_path = candidate.source_html_path
        WHERE manifest.source_html_path IS NULL
           OR manifest.article_id IS DISTINCT FROM candidate.article_id
           OR manifest.source_html_sha256
                IS DISTINCT FROM candidate.source_html_sha256
           OR manifest.extractor_version
                IS DISTINCT FROM candidate.extractor_version
           OR manifest.header_image_path
                IS DISTINCT FROM candidate.header_image_path
           OR manifest.status <> 'succeeded'
        """,
    )
    inconsistent_candidate_manifests += _count(
        connection,
        """
        SELECT count(*)
        FROM source_manifest AS manifest
        LEFT JOIN candidates AS candidate
          ON candidate.source_html_path = manifest.source_html_path
         AND candidate.article_id IS NOT DISTINCT FROM manifest.article_id
         AND candidate.source_html_sha256
                IS NOT DISTINCT FROM manifest.source_html_sha256
         AND candidate.extractor_version
                IS NOT DISTINCT FROM manifest.extractor_version
         AND candidate.header_image_path
                IS NOT DISTINCT FROM manifest.header_image_path
        WHERE manifest.status = 'succeeded'
          AND candidate.source_html_path IS NULL
        """,
    )
    _append_count_issue(
        issues,
        "inconsistent_candidate_manifest",
        inconsistent_candidate_manifests,
        "candidates disagree with source manifests",
    )

    broken_duplicate_winners = _count(
        connection,
        """
        SELECT count(*)
        FROM duplicates AS duplicate
        LEFT JOIN articles AS article
          ON article.article_id = duplicate.article_id
         AND article.source_html_path = duplicate.winner_source_html_path
        LEFT JOIN candidates AS winner
          ON winner.article_id = duplicate.article_id
         AND winner.source_html_path = duplicate.winner_source_html_path
        WHERE article.article_id IS NULL OR winner.source_html_path IS NULL
        """,
    )
    _append_count_issue(
        issues,
        "broken_duplicate_winner",
        broken_duplicate_winners,
        "duplicate rows contain broken winner references",
    )

    inconsistent_duplicate_candidates = _count(
        connection,
        """
        SELECT count(*)
        FROM duplicates AS duplicate
        LEFT JOIN candidates AS candidate
          ON candidate.article_id = duplicate.article_id
         AND candidate.source_html_path = duplicate.source_html_path
        WHERE candidate.source_html_path IS NULL
           OR duplicate.source_html_path = duplicate.winner_source_html_path
           OR duplicate.duplicate_rank < 2
        """,
    )
    _append_count_issue(
        issues,
        "inconsistent_duplicate_candidate",
        inconsistent_duplicate_candidates,
        "duplicate rows disagree with candidate rankings",
    )

    incomplete_duplicate_sets = _count(
        connection,
        """
        SELECT count(*)
        FROM candidates AS candidate
        LEFT JOIN articles AS article
          ON article.article_id = candidate.article_id
        LEFT JOIN duplicates AS duplicate
          ON duplicate.article_id = candidate.article_id
         AND duplicate.source_html_path = candidate.source_html_path
        WHERE article.article_id IS NULL
           OR (
                candidate.source_html_path = article.source_html_path
                AND duplicate.source_html_path IS NOT NULL
           )
           OR (
                candidate.source_html_path <> article.source_html_path
                AND duplicate.source_html_path IS NULL
           )
        """,
    )
    _append_count_issue(
        issues,
        "inconsistent_candidate_set",
        incomplete_duplicate_sets,
        "candidate winner and duplicate membership is inconsistent",
    )

    inconsistent_duplicate_audit = _count(
        connection,
        """
        WITH ranked_candidates AS (
            SELECT candidate.*,
                   row_number() OVER (
                       PARTITION BY article_id
                       ORDER BY
                           CASE
                               WHEN editorial_character_count > 0 THEN 0
                               ELSE 1
                           END,
                           editorial_character_count DESC,
                           editorial_block_count DESC,
                           metadata_completeness DESC,
                           updated_at_utc DESC NULLS LAST,
                           source_html_path
                   ) AS expected_rank
            FROM candidates AS candidate
        ),
        expected_duplicates AS (
            SELECT duplicate.article_id,
                   duplicate.source_html_path,
                   winner.source_html_path AS winner_source_html_path,
                   duplicate.expected_rank AS duplicate_rank,
                   CASE
                       WHEN (winner.editorial_character_count > 0)
                            <> (duplicate.editorial_character_count > 0)
                           THEN 'empty_content'
                       WHEN winner.editorial_character_count
                            <> duplicate.editorial_character_count
                           THEN 'lower_editorial_character_count'
                       WHEN winner.editorial_block_count
                            <> duplicate.editorial_block_count
                           THEN 'lower_editorial_block_count'
                       WHEN winner.metadata_completeness
                            <> duplicate.metadata_completeness
                           THEN 'lower_metadata_completeness'
                       WHEN winner.updated_at_utc
                            IS DISTINCT FROM duplicate.updated_at_utc
                           THEN 'older_update'
                       ELSE 'lexical_tiebreak'
                   END AS reason
            FROM ranked_candidates AS duplicate
            JOIN ranked_candidates AS winner
              ON winner.article_id = duplicate.article_id
             AND winner.expected_rank = 1
            WHERE duplicate.expected_rank > 1
        ),
        audit_differences AS (
            (
                SELECT article_id, source_html_path,
                       winner_source_html_path, duplicate_rank, reason
                FROM expected_duplicates
                EXCEPT ALL
                SELECT article_id, source_html_path,
                       winner_source_html_path, duplicate_rank, reason
                FROM duplicates
            )
            UNION ALL
            (
                SELECT article_id, source_html_path,
                       winner_source_html_path, duplicate_rank, reason
                FROM duplicates
                EXCEPT ALL
                SELECT article_id, source_html_path,
                       winner_source_html_path, duplicate_rank, reason
                FROM expected_duplicates
            )
        )
        SELECT
            (
                SELECT count(*)
                FROM ranked_candidates AS winner
                LEFT JOIN articles AS article
                  ON article.article_id = winner.article_id
                WHERE winner.expected_rank = 1
                  AND article.source_html_path
                        IS DISTINCT FROM winner.source_html_path
            )
            + (SELECT count(*) FROM audit_differences)
        """,
    )
    _append_count_issue(
        issues,
        "inconsistent_duplicate_audit",
        inconsistent_duplicate_audit,
        "candidate rankings and duplicate audit rows are inconsistent",
    )


def _count(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    parameters: list[str] | None = None,
) -> int:
    return int(connection.execute(query, parameters or []).fetchone()[0])


def _append_count_issue(
    issues: list[ValidationIssue],
    code: str,
    count: int,
    message: str,
) -> None:
    if count:
        issues.append(ValidationIssue(code, f"{message}: {count}"))


def _iter_catalog_files(
    connection: duckdb.DuckDBPyConnection,
) -> Iterator[_CatalogFile]:
    """Yield content-free file metadata without materializing all article rows."""

    result = connection.execute(
        """
        SELECT article_id, cleaned_markdown_path, header_image_path,
               inline_image_urls, published_at_utc,
               publication_date_new_york, cleaned_markdown_sha256
        FROM articles
        ORDER BY article_id
        """
    )
    while row := result.fetchone():
        yield _CatalogFile(
            article_id=row[0],
            cleaned_markdown_path=row[1],
            header_image_path=row[2],
            inline_image_urls=tuple(row[3]),
            published_at_utc=row[4],
            publication_date_new_york=row[5],
            cleaned_markdown_sha256=row[6],
        )


def _validate_catalog_file(
    config: PipelineConfig,
    item: _CatalogFile,
    issues: list[ValidationIssue],
) -> None:
    try:
        derived_date = derive_publication_date_new_york(item.published_at_utc)
    except (AttributeError, TypeError, ValueError):
        derived_date = None
    if derived_date != item.publication_date_new_york:
        issues.append(
            ValidationIssue(
                "publication_date_mismatch",
                f"publication date is inconsistent for {item.article_id}",
            )
        )

    expected_path = _expected_markdown_path(
        item.article_id,
        item.publication_date_new_york,
    )
    if item.cleaned_markdown_path != expected_path:
        issues.append(
            ValidationIssue(
                "markdown_path_mismatch",
                f"Markdown path {item.cleaned_markdown_path} should be {expected_path}",
            )
        )

    markdown_status, descriptor = _open_relative_regular(
        config.output_root,
        item.cleaned_markdown_path,
    )
    if markdown_status == "missing":
        issues.append(
            ValidationIssue(
                "missing_markdown",
                f"Markdown file is missing: {item.cleaned_markdown_path}",
            )
        )
    elif markdown_status == "unsafe":
        issues.append(
            ValidationIssue(
                "unsafe_markdown_path",
                f"Markdown path is unsafe: {item.cleaned_markdown_path}",
            )
        )
    else:
        assert descriptor is not None
        actual_hash = _sha256_descriptor(descriptor)
        if actual_hash != item.cleaned_markdown_sha256:
            issues.append(
                ValidationIssue(
                    "markdown_hash_mismatch",
                    f"Markdown hash does not match: {item.cleaned_markdown_path}",
                )
            )

    if item.header_image_path is not None:
        header_status, descriptor = _open_relative_regular(
            config.source_root,
            item.header_image_path,
        )
        if descriptor is not None:
            os.close(descriptor)
        if header_status == "missing":
            issues.append(
                ValidationIssue(
                    "missing_header_image",
                    f"header image is missing: {item.header_image_path}",
                )
            )
        elif header_status == "unsafe":
            issues.append(
                ValidationIssue(
                    "unsafe_header_image_path",
                    f"header image path is unsafe: {item.header_image_path}",
                )
            )

    if len(set(item.inline_image_urls)) != len(item.inline_image_urls):
        issues.append(
            ValidationIssue(
                "unstable_inline_image_urls",
                f"inline image URL list is not stably deduplicated: {item.article_id}",
            )
        )
    invalid_url_count = sum(
        not _valid_http_url(value) for value in item.inline_image_urls
    )
    if invalid_url_count:
        issues.append(
            ValidationIssue(
                "invalid_inline_image_url",
                f"article has invalid inline image URLs: {item.article_id}: "
                f"{invalid_url_count}",
            )
        )


def _expected_markdown_path(article_id: str, publication_date: date) -> str:
    digest = hashlib.sha256(article_id.encode()).hexdigest()
    return (
        f"text/{publication_date.year:04d}/{publication_date.month:02d}/"
        f"{publication_date.day:02d}/{digest}.md"
    )


def _valid_http_url(value: object) -> bool:
    if not isinstance(value, str) or not value or any(
        character.isspace() for character in value
    ):
        return False
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError:
        return False
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.hostname)


def _open_relative_regular(root: Path, relative_value: str) -> tuple[str, int | None]:
    relative = Path(relative_value)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(not _safe_component(part) for part in relative.parts)
    ):
        return "unsafe", None

    try:
        root_descriptor = os.open(root, _DIRECTORY_FLAGS)
    except FileNotFoundError:
        return "missing", None
    except OSError:
        return "unsafe", None

    parent_descriptor = os.dup(root_descriptor)
    file_descriptor: int | None = None
    try:
        for component in relative.parts[:-1]:
            try:
                next_descriptor = os.open(
                    component,
                    _DIRECTORY_FLAGS,
                    dir_fd=parent_descriptor,
                )
            except FileNotFoundError:
                return "missing", None
            except OSError:
                return "unsafe", None
            os.close(parent_descriptor)
            parent_descriptor = next_descriptor

        try:
            file_descriptor = os.open(
                relative.parts[-1],
                _READ_FLAGS,
                dir_fd=parent_descriptor,
            )
        except FileNotFoundError:
            return "missing", None
        except OSError as error:
            if error.errno == errno.ENOENT:
                return "missing", None
            return "unsafe", None
        file_stat = os.fstat(file_descriptor)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
            os.close(file_descriptor)
            file_descriptor = None
            return "unsafe", None
        return "ok", file_descriptor
    finally:
        os.close(parent_descriptor)
        os.close(root_descriptor)


def _safe_component(component: str) -> bool:
    return (
        bool(component)
        and component not in {".", ".."}
        and Path(component).name == component
    )


def _sha256_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            while chunk := stream.read(_HASH_CHUNK_SIZE):
                digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _validate_orphans(
    config: PipelineConfig,
    connection: duckdb.DuckDBPyConnection,
    issues: list[ValidationIssue],
) -> None:
    for path in _iter_markdown_files(config.text_root):
        relative_path = (Path("text") / path.relative_to(config.text_root)).as_posix()
        referenced = connection.execute(
            "SELECT count(*) FROM articles WHERE cleaned_markdown_path = ?",
            [relative_path],
        ).fetchone()[0]
        if not referenced:
            issues.append(
                ValidationIssue(
                    "orphan_markdown",
                    f"generated Markdown is not catalogued: {relative_path}",
                )
            )


def _iter_markdown_files(root: Path) -> Iterator[Path]:
    """Yield Markdown tree entries without following directory symlinks."""

    try:
        root_descriptor = os.open(root, _DIRECTORY_FLAGS)
    except OSError:
        return
    try:
        yield from _walk_markdown(root, root_descriptor, Path())
    finally:
        os.close(root_descriptor)


def _walk_markdown(
    root: Path,
    directory_descriptor: int,
    relative_directory: Path,
) -> Iterator[Path]:
    with os.scandir(directory_descriptor) as entries:
        for entry in entries:
            relative_path = relative_directory / entry.name
            if entry.is_symlink():
                continue
            if entry.is_dir(follow_symlinks=False):
                try:
                    child_descriptor = os.open(
                        entry.name,
                        _DIRECTORY_FLAGS,
                        dir_fd=directory_descriptor,
                    )
                except OSError:
                    continue
                try:
                    yield from _walk_markdown(
                        root,
                        child_descriptor,
                        relative_path,
                    )
                finally:
                    os.close(child_descriptor)
            elif entry.name.endswith(".md"):
                yield root / relative_path
