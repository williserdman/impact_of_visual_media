"""Deterministic canonical Markdown selection and pipeline results."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Collection, Iterable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from datetime import date
from itertools import islice
from pathlib import Path

from wsj_pipeline.catalog import (
    ArticleRecord,
    CandidateRecord,
    Catalog,
    DuplicateRecord,
)
from wsj_pipeline.config import PipelineConfig
from wsj_pipeline.discovery import discover_sources
from wsj_pipeline.extract import EXTRACTOR_VERSION, ExtractionError, extract_article
from wsj_pipeline.models import ExtractedArticle, SourceCandidate


class PipelineLockedError(RuntimeError):
    """Raised when another mutating pipeline process owns the lock."""


@dataclass(frozen=True, slots=True)
class PipelineSummary:
    """Content-free counts and validation status for one pipeline run."""

    discovered: int = 0
    new: int = 0
    changed: int = 0
    metadata_only: int = 0
    unchanged: int = 0
    stale: int = 0
    reprocessed: int = 0
    succeeded: int = 0
    failed: int = 0
    removed: int = 0
    articles: int = 0
    validation_ok: bool = False


@contextmanager
def pipeline_lock(lock_path: Path) -> Iterator[None]:
    """Own an exclusive lock file for one mutating pipeline run."""

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as error:
        raise PipelineLockedError(
            f"pipeline is already running; lock exists at {lock_path}"
        ) from error

    try:
        try:
            os.write(descriptor, f"{os.getpid()}\n".encode())
        finally:
            os.close(descriptor)
        yield
    finally:
        with suppress(FileNotFoundError):
            lock_path.unlink()


def ensure_no_legacy_output(config: PipelineConfig) -> None:
    """Refuse an unmigrated derived-output tree without altering it."""

    if config.catalog_db.exists():
        return
    legacy_markers = (
        config.output_root / "state" / "pipeline.duckdb",
        config.output_root / "parquet",
        config.output_root / "index" / "wsj.duckdb",
    )
    if any(marker.exists() for marker in legacy_markers):
        raise RuntimeError(
            "legacy derived output detected; move derived output aside or "
            "choose a fresh output root"
        )


def _is_single_path_component(value: str) -> bool:
    return bool(value) and value not in {".", ".."} and Path(value).name == value


def _open_directory(path: Path) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    return os.open(path, flags)


def _open_directory_components(
    root_descriptor: int,
    components: Iterable[str],
    *,
    create: bool,
    mode: int,
) -> int:
    """Open a relative directory chain without following any component."""

    descriptor = os.dup(root_descriptor)
    try:
        for component in components:
            if not _is_single_path_component(component):
                raise ValueError("directory path contains an unsafe component")
            if create:
                with suppress(FileExistsError):
                    os.mkdir(component, mode=mode, dir_fd=descriptor)
            next_descriptor = os.open(
                component,
                os.O_RDONLY
                | os.O_DIRECTORY
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def write_markdown_atomic(
    markdown: str,
    target: Path,
    staging_root: Path,
    run_id: str,
) -> str:
    """Publish verified UTF-8 Markdown with one atomic replacement."""

    output_root = staging_root.parent.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    target_parent = target.parent.resolve()
    if not target.name or not target_parent.is_relative_to(output_root):
        raise ValueError("target must be inside output root")
    if not _is_single_path_component(run_id):
        raise ValueError("run staging path must be inside staging root")

    relative_target = (
        target_parent.relative_to(output_root) / target.name
    ).as_posix()
    staged_name = f"{hashlib.sha256(relative_target.encode()).hexdigest()}.md"
    markdown_bytes = markdown.encode("utf-8")
    expected_hash = hashlib.sha256(markdown_bytes).hexdigest()

    root_descriptor = _open_directory(output_root)
    staging_descriptor: int | None = None
    run_descriptor: int | None = None
    target_descriptor: int | None = None
    try:
        try:
            staging_descriptor = _open_directory_components(
                root_descriptor,
                (staging_root.name,),
                create=True,
                mode=0o700,
            )
        except OSError as error:
            raise ValueError(
                "staging root must be inside output root and not a symlink"
            ) from error
        run_descriptor = _open_directory_components(
            staging_descriptor,
            (run_id,),
            create=True,
            mode=0o700,
        )
        target_descriptor = _open_directory_components(
            root_descriptor,
            target_parent.relative_to(output_root).parts,
            create=True,
            mode=0o755,
        )

        create_flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        create_flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(
            staged_name,
            create_flags,
            0o600,
            dir_fd=run_descriptor,
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(markdown_bytes)
        finally:
            os.close(descriptor)

        read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(
            staged_name,
            read_flags,
            dir_fd=run_descriptor,
        )
        try:
            staged_stat = os.fstat(descriptor)
            if not stat.S_ISREG(staged_stat.st_mode) or staged_stat.st_nlink != 1:
                raise ValueError("staged Markdown must be one regular file")
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                staged_bytes = stream.read()
        finally:
            os.close(descriptor)
        staged_hash = hashlib.sha256(staged_bytes).hexdigest()
        if staged_hash != expected_hash:
            raise ValueError("staged Markdown verification failed")
        os.replace(
            staged_name,
            target.name,
            src_dir_fd=run_descriptor,
            dst_dir_fd=target_descriptor,
        )
        return staged_hash
    finally:
        if run_descriptor is not None:
            with suppress(FileNotFoundError):
                os.unlink(staged_name, dir_fd=run_descriptor)
            os.close(run_descriptor)
        if staging_descriptor is not None:
            with suppress(OSError):
                os.rmdir(run_id, dir_fd=staging_descriptor)
            os.close(staging_descriptor)
        if target_descriptor is not None:
            os.close(target_descriptor)
        os.close(root_descriptor)


def run_pipeline(
    config: PipelineConfig,
    *,
    limit: int | None,
    full: bool,
    reprocess: bool = False,
) -> PipelineSummary:
    """Incrementally extract and publish bounded transactional batches."""

    _require_scope(limit, full)
    if config.batch_size < 1:
        raise ValueError("batch_size must be positive")
    ensure_no_legacy_output(config)
    scope = "full" if full else f"limit:{limit}"
    if reprocess:
        scope = f"{scope},reprocess"
    counters = _new_summary_counters()

    with pipeline_lock(config.lock_path), Catalog.open(config.catalog_db) as catalog:
        run_id = catalog.begin_run("run", scope)
        try:
            candidates = discover_sources(
                config.source_root,
                limit=None if full else limit,
            )
            for batch in _batches(candidates, config.batch_size):
                orphan_candidates: list[tuple[str, str]] = []
                with catalog.transaction():
                    for candidate in batch:
                        counters["discovered"] += 1
                        orphan_candidates.extend(
                            _process_candidate(
                                catalog,
                                config,
                                candidate,
                                run_id,
                                counters,
                                reprocess=reprocess,
                            )
                        )
                _cleanup_orphans(catalog, config, orphan_candidates)

            summary = _pipeline_summary(counters, catalog.article_count())
            catalog.finish_run(run_id, "succeeded", asdict(summary))
            return summary
        except Exception:
            with suppress(Exception):
                failed_summary = _pipeline_summary(
                    counters,
                    catalog.article_count(),
                )
                catalog.finish_run(run_id, "failed", asdict(failed_summary))
            raise


def _new_summary_counters() -> dict[str, int]:
    return {
        "discovered": 0,
        "new": 0,
        "changed": 0,
        "metadata_only": 0,
        "unchanged": 0,
        "stale": 0,
        "reprocessed": 0,
        "succeeded": 0,
        "failed": 0,
        "removed": 0,
    }


def _pipeline_summary(
    counters: dict[str, int],
    article_count: int,
) -> PipelineSummary:
    return PipelineSummary(
        discovered=counters["discovered"],
        new=counters["new"],
        changed=counters["changed"],
        metadata_only=counters["metadata_only"],
        unchanged=counters["unchanged"],
        stale=counters["stale"],
        reprocessed=counters["reprocessed"],
        succeeded=counters["succeeded"],
        failed=counters["failed"],
        removed=counters["removed"],
        articles=article_count,
    )


def _batches(
    candidates: Iterable[SourceCandidate],
    batch_size: int,
) -> Iterator[tuple[SourceCandidate, ...]]:
    iterator = iter(candidates)
    while batch := tuple(islice(iterator, batch_size)):
        yield batch


def _process_candidate(
    catalog: Catalog,
    config: PipelineConfig,
    candidate: SourceCandidate,
    run_id: str,
    counters: dict[str, int],
    *,
    reprocess: bool,
) -> tuple[tuple[str, str], ...]:
    classification = catalog.classify(candidate, EXTRACTOR_VERSION)
    if classification.kind == "unchanged":
        counters["unchanged"] += 1
        catalog.mark_seen(run_id, candidate.relative_html_path)
        return ()
    if classification.kind == "stale_extractor" and not reprocess:
        counters["stale"] += 1
        catalog.mark_seen(run_id, candidate.relative_html_path)
        return ()

    html_path = config.source_root / candidate.relative_html_path
    source_html_sha256 = hashlib.sha256(html_path.read_bytes()).hexdigest()
    if (
        classification.kind == "stat_changed"
        and not classification.was_failed
        and not classification.image_changed
        and source_html_sha256 == classification.old_html_sha256
    ):
        counters["metadata_only"] += 1
        catalog.replace_manifest(
            run_id,
            candidate,
            extractor_version=EXTRACTOR_VERSION,
            status="succeeded",
            source_html_sha256=source_html_sha256,
            article_id=classification.old_article_id,
        )
        return ()

    if classification.kind == "new":
        counters["new"] += 1
    elif classification.kind == "stale_extractor":
        counters["reprocessed"] += 1
    else:
        counters["changed"] += 1

    try:
        article = extract_article(candidate, config.source_root)
    except ExtractionError as error:
        counters["failed"] += 1
        catalog.replace_failure(
            run_id,
            candidate.relative_html_path,
            stage="extract",
            error_code=error.code,
            extractor_version=EXTRACTOR_VERSION,
        )
        catalog.replace_manifest(
            run_id,
            candidate,
            extractor_version=EXTRACTOR_VERSION,
            status="failed",
            source_html_sha256=source_html_sha256,
            article_id=classification.old_article_id,
        )
        return ()

    orphan_candidates: tuple[tuple[str, str], ...] = ()
    if (
        classification.old_article_id is not None
        and classification.old_article_id != article.article_id
    ):
        catalog.replace_candidate(run_id, article)
        _, old_orphans = _recompute_article_in_transaction(
            catalog,
            config,
            classification.old_article_id,
            run_id,
            current_articles=(),
            remove_if_empty=True,
        )
        orphan_candidates += old_orphans
    winner, new_orphans = _recompute_article_in_transaction(
        catalog,
        config,
        article.article_id,
        run_id,
        current_articles=(article,),
    )
    orphan_candidates += new_orphans
    if winner is None:
        raise RuntimeError("extracted article did not produce a winner")
    catalog.replace_manifest(
        run_id,
        candidate,
        extractor_version=EXTRACTOR_VERSION,
        status="succeeded",
        source_html_sha256=article.source_html_sha256,
        article_id=article.article_id,
    )
    counters["succeeded"] += 1
    return orphan_candidates


def _require_scope(limit: int | None, full: bool) -> None:
    if full == (limit is not None):
        raise ValueError("choose exactly one of a positive limit or full mode")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")


def article_markdown_path(article_id: str, publication_date: date) -> Path:
    """Return the portable dated path for one stable article identity."""

    identity_hash = hashlib.sha256(article_id.encode("utf-8")).hexdigest()
    return (
        Path("text")
        / f"{publication_date.year:04d}"
        / f"{publication_date.month:02d}"
        / f"{publication_date.day:02d}"
        / f"{identity_hash}.md"
    )


def candidate_rank(article: ExtractedArticle) -> tuple[object, ...]:
    """Return an ascending key in the approved winner preference order."""

    return _rank_values(
        has_content=bool(article.markdown_text),
        character_count=article.editorial_character_count,
        block_count=article.editorial_block_count,
        completeness=article.metadata_completeness,
        updated_at_timestamp=(
            article.updated_at_utc.timestamp()
            if article.updated_at_utc is not None
            else None
        ),
        source_path=article.relative_html_path,
    )


def recompute_article(
    catalog: Catalog,
    config: PipelineConfig,
    article_id: str,
    run_id: str,
    *,
    current_articles: Iterable[ExtractedArticle],
    invalid_source_paths: Collection[str] = (),
) -> ArticleRecord | None:
    """Recompute one affected identity and publish its selected Markdown.

    Current extraction objects are preferred as the source of Markdown. A raw
    source is re-extracted only when a different stored candidate must be
    promoted and no current object exists for it.
    """

    with catalog.transaction():
        winner, orphan_candidates = _recompute_article_in_transaction(
            catalog,
            config,
            article_id,
            run_id,
            current_articles=current_articles,
            invalid_source_paths=invalid_source_paths,
        )
    _cleanup_orphans(catalog, config, orphan_candidates)
    return winner


def _recompute_article_in_transaction(
    catalog: Catalog,
    config: PipelineConfig,
    article_id: str,
    run_id: str,
    *,
    current_articles: Iterable[ExtractedArticle],
    invalid_source_paths: Collection[str] = (),
    remove_if_empty: bool = False,
) -> tuple[ArticleRecord | None, tuple[tuple[str, str], ...]]:
    invalid_paths = frozenset(invalid_source_paths)
    current_by_path = {
        article.relative_html_path: article
        for article in current_articles
        if article.article_id == article_id
        and article.relative_html_path not in invalid_paths
    }

    for article in current_by_path.values():
        catalog.replace_candidate(run_id, article)

    existing = catalog.article_for_id(article_id)
    while True:
        candidates = tuple(
            candidate
            for candidate in catalog.candidates_for_article(article_id)
            if candidate.source_html_path not in invalid_paths
        )
        if not candidates:
            catalog.replace_duplicates(run_id, article_id, "", ())
            if not remove_if_empty:
                return None, ()
            removed = catalog.remove_article(article_id)
            return None, _orphan_candidate(removed, None)

        ranked = tuple(sorted(candidates, key=_stored_candidate_rank))
        selected = ranked[0]
        winner_article = current_by_path.get(selected.source_html_path)

        if winner_article is not None:
            winner_record = _publish_markdown(config, winner_article, run_id)
            catalog.replace_article(winner_record)
            break
        if _can_reuse_existing(existing, selected, config):
            winner_record = existing
            break

        winner_article = _reextract(config, selected)
        if winner_article.article_id != article_id:
            raise ValueError(
                "stored candidate identity changed during re-extraction"
            )
        catalog.replace_candidate(run_id, winner_article)
        current_by_path[selected.source_html_path] = winner_article

    catalog.replace_duplicates(
        run_id,
        article_id,
        selected.source_html_path,
        tuple(
            DuplicateRecord(
                source_html_path=candidate.source_html_path,
                duplicate_rank=duplicate_rank,
                reason=_duplicate_reason(selected, candidate),
            )
            for duplicate_rank, candidate in enumerate(ranked[1:], start=2)
        ),
    )
    orphan_candidates = _orphan_candidate(
        existing,
        winner_record.cleaned_markdown_path,
    )
    return winner_record, orphan_candidates


def _orphan_candidate(
    existing: ArticleRecord | None,
    replacement_path: str | None,
) -> tuple[tuple[str, str], ...]:
    if (
        existing is None
        or existing.cleaned_markdown_path == replacement_path
        or existing.cleaned_markdown_path
        != article_markdown_path(
            existing.article_id,
            existing.publication_date_new_york,
        ).as_posix()
    ):
        return ()
    return ((existing.article_id, existing.cleaned_markdown_path),)


def _rank_values(
    *,
    has_content: bool,
    character_count: int,
    block_count: int,
    completeness: int,
    updated_at_timestamp: float | None,
    source_path: str,
) -> tuple[object, ...]:
    updated_score = (
        -updated_at_timestamp
        if updated_at_timestamp is not None
        else float("inf")
    )
    return (
        0 if has_content else 1,
        -character_count,
        -block_count,
        -completeness,
        updated_score,
        source_path,
    )


def _stored_candidate_rank(candidate: CandidateRecord) -> tuple[object, ...]:
    return _rank_values(
        has_content=candidate.editorial_character_count > 0,
        character_count=candidate.editorial_character_count,
        block_count=candidate.editorial_block_count,
        completeness=candidate.metadata_completeness,
        updated_at_timestamp=(
            candidate.updated_at_utc.timestamp()
            if candidate.updated_at_utc is not None
            else None
        ),
        source_path=candidate.source_html_path,
    )


def _duplicate_reason(
    winner: CandidateRecord,
    duplicate: CandidateRecord,
) -> str:
    if (winner.editorial_character_count > 0) != (
        duplicate.editorial_character_count > 0
    ):
        return "empty_content"
    if winner.editorial_character_count != duplicate.editorial_character_count:
        return "lower_editorial_character_count"
    if winner.editorial_block_count != duplicate.editorial_block_count:
        return "lower_editorial_block_count"
    if winner.metadata_completeness != duplicate.metadata_completeness:
        return "lower_metadata_completeness"
    if winner.updated_at_utc != duplicate.updated_at_utc:
        return "older_update"
    return "lexical_tiebreak"


def _can_reuse_existing(
    existing: ArticleRecord | None,
    selected: CandidateRecord,
    config: PipelineConfig,
) -> bool:
    if existing is None or existing.source_html_path != selected.source_html_path:
        return False
    markdown_path = config.output_root / existing.cleaned_markdown_path
    if not markdown_path.is_file():
        return False
    return (
        hashlib.sha256(markdown_path.read_bytes()).hexdigest()
        == existing.cleaned_markdown_sha256
    )


def _reextract(
    config: PipelineConfig,
    candidate: CandidateRecord,
) -> ExtractedArticle:
    html_path = config.source_root / candidate.source_html_path
    html_stat = html_path.stat()
    header_path = (
        config.source_root / candidate.header_image_path
        if candidate.header_image_path is not None
        else None
    )
    header_stat = (
        header_path.stat()
        if header_path is not None and header_path.is_file()
        else None
    )
    source = SourceCandidate(
        relative_html_path=candidate.source_html_path,
        html_size=html_stat.st_size,
        html_mtime_ns=html_stat.st_mtime_ns,
        relative_header_image_path=(
            candidate.header_image_path if header_stat is not None else None
        ),
        header_image_size=header_stat.st_size if header_stat is not None else None,
        header_image_mtime_ns=(
            header_stat.st_mtime_ns if header_stat is not None else None
        ),
        warnings=candidate.warnings,
    )
    return extract_article(source, config.source_root)


def _publish_markdown(
    config: PipelineConfig,
    article: ExtractedArticle,
    run_id: str,
) -> ArticleRecord:
    relative_path = article_markdown_path(
        article.article_id,
        article.publication_date_new_york,
    )
    target = config.output_root / relative_path
    markdown_hash = write_markdown_atomic(
        article.markdown_text,
        target,
        config.staging_root,
        run_id,
    )
    return ArticleRecord(
        article_id=article.article_id,
        source_html_path=article.relative_html_path,
        cleaned_markdown_path=relative_path.as_posix(),
        header_image_path=article.relative_header_image_path,
        inline_image_urls=article.inline_image_urls,
        published_at_utc=article.published_at_utc,
        publication_date_new_york=article.publication_date_new_york,
        cleaned_markdown_sha256=markdown_hash,
    )


def _cleanup_orphans(
    catalog: Catalog,
    config: PipelineConfig,
    orphan_candidates: Iterable[tuple[str, str]],
) -> None:
    for article_id, relative_path in sorted(set(orphan_candidates)):
        current = catalog.article_for_id(article_id)
        if current is not None and current.cleaned_markdown_path == relative_path:
            continue
        with suppress(OSError):
            _unlink_generated_file(config.output_root, relative_path)


def _unlink_generated_file(output_root: Path, relative_path: str) -> None:
    relative = Path(relative_path)
    if (
        relative.is_absolute()
        or not relative.name
        or any(not _is_single_path_component(part) for part in relative.parts)
    ):
        return

    resolved_root = output_root.resolve()
    root_descriptor = _open_directory(resolved_root)
    parent_descriptor: int | None = None
    file_descriptor: int | None = None
    try:
        parent_descriptor = _open_directory_components(
            root_descriptor,
            relative.parent.parts,
            create=False,
            mode=0o755,
        )
        file_descriptor = os.open(
            relative.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        file_stat = os.fstat(file_descriptor)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
            return
        os.unlink(relative.name, dir_fd=parent_descriptor)
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)
        os.close(root_descriptor)
