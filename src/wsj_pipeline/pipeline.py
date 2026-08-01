"""Deterministic canonical Markdown selection and pipeline results."""

from __future__ import annotations

import hashlib
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from wsj_pipeline.catalog import (
    ArticleRecord,
    CandidateRecord,
    Catalog,
    DuplicateRecord,
)
from wsj_pipeline.config import PipelineConfig
from wsj_pipeline.extract import extract_article
from wsj_pipeline.models import ExtractedArticle, SourceCandidate


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

    invalid_paths = frozenset(invalid_source_paths)
    current_by_path = {
        article.relative_html_path: article
        for article in current_articles
        if article.article_id == article_id
        and article.relative_html_path not in invalid_paths
    }

    with catalog.transaction():
        for article in current_by_path.values():
            catalog.replace_candidate(run_id, article)

        candidates = tuple(
            candidate
            for candidate in catalog.candidates_for_article(article_id)
            if candidate.source_html_path not in invalid_paths
        )
        if not candidates:
            catalog.replace_duplicates(run_id, article_id, "", ())
            return None

        ranked = tuple(sorted(candidates, key=_stored_candidate_rank))
        selected = ranked[0]
        existing = catalog.article_for_id(article_id)
        winner_article = current_by_path.get(selected.source_html_path)

        if winner_article is None and _can_reuse_existing(existing, selected, config):
            winner_record = existing
        else:
            if winner_article is None:
                winner_article = _reextract(config, selected)
                if winner_article.article_id != article_id:
                    raise ValueError(
                        "stored candidate identity changed during re-extraction"
                    )
                catalog.replace_candidate(run_id, winner_article)
            winner_record = _publish_markdown(config, winner_article)
            catalog.replace_article(winner_record)

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
        return winner_record


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
) -> ArticleRecord:
    relative_path = article_markdown_path(
        article.article_id,
        article.publication_date_new_york,
    )
    target = config.output_root / relative_path
    markdown_bytes = article.markdown_text.encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(markdown_bytes)
    return ArticleRecord(
        article_id=article.article_id,
        source_html_path=article.relative_html_path,
        cleaned_markdown_path=relative_path.as_posix(),
        header_image_path=article.relative_header_image_path,
        inline_image_urls=article.inline_image_urls,
        published_at_utc=article.published_at_utc,
        publication_date_new_york=article.publication_date_new_york,
        cleaned_markdown_sha256=hashlib.sha256(markdown_bytes).hexdigest(),
    )
