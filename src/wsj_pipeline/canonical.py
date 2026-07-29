"""Stable article identity and deterministic duplicate selection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from wsj_pipeline.models import ExtractedArticle

if TYPE_CHECKING:
    from collections.abc import Collection

    from wsj_pipeline.state import PipelineState


@dataclass(frozen=True, order=True, slots=True)
class PartitionKey:
    """One New York publication year/month partition."""

    year: int
    month: int


@dataclass(frozen=True, slots=True)
class CanonicalSummary:
    """Results of recomputing a bounded set of article identities."""

    article_keys_recomputed: int
    duplicates_recorded: int
    affected_partitions: tuple[PartitionKey, ...]


def derive_article_key(
    wsj_id: str | None,
    canonical_url: str | None,
    html_sha256: str,
) -> str:
    """Derive a namespaced stable identity using documented fallbacks."""

    if wsj_id and wsj_id.strip():
        return f"wsj:{wsj_id.strip()}"
    if canonical_url and canonical_url.strip():
        normalized = _normalize_canonical_url(canonical_url)
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"url:{digest}"
    return f"html:{html_sha256}"


def rank_candidate(article: ExtractedArticle) -> tuple[object, ...]:
    """Return an ascending sort key whose first row is the winner."""

    updated_score = (
        -article.updated_at_utc.timestamp()
        if article.updated_at_utc is not None
        else float("inf")
    )
    return (
        0 if article.published_at_utc is not None else 1,
        0 if article.body_text else 1,
        -article.body_word_count,
        -article.metadata_completeness,
        updated_score,
        article.relative_html_path,
    )


def recompute_canonical(
    state: PipelineState,
    article_keys: Collection[str],
    run_id: str,
) -> CanonicalSummary:
    """Replace canonical and duplicate decisions for affected identities."""

    affected_partitions: set[PartitionKey] = set()
    duplicates_recorded = 0
    keys = sorted(set(article_keys))

    state.connection.execute("BEGIN TRANSACTION")
    try:
        for article_key in keys:
            old = state.connection.execute(
                """
                SELECT publication_year_ny, publication_month_ny
                FROM canonical_sources
                WHERE article_key = ?
                """,
                [article_key],
            ).fetchone()
            if old:
                affected_partitions.add(PartitionKey(old[0], old[1]))

            rows = state.connection.execute(
                """
                SELECT source_path, payload_json
                FROM extracted_sources
                WHERE article_key = ?
                ORDER BY source_path
                """,
                [article_key],
            ).fetchall()
            state.connection.execute(
                "DELETE FROM duplicates WHERE article_key = ?",
                [article_key],
            )
            state.connection.execute(
                "DELETE FROM canonical_sources WHERE article_key = ?",
                [article_key],
            )
            if not rows:
                continue

            articles = [
                deserialize_article(payload_json) for _source_path, payload_json in rows
            ]
            articles.sort(key=rank_candidate)
            winner = articles[0]
            affected_partitions.add(
                PartitionKey(
                    winner.publication_year_ny,
                    winner.publication_month_ny,
                )
            )
            state.connection.execute(
                "INSERT INTO canonical_sources VALUES (?, ?, ?, ?, ?)",
                [
                    article_key,
                    winner.relative_html_path,
                    winner.publication_year_ny,
                    winner.publication_month_ny,
                    run_id,
                ],
            )
            for duplicate_rank, duplicate in enumerate(articles[1:], start=2):
                state.connection.execute(
                    "INSERT INTO duplicates VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        article_key,
                        duplicate.relative_html_path,
                        winner.relative_html_path,
                        duplicate_rank,
                        _duplicate_reason(winner, duplicate),
                        run_id,
                    ],
                )
                duplicates_recorded += 1
    except Exception:
        state.connection.execute("ROLLBACK")
        raise
    else:
        state.connection.execute("COMMIT")

    return CanonicalSummary(
        article_keys_recomputed=len(keys),
        duplicates_recorded=duplicates_recorded,
        affected_partitions=tuple(sorted(affected_partitions)),
    )


def deserialize_article(payload_json: str) -> ExtractedArticle:
    """Restore an extracted article from deterministic operational JSON."""

    raw = json.loads(payload_json)
    for field_name in (
        "published_at_utc",
        "published_at_new_york",
        "updated_at_utc",
        "created_at_utc",
        "last_published_at_utc",
    ):
        if raw[field_name] is not None:
            raw[field_name] = datetime.fromisoformat(raw[field_name])
    for field_name in (
        "publication_date_utc",
        "publication_date_new_york",
    ):
        raw[field_name] = date.fromisoformat(raw[field_name])
    for field_name in (
        "authors",
        "keywords",
        "body_paragraphs",
        "warnings",
    ):
        raw[field_name] = tuple(raw[field_name])
    return ExtractedArticle(**raw)


def _normalize_canonical_url(value: str) -> str:
    parts = urlsplit(value.strip())
    scheme = parts.scheme.casefold()
    hostname = (parts.hostname or "").casefold()
    port = parts.port
    include_port = port is not None and not (
        (scheme == "https" and port == 443) or (scheme == "http" and port == 80)
    )
    netloc = f"{hostname}:{port}" if include_port else hostname
    path = parts.path or "/"
    if path != "/":
        path = path.rstrip("/")
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return urlunsplit((scheme, netloc, path, query, ""))


def _duplicate_reason(
    winner: ExtractedArticle,
    duplicate: ExtractedArticle,
) -> str:
    if winner.published_at_utc is not None and duplicate.published_at_utc is None:
        return "missing_publication"
    if winner.body_text and not duplicate.body_text:
        return "empty_body"
    if winner.body_word_count != duplicate.body_word_count:
        return "shorter_body"
    if winner.metadata_completeness != duplicate.metadata_completeness:
        return "less_metadata"
    if winner.updated_at_utc != duplicate.updated_at_utc:
        return "older_update"
    return "lexical_tiebreak"
