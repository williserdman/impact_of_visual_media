from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from tests.fixtures import write_article_fixture
from wsj_pipeline.canonical import (
    PartitionKey,
    derive_article_key,
    rank_candidate,
    recompute_canonical,
)
from wsj_pipeline.discovery import discover_sources
from wsj_pipeline.extract import extract_article
from wsj_pipeline.state import PipelineState


def test_identity_prefers_trimmed_wsj_article_id() -> None:
    assert (
        derive_article_key(
            " WP-WSJ-000123 ",
            "https://www.wsj.com/ignored",
            "a" * 64,
        )
        == "wsj:WP-WSJ-000123"
    )


def test_url_identity_normalizes_equivalent_canonical_urls() -> None:
    first = derive_article_key(
        None,
        "HTTPS://WWW.WSJ.COM:443/business/story/?b=2&a=1#comments",
        "a" * 64,
    )
    second = derive_article_key(
        None,
        "https://www.wsj.com/business/story?a=1&b=2",
        "b" * 64,
    )

    assert first == second
    assert first.startswith("url:")


def test_identity_falls_back_to_html_hash() -> None:
    assert derive_article_key(None, None, "abc123") == "html:abc123"


def test_ranking_prefers_body_then_word_count_then_metadata_then_update(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    write_article_fixture(archive, "2016/story.html")
    [candidate] = list(discover_sources(archive))
    base = extract_article(candidate, archive)
    no_body = replace(base, body_text="", body_paragraphs=(), body_word_count=0)
    short = replace(base, body_word_count=5, metadata_completeness=9)
    complete = replace(
        base,
        body_word_count=10,
        metadata_completeness=10,
        updated_at_utc=datetime(2024, 1, 2, 18, tzinfo=UTC),
    )
    older = replace(
        complete,
        updated_at_utc=datetime(2024, 1, 2, 17, tzinfo=UTC),
    )

    ordered = sorted(
        [no_body, short, older, complete],
        key=rank_candidate,
    )

    assert ordered == [complete, older, short, no_body]


def store_duplicate_sources(
    tmp_path: Path,
    *,
    first_paragraphs: tuple[str, ...],
    second_paragraphs: tuple[str, ...],
):
    archive = tmp_path / "archive"
    write_article_fixture(
        archive,
        "2016/a.html",
        article_id="WP-DUPLICATE",
        paragraphs=first_paragraphs,
    )
    write_article_fixture(
        archive,
        "2016/b.html",
        article_id="WP-DUPLICATE",
        paragraphs=second_paragraphs,
    )
    state_path = tmp_path / "pipeline.duckdb"
    with PipelineState.open(state_path) as state:
        run_id = state.begin_run("test", "fixture")
        for candidate in discover_sources(archive):
            state.store_success(
                run_id,
                candidate,
                extract_article(candidate, archive),
            )
        summary = recompute_canonical(
            state,
            ["wsj:WP-DUPLICATE"],
            run_id,
        )
    return state_path, summary


def test_recompute_selects_longer_body_and_audits_nonwinner(
    tmp_path: Path,
) -> None:
    state_path, summary = store_duplicate_sources(
        tmp_path,
        first_paragraphs=("Short body.",),
        second_paragraphs=(
            "This second source has a substantially longer article body.",
        ),
    )

    with duckdb.connect(str(state_path), read_only=True) as connection:
        winner = connection.execute(
            "SELECT source_path FROM canonical_sources"
        ).fetchone()[0]
        duplicate = connection.execute(
            """
            SELECT source_path, winner_source_path, duplicate_rank, reason
            FROM duplicates
            """
        ).fetchone()

    assert winner == "2016/b.html"
    assert duplicate == (
        "2016/a.html",
        "2016/b.html",
        2,
        "shorter_body",
    )
    assert summary.article_keys_recomputed == 1
    assert summary.duplicates_recorded == 1
    assert summary.affected_partitions == (PartitionKey(2024, 1),)


def test_lexical_source_path_is_final_tie_breaker(tmp_path: Path) -> None:
    state_path, _summary = store_duplicate_sources(
        tmp_path,
        first_paragraphs=("Identical words here.",),
        second_paragraphs=("Identical words here.",),
    )

    with duckdb.connect(str(state_path), read_only=True) as connection:
        winner = connection.execute(
            "SELECT source_path FROM canonical_sources"
        ).fetchone()[0]
        reason = connection.execute("SELECT reason FROM duplicates").fetchone()[0]

    assert winner == "2016/a.html"
    assert reason == "lexical_tiebreak"
