from __future__ import annotations

from dataclasses import asdict, replace
from datetime import UTC, date, datetime
from hashlib import sha256
from pathlib import Path

import duckdb

from tests.fixtures import write_article_fixture
from wsj_pipeline.catalog import Catalog
from wsj_pipeline.config import PipelineConfig
from wsj_pipeline.discovery import discover_sources
from wsj_pipeline.extract import extract_article
from wsj_pipeline.models import ExtractedArticle
from wsj_pipeline.pipeline import (
    PipelineSummary,
    article_markdown_path,
    candidate_rank,
    recompute_article,
)


def _article(
    *,
    source_path: str = "2024/a.html",
    markdown_text: str = "Synthetic editorial content.",
    character_count: int = 28,
    block_count: int = 2,
    completeness: int = 5,
    updated_at: datetime | None = datetime(2024, 1, 2, 16, tzinfo=UTC),
) -> ExtractedArticle:
    return ExtractedArticle(
        article_id="wsj:WP-SYNTHETIC",
        relative_html_path=source_path,
        source_html_sha256="a" * 64,
        extractor_version="3",
        canonical_url="https://example.test/synthetic",
        published_at_utc=datetime(2024, 1, 2, 15, 30, tzinfo=UTC),
        publication_date_new_york=date(2024, 1, 2),
        updated_at_utc=updated_at,
        markdown_text=markdown_text,
        editorial_character_count=character_count,
        editorial_block_count=block_count,
        metadata_completeness=completeness,
        relative_header_image_path=None,
        inline_image_urls=(),
        warnings=(),
    )


def _duplicate_articles(
    tmp_path: Path,
) -> tuple[PipelineConfig, tuple[ExtractedArticle, ...]]:
    archive = tmp_path / "archive"
    write_article_fixture(
        archive,
        "2024/a.html",
        article_id="WP-DUPLICATE",
        paragraphs=("Short losing body.",),
    )
    write_article_fixture(
        archive,
        "2024/b.html",
        article_id="WP-DUPLICATE",
        paragraphs=(
            "This winning synthetic body has substantially more characters.",
        ),
    )
    articles = tuple(
        extract_article(candidate, archive)
        for candidate in discover_sources(archive)
    )
    return PipelineConfig(archive, tmp_path / "processed"), articles


def test_article_markdown_path_uses_new_york_date_and_hashed_identity() -> None:
    path = article_markdown_path("wsj:ABC", date(2024, 1, 15))

    assert path.parent.as_posix() == "text/2024/01/15"
    assert path.name == f"{sha256(b'wsj:ABC').hexdigest()}.md"


def test_pipeline_summary_exposes_only_approved_counts_and_validation() -> None:
    assert asdict(PipelineSummary()) == {
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
        "articles": 0,
        "validation_ok": False,
    }


def test_rank_prefers_nonempty_content_before_all_other_metrics() -> None:
    nonempty = _article(
        source_path="2024/z.html",
        markdown_text="x",
        character_count=1,
        block_count=1,
        completeness=1,
        updated_at=None,
    )
    empty = _article(
        source_path="2024/a.html",
        markdown_text="",
        character_count=100,
        block_count=100,
        completeness=100,
        updated_at=datetime(2025, 1, 1, tzinfo=UTC),
    )

    assert min((empty, nonempty), key=candidate_rank) is nonempty


def test_rank_prefers_character_count_before_lower_priority_metrics() -> None:
    longer = _article(
        source_path="2024/z.html",
        character_count=20,
        block_count=1,
        completeness=1,
        updated_at=None,
    )
    shorter = _article(
        source_path="2024/a.html",
        character_count=10,
        block_count=100,
        completeness=100,
        updated_at=datetime(2025, 1, 1, tzinfo=UTC),
    )

    assert min((shorter, longer), key=candidate_rank) is longer


def test_rank_prefers_block_count_before_lower_priority_metrics() -> None:
    more_blocks = _article(
        source_path="2024/z.html",
        block_count=3,
        completeness=1,
        updated_at=None,
    )
    fewer_blocks = _article(
        source_path="2024/a.html",
        block_count=2,
        completeness=100,
        updated_at=datetime(2025, 1, 1, tzinfo=UTC),
    )

    assert min((fewer_blocks, more_blocks), key=candidate_rank) is more_blocks


def test_rank_prefers_completeness_before_update_and_path() -> None:
    complete = _article(
        source_path="2024/z.html",
        completeness=6,
        updated_at=None,
    )
    incomplete = _article(
        source_path="2024/a.html",
        completeness=5,
        updated_at=datetime(2025, 1, 1, tzinfo=UTC),
    )

    assert min((incomplete, complete), key=candidate_rank) is complete


def test_rank_prefers_later_update_before_lexical_path() -> None:
    later = _article(
        source_path="2024/z.html",
        updated_at=datetime(2024, 1, 3, tzinfo=UTC),
    )
    earlier = _article(
        source_path="2024/a.html",
        updated_at=datetime(2024, 1, 2, tzinfo=UTC),
    )

    assert min((earlier, later), key=candidate_rank) is later


def test_rank_uses_lexical_source_path_as_final_tiebreaker() -> None:
    first = _article(source_path="2024/a.html")
    second = replace(first, relative_html_path="2024/b.html")

    assert min((second, first), key=candidate_rank) is first


def test_recompute_writes_one_winner_and_content_free_audit_rows(
    tmp_path: Path,
) -> None:
    config, articles = _duplicate_articles(tmp_path)
    database_path = config.catalog_db

    with Catalog.open(database_path) as catalog:
        run_id = catalog.begin_run("run", "fixture")
        winner = recompute_article(
            catalog,
            config,
            "wsj:WP-DUPLICATE",
            run_id,
            current_articles=reversed(articles),
        )

    assert winner is not None
    expected_path = article_markdown_path(
        winner.article_id,
        winner.publication_date_new_york,
    )
    markdown_path = config.output_root / expected_path
    assert winner.source_html_path == "2024/b.html"
    assert winner.cleaned_markdown_path == expected_path.as_posix()
    assert markdown_path.read_text(encoding="utf-8") == articles[1].markdown_text
    assert winner.cleaned_markdown_sha256 == sha256(
        articles[1].markdown_text.encode()
    ).hexdigest()

    with duckdb.connect(str(database_path), read_only=True) as connection:
        article_rows = connection.execute(
            "SELECT article_id, source_html_path, cleaned_markdown_path FROM articles"
        ).fetchall()
        candidate_rows = connection.execute(
            """
            SELECT source_html_path, article_id
            FROM candidates
            ORDER BY source_html_path
            """
        ).fetchall()
        duplicate_rows = connection.execute(
            """
            SELECT article_id, source_html_path, winner_source_html_path,
                   duplicate_rank, reason
            FROM duplicates
            """
        ).fetchall()
        candidate_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info('candidates')").fetchall()
        }

    assert article_rows == [
        (
            "wsj:WP-DUPLICATE",
            "2024/b.html",
            expected_path.as_posix(),
        )
    ]
    assert candidate_rows == [
        ("2024/a.html", "wsj:WP-DUPLICATE"),
        ("2024/b.html", "wsj:WP-DUPLICATE"),
    ]
    assert duplicate_rows == [
        (
            "wsj:WP-DUPLICATE",
            "2024/a.html",
            "2024/b.html",
            2,
            "lower_editorial_character_count",
        )
    ]
    assert "markdown_text" not in candidate_columns
    assert "Short losing body." not in repr((candidate_rows, duplicate_rows))


def test_recompute_reuses_valid_stored_winner_without_reextracting(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config, articles = _duplicate_articles(tmp_path)

    with Catalog.open(config.catalog_db) as catalog:
        run_id = catalog.begin_run("run", "fixture")
        first = recompute_article(
            catalog,
            config,
            "wsj:WP-DUPLICATE",
            run_id,
            current_articles=articles,
        )
        monkeypatch.setattr(
            "wsj_pipeline.pipeline.extract_article",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("valid stored winner was re-extracted")
            ),
        )

        second = recompute_article(
            catalog,
            config,
            "wsj:WP-DUPLICATE",
            run_id,
            current_articles=(),
        )

    assert second == first


def test_recompute_uses_current_alternative_when_stored_winner_is_invalid(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config, articles = _duplicate_articles(tmp_path)

    with Catalog.open(config.catalog_db) as catalog:
        run_id = catalog.begin_run("run", "fixture")
        recompute_article(
            catalog,
            config,
            "wsj:WP-DUPLICATE",
            run_id,
            current_articles=articles,
        )
        monkeypatch.setattr(
            "wsj_pipeline.pipeline.extract_article",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("available current article was re-extracted")
            ),
        )

        promoted = recompute_article(
            catalog,
            config,
            "wsj:WP-DUPLICATE",
            run_id,
            current_articles=(articles[0],),
            invalid_source_paths=("2024/b.html",),
        )

    assert promoted is not None
    assert promoted.source_html_path == "2024/a.html"


def test_recompute_reextracts_alternative_only_when_no_current_object_exists(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config, articles = _duplicate_articles(tmp_path)
    extracted_paths: list[str] = []

    with Catalog.open(config.catalog_db) as catalog:
        run_id = catalog.begin_run("run", "fixture")
        recompute_article(
            catalog,
            config,
            "wsj:WP-DUPLICATE",
            run_id,
            current_articles=articles,
        )
        real_extract = extract_article

        def recording_extract(candidate, source_root):
            extracted_paths.append(candidate.relative_html_path)
            return real_extract(candidate, source_root)

        monkeypatch.setattr("wsj_pipeline.pipeline.extract_article", recording_extract)

        promoted = recompute_article(
            catalog,
            config,
            "wsj:WP-DUPLICATE",
            run_id,
            current_articles=(),
            invalid_source_paths=("2024/b.html",),
        )

    assert promoted is not None
    assert promoted.source_html_path == "2024/a.html"
    assert extracted_paths == ["2024/a.html"]


def test_recompute_reranks_after_a_stored_alternative_degrades(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = PipelineConfig(tmp_path / "archive", tmp_path / "processed")
    for relative_path in ("2024/a.html", "2024/b.html", "2024/c.html"):
        raw_path = config.source_root / relative_path
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text("Synthetic raw placeholder.", encoding="utf-8")

    fallback = _article(
        source_path="2024/a.html",
        markdown_text="Second-ranked synthetic content.",
        character_count=20,
    )
    original_winner = _article(
        source_path="2024/b.html",
        markdown_text="Initially winning synthetic content.",
        character_count=30,
    )
    eventual_winner = _article(
        source_path="2024/c.html",
        markdown_text="Stable third synthetic content.",
        character_count=10,
    )
    degraded_fallback = replace(
        fallback,
        markdown_text="x",
        editorial_character_count=1,
    )
    extracted_paths: list[str] = []

    def changed_extract(candidate, _source_root):
        extracted_paths.append(candidate.relative_html_path)
        if candidate.relative_html_path == fallback.relative_html_path:
            return degraded_fallback
        return eventual_winner

    with Catalog.open(config.catalog_db) as catalog:
        run_id = catalog.begin_run("run", "fixture")
        recompute_article(
            catalog,
            config,
            fallback.article_id,
            run_id,
            current_articles=(fallback, original_winner, eventual_winner),
        )
        monkeypatch.setattr("wsj_pipeline.pipeline.extract_article", changed_extract)

        promoted = recompute_article(
            catalog,
            config,
            fallback.article_id,
            run_id,
            current_articles=(),
            invalid_source_paths=(original_winner.relative_html_path,),
        )
        duplicate = catalog.connection.execute(
            """
            SELECT source_html_path, winner_source_html_path,
                   duplicate_rank, reason
            FROM duplicates
            """
        ).fetchone()

    assert promoted is not None
    assert promoted.source_html_path == eventual_winner.relative_html_path
    assert (
        config.output_root / promoted.cleaned_markdown_path
    ).read_text(encoding="utf-8") == eventual_winner.markdown_text
    assert extracted_paths == [
        fallback.relative_html_path,
        eventual_winner.relative_html_path,
    ]
    assert duplicate == (
        fallback.relative_html_path,
        eventual_winner.relative_html_path,
        2,
        "lower_editorial_character_count",
    )
