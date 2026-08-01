from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import UTC, date, datetime
from hashlib import sha256
from pathlib import Path

import duckdb
import pytest

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
    run_pipeline,
    write_markdown_atomic,
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


def test_atomic_markdown_reopens_and_verifies_staged_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "processed"
    staging_root = output_root / "staging"
    target = output_root / "text" / "article.md"
    relative_target = target.relative_to(output_root).as_posix()
    staged = staging_root / "run-1" / (
        f"{sha256(relative_target.encode()).hexdigest()}.md"
    )
    real_open = os.open

    def corrupt_before_reopen(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if Path(path).name == staged.name and not flags & os.O_CREAT:
            staged.write_bytes(b"corrupt staged bytes")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", corrupt_before_reopen)

    with pytest.raises(ValueError, match="staged Markdown verification failed"):
        write_markdown_atomic("expected", target, staging_root, "run-1")

    assert not target.exists()
    assert list(staging_root.rglob("*")) == []


def test_failed_atomic_markdown_replace_preserves_target_and_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "processed"
    staging_root = output_root / "staging"
    target = output_root / "text" / "article.md"
    target.parent.mkdir(parents=True)
    target.write_text("original", encoding="utf-8")

    def fail_replace(
        source: str,
        destination: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
    ) -> None:
        assert source.endswith(".md")
        assert destination == target.name
        assert src_dir_fd != dst_dir_fd
        raise OSError("simulated replacement interruption")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replacement interruption"):
        write_markdown_atomic("replacement", target, staging_root, "run-2")

    assert target.read_text(encoding="utf-8") == "original"
    assert list(staging_root.rglob("*")) == []


def test_atomic_markdown_refuses_targets_outside_output_root(
    tmp_path: Path,
) -> None:
    staging_root = tmp_path / "processed" / "staging"
    outside_target = tmp_path / "outside.md"

    with pytest.raises(ValueError, match="target must be inside output root"):
        write_markdown_atomic(
            "must not escape",
            outside_target,
            staging_root,
            "run-3",
        )

    assert not outside_target.exists()


def test_atomic_markdown_refuses_preexisting_staged_symlink(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "processed"
    staging_root = output_root / "staging"
    run_staging = staging_root / "run-symlink"
    target = output_root / "text" / "article.md"
    target.parent.mkdir(parents=True)
    target.write_text("original target", encoding="utf-8")
    outside = tmp_path / "outside-victim.md"
    outside.write_text("outside original", encoding="utf-8")
    relative_target = target.relative_to(output_root).as_posix()
    staged = run_staging / f"{sha256(relative_target.encode()).hexdigest()}.md"
    run_staging.mkdir(parents=True)
    staged.symlink_to(outside)

    with pytest.raises(FileExistsError):
        write_markdown_atomic(
            "maliciously redirected replacement",
            target,
            staging_root,
            "run-symlink",
        )

    assert target.read_text(encoding="utf-8") == "original target"
    assert not target.is_symlink()
    assert outside.read_text(encoding="utf-8") == "outside original"
    assert list(staging_root.rglob("*")) == []


def test_atomic_markdown_replaces_target_symlink_not_its_referent(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "processed"
    staging_root = output_root / "staging"
    target = output_root / "text" / "article.md"
    victim = output_root / "text" / "existing.md"
    victim.parent.mkdir(parents=True)
    victim.write_text("referent must survive", encoding="utf-8")
    target.symlink_to(victim)

    published_hash = write_markdown_atomic(
        "safe replacement",
        target,
        staging_root,
        "run-target-symlink",
    )

    assert not target.is_symlink()
    assert target.read_text(encoding="utf-8") == "safe replacement"
    assert published_hash == sha256(b"safe replacement").hexdigest()
    assert victim.read_text(encoding="utf-8") == "referent must survive"


def test_atomic_markdown_refuses_staging_root_symlink_escape(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "processed"
    output_root.mkdir()
    external_staging = tmp_path / "external-staging"
    external_staging.mkdir()
    staging_root = output_root / "staging"
    staging_root.symlink_to(external_staging, target_is_directory=True)
    target = output_root / "text" / "article.md"

    with pytest.raises(ValueError, match="staging root must be inside output root"):
        write_markdown_atomic(
            "must not stage externally",
            target,
            staging_root,
            "run-staging-symlink",
        )

    assert not target.exists()
    assert list(external_staging.iterdir()) == []


def test_pipeline_extracts_new_sources_then_skips_unchanged_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "archive"
    write_article_fixture(archive, "2024/one.html", article_id="WP-ONE")
    write_article_fixture(archive, "2024/two.html", article_id="WP-TWO")
    config = PipelineConfig(archive, tmp_path / "processed", batch_size=1)

    first = run_pipeline(config, limit=10, full=False)
    monkeypatch.setattr(
        "wsj_pipeline.pipeline.extract_article",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unchanged source was extracted")
        ),
    )
    second = run_pipeline(config, limit=10, full=False)

    assert first == PipelineSummary(
        discovered=2,
        new=2,
        succeeded=2,
        articles=2,
    )
    assert second == PipelineSummary(
        discovered=2,
        unchanged=2,
        articles=2,
    )
    with duckdb.connect(str(config.catalog_db), read_only=True) as connection:
        seen_run_ids = connection.execute(
            "SELECT DISTINCT last_seen_run_id FROM source_manifest"
        ).fetchall()
        latest_run_id = connection.execute(
            "SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()[0]
    assert seen_run_ids == [(latest_run_id,)]


def test_completed_full_run_removes_unseen_unique_article_and_markdown(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    removed_html = write_article_fixture(
        archive,
        "2024/a.html",
        article_id="WP-REMOVED",
    )
    write_article_fixture(
        archive,
        "2024/b.html",
        article_id="WP-RETAINED",
    )
    config = PipelineConfig(archive, tmp_path / "processed")
    run_pipeline(config, limit=None, full=True)
    removed_markdown = config.output_root / article_markdown_path(
        "wsj:WP-REMOVED",
        date(2024, 1, 2),
    )
    removed_html.unlink()
    removed_html.with_name("a_main_image.jpg").unlink()

    summary = run_pipeline(config, limit=None, full=True)

    assert summary.removed == 1
    assert summary.articles == 1
    assert not removed_markdown.exists()
    with duckdb.connect(str(config.catalog_db), read_only=True) as connection:
        articles = connection.execute(
            "SELECT article_id FROM articles ORDER BY article_id"
        ).fetchall()
        manifests = connection.execute(
            "SELECT source_html_path, status FROM source_manifest "
            "ORDER BY source_html_path"
        ).fetchall()
        candidates = connection.execute(
            "SELECT source_html_path FROM candidates ORDER BY source_html_path"
        ).fetchall()
    assert articles == [("wsj:WP-RETAINED",)]
    assert manifests == [
        ("2024/a.html", "missing"),
        ("2024/b.html", "succeeded"),
    ]
    assert candidates == [("2024/b.html",)]


def test_limited_run_does_not_reconcile_unseen_source(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    write_article_fixture(archive, "2024/a.html", article_id="WP-SEEN")
    unseen_html = write_article_fixture(
        archive,
        "2024/b.html",
        article_id="WP-UNSEEN",
    )
    config = PipelineConfig(archive, tmp_path / "processed")
    run_pipeline(config, limit=10, full=False)
    unseen_markdown = config.output_root / article_markdown_path(
        "wsj:WP-UNSEEN",
        date(2024, 1, 2),
    )
    unseen_html.unlink()
    unseen_html.with_name("b_main_image.jpg").unlink()

    summary = run_pipeline(config, limit=1, full=False)

    assert summary.removed == 0
    assert summary.articles == 2
    assert unseen_markdown.is_file()
    with duckdb.connect(str(config.catalog_db), read_only=True) as connection:
        manifest = connection.execute(
            "SELECT status FROM source_manifest WHERE source_html_path = '2024/b.html'"
        ).fetchone()
        article = connection.execute(
            "SELECT source_html_path FROM articles WHERE article_id = 'wsj:WP-UNSEEN'"
        ).fetchone()
    assert manifest == ("succeeded",)
    assert article == ("2024/b.html",)


def test_completed_full_run_promotes_reextracted_surviving_duplicate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "archive"
    write_article_fixture(
        archive,
        "2024/a.html",
        article_id="WP-FULL-PROMOTION",
        paragraphs=("Short surviving body.",),
    )
    removed_html = write_article_fixture(
        archive,
        "2024/b.html",
        article_id="WP-FULL-PROMOTION",
        paragraphs=("This removed winner originally had much more content.",),
    )
    config = PipelineConfig(archive, tmp_path / "processed")
    run_pipeline(config, limit=None, full=True)
    markdown_path = config.output_root / article_markdown_path(
        "wsj:WP-FULL-PROMOTION",
        date(2024, 1, 2),
    )
    removed_html.unlink()
    removed_html.with_name("b_main_image.jpg").unlink()
    real_extract = extract_article
    extracted_paths: list[str] = []

    def recording_extract(candidate, source_root):
        extracted_paths.append(candidate.relative_html_path)
        return real_extract(candidate, source_root)

    monkeypatch.setattr("wsj_pipeline.pipeline.extract_article", recording_extract)

    summary = run_pipeline(config, limit=None, full=True)

    assert summary.removed == 1
    assert summary.articles == 1
    assert extracted_paths == ["2024/a.html"]
    assert "Short surviving body." in markdown_path.read_text(encoding="utf-8")
    with duckdb.connect(str(config.catalog_db), read_only=True) as connection:
        winner = connection.execute(
            "SELECT source_html_path, cleaned_markdown_path FROM articles"
        ).fetchone()
        missing = connection.execute(
            "SELECT status FROM source_manifest WHERE source_html_path = '2024/b.html'"
        ).fetchone()
        candidates = connection.execute(
            "SELECT source_html_path FROM candidates ORDER BY source_html_path"
        ).fetchall()
        duplicates = connection.execute("SELECT * FROM duplicates").fetchall()
    assert winner == (
        "2024/a.html",
        markdown_path.relative_to(config.output_root).as_posix(),
    )
    assert missing == ("missing",)
    assert candidates == [("2024/a.html",)]
    assert duplicates == []


def test_failed_current_winner_promotes_valid_duplicate(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    write_article_fixture(
        archive,
        "2024/a.html",
        article_id="WP-FAILED-PROMOTION",
        paragraphs=("Short valid fallback.",),
    )
    winner_html = write_article_fixture(
        archive,
        "2024/b.html",
        article_id="WP-FAILED-PROMOTION",
        paragraphs=("This initial winner has substantially more editorial content.",),
    )
    config = PipelineConfig(archive, tmp_path / "processed")
    run_pipeline(config, limit=10, full=False)
    write_article_fixture(
        archive,
        "2024/b.html",
        article_id="WP-FAILED-PROMOTION",
        published=None,
    )

    summary = run_pipeline(config, limit=10, full=False)

    assert summary.failed == 1
    assert summary.articles == 1
    with duckdb.connect(str(config.catalog_db), read_only=True) as connection:
        winner = connection.execute(
            "SELECT source_html_path FROM articles"
        ).fetchone()
        failed = connection.execute(
            "SELECT status FROM source_manifest WHERE source_html_path = ?",
            [winner_html.relative_to(archive).as_posix()],
        ).fetchone()
        candidates = connection.execute(
            "SELECT source_html_path FROM candidates ORDER BY source_html_path"
        ).fetchall()
    assert winner == ("2024/a.html",)
    assert failed == ("failed",)
    assert candidates == [("2024/a.html",)]


def test_failed_winner_and_malformed_stored_fallback_do_not_fail_run(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    for relative_path in ("2024/a.html", "2024/b.html"):
        write_article_fixture(
            archive,
            relative_path,
            article_id="WP-FAILED-FALLBACK",
            paragraphs=("Identically ranked synthetic content.",),
        )
    config = PipelineConfig(archive, tmp_path / "processed")
    run_pipeline(config, limit=10, full=False)
    markdown_path = config.output_root / article_markdown_path(
        "wsj:WP-FAILED-FALLBACK",
        date(2024, 1, 2),
    )
    for relative_path in ("2024/a.html", "2024/b.html"):
        write_article_fixture(
            archive,
            relative_path,
            article_id="WP-FAILED-FALLBACK",
            published=None,
        )

    summary = run_pipeline(config, limit=10, full=False)

    assert summary.failed == 2
    assert summary.removed == 0
    assert summary.articles == 0
    assert not markdown_path.exists()
    with duckdb.connect(str(config.catalog_db), read_only=True) as connection:
        run_status = connection.execute(
            "SELECT status FROM runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()[0]
        manifests = connection.execute(
            "SELECT source_html_path, status FROM source_manifest "
            "ORDER BY source_html_path"
        ).fetchall()
        failures = connection.execute(
            "SELECT source_html_path, error_code FROM failures "
            "WHERE run_id = (SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1) "
            "ORDER BY source_html_path"
        ).fetchall()
        candidate_count = connection.execute(
            "SELECT count(*) FROM candidates"
        ).fetchone()[0]
    assert run_status == "succeeded"
    assert manifests == [
        ("2024/a.html", "failed"),
        ("2024/b.html", "failed"),
    ]
    assert failures == [
        ("2024/a.html", "missing_publication_timestamp"),
        ("2024/b.html", "missing_publication_timestamp"),
    ]
    assert candidate_count == 0


def test_failed_only_candidate_removes_article_and_markdown(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    html_path = write_article_fixture(
        archive,
        "2024/one.html",
        article_id="WP-FAILED-REMOVAL",
    )
    config = PipelineConfig(archive, tmp_path / "processed")
    run_pipeline(config, limit=10, full=False)
    markdown_path = config.output_root / article_markdown_path(
        "wsj:WP-FAILED-REMOVAL",
        date(2024, 1, 2),
    )
    write_article_fixture(
        archive,
        html_path.relative_to(archive).as_posix(),
        article_id="WP-FAILED-REMOVAL",
        published=None,
    )

    summary = run_pipeline(config, limit=10, full=False)

    assert summary.failed == 1
    assert summary.articles == 0
    assert not markdown_path.exists()
    with duckdb.connect(str(config.catalog_db), read_only=True) as connection:
        manifest_status = connection.execute(
            "SELECT status FROM source_manifest"
        ).fetchone()
        candidate_count = connection.execute(
            "SELECT count(*) FROM candidates"
        ).fetchone()[0]
        article_count = connection.execute(
            "SELECT count(*) FROM articles"
        ).fetchone()[0]
    assert manifest_status == ("failed",)
    assert candidate_count == 0
    assert article_count == 0


def test_unchanged_failed_winner_retained_by_prior_run_is_removed(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    write_article_fixture(
        archive,
        "2024/one.html",
        article_id="WP-DEFERRED-FAILED-WINNER",
    )
    config = PipelineConfig(archive, tmp_path / "processed")
    run_pipeline(config, limit=10, full=False)
    markdown_path = config.output_root / article_markdown_path(
        "wsj:WP-DEFERRED-FAILED-WINNER",
        date(2024, 1, 2),
    )
    with duckdb.connect(str(config.catalog_db)) as connection:
        connection.execute("UPDATE source_manifest SET status = 'failed'")

    summary = run_pipeline(config, limit=10, full=False)

    assert summary.unchanged == 1
    assert summary.articles == 0
    assert not markdown_path.exists()
    with duckdb.connect(str(config.catalog_db), read_only=True) as connection:
        candidate_count = connection.execute(
            "SELECT count(*) FROM candidates"
        ).fetchone()[0]
        manifest_status = connection.execute(
            "SELECT status FROM source_manifest"
        ).fetchone()
    assert candidate_count == 0
    assert manifest_status == ("failed",)


def test_mtime_only_change_updates_manifest_without_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "archive"
    html_path = write_article_fixture(
        archive,
        "2024/one.html",
        article_id="WP-MTIME",
    )
    config = PipelineConfig(archive, tmp_path / "processed")
    run_pipeline(config, limit=10, full=False)
    markdown_path = next(config.text_root.rglob("*.md"))
    original_markdown_hash = sha256(markdown_path.read_bytes()).hexdigest()
    original = html_path.stat()
    os.utime(
        html_path,
        ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000),
    )
    monkeypatch.setattr(
        "wsj_pipeline.pipeline.extract_article",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("mtime-only source was extracted")
        ),
    )

    summary = run_pipeline(config, limit=10, full=False)

    assert summary.metadata_only == 1
    assert summary.succeeded == 0
    assert sha256(markdown_path.read_bytes()).hexdigest() == original_markdown_hash
    with duckdb.connect(str(config.catalog_db), read_only=True) as connection:
        stored_mtime = connection.execute(
            "SELECT html_mtime_ns FROM source_manifest"
        ).fetchone()[0]
    assert stored_mtime == html_path.stat().st_mtime_ns


def test_html_change_reextracts_and_republishes_markdown(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    html_path = write_article_fixture(
        archive,
        "2024/one.html",
        article_id="WP-CHANGED",
    )
    config = PipelineConfig(archive, tmp_path / "processed")
    run_pipeline(config, limit=10, full=False)
    html_path.write_text(
        html_path.read_text(encoding="utf-8").replace(
            "First paragraph.",
            "A materially changed first paragraph.",
        ),
        encoding="utf-8",
    )

    summary = run_pipeline(config, limit=10, full=False)

    assert summary.changed == 1
    assert summary.succeeded == 1
    assert "materially changed" in next(config.text_root.rglob("*.md")).read_text(
        encoding="utf-8"
    )


def test_header_image_only_change_reextracts_candidate(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    html_path = write_article_fixture(
        archive,
        "2024/one.html",
        article_id="WP-IMAGE",
    )
    config = PipelineConfig(archive, tmp_path / "processed")
    run_pipeline(config, limit=10, full=False)
    image_path = html_path.with_name("one_main_image.jpg")
    image_path.write_bytes(b"materially-different-synthetic-image")

    summary = run_pipeline(config, limit=10, full=False)

    assert summary.changed == 1
    assert summary.succeeded == 1
    with duckdb.connect(str(config.catalog_db), read_only=True) as connection:
        stored_size = connection.execute(
            "SELECT header_image_size FROM source_manifest"
        ).fetchone()[0]
    assert stored_size == image_path.stat().st_size


def test_missing_header_image_is_a_successful_candidate_warning(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    write_article_fixture(
        archive,
        "2024/one.html",
        article_id="WP-NO-IMAGE",
        with_image=False,
    )
    config = PipelineConfig(archive, tmp_path / "processed")

    summary = run_pipeline(config, limit=10, full=False)

    assert summary.succeeded == 1
    assert summary.failed == 0
    with duckdb.connect(str(config.catalog_db), read_only=True) as connection:
        header_image_path, warnings = connection.execute(
            "SELECT header_image_path, warnings FROM candidates"
        ).fetchone()
    assert header_image_path is None
    assert warnings == ["missing_image"]


def test_extraction_failure_is_isolated_and_run_finishes(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    write_article_fixture(
        archive,
        "2024/a-malformed.html",
        article_id="WP-BAD",
        published=None,
    )
    write_article_fixture(
        archive,
        "2024/b-valid.html",
        article_id="WP-GOOD",
    )
    config = PipelineConfig(archive, tmp_path / "processed", batch_size=2)

    summary = run_pipeline(config, limit=10, full=False)

    assert summary.failed == 1
    assert summary.succeeded == 1
    with duckdb.connect(str(config.catalog_db), read_only=True) as connection:
        failure = connection.execute(
            "SELECT source_html_path, error_code, message FROM failures"
        ).fetchone()
        manifests = connection.execute(
            "SELECT source_html_path, status FROM source_manifest "
            "ORDER BY source_html_path"
        ).fetchall()
        run_status = connection.execute(
            "SELECT status FROM runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()[0]
    assert failure == (
        "2024/a-malformed.html",
        "missing_publication_timestamp",
        "article has no publication timestamp",
    )
    assert manifests == [
        ("2024/a-malformed.html", "failed"),
        ("2024/b-valid.html", "succeeded"),
    ]
    assert run_status == "succeeded"


def test_unchanged_failed_source_is_marked_seen_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "archive"
    write_article_fixture(
        archive,
        "2024/malformed.html",
        article_id="WP-PERMANENT-FAILURE",
        published=None,
    )
    config = PipelineConfig(archive, tmp_path / "processed")
    first = run_pipeline(config, limit=10, full=False)
    monkeypatch.setattr(
        "wsj_pipeline.pipeline.extract_article",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unchanged failed source was retried")
        ),
    )

    second = run_pipeline(config, limit=10, full=False)

    assert first.new == 1
    assert first.failed == 1
    assert second.unchanged == 1
    assert second.failed == 0
    with duckdb.connect(str(config.catalog_db), read_only=True) as connection:
        status, seen_run_id = connection.execute(
            "SELECT status, last_seen_run_id FROM source_manifest"
        ).fetchone()
        second_run_id = connection.execute(
            "SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()[0]
    assert status == "failed"
    assert seen_run_id == second_run_id


def test_failed_source_stat_change_retries_without_metadata_only_success(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    html_path = write_article_fixture(
        archive,
        "2024/malformed.html",
        article_id="WP-RETRY-FAILURE",
        published=None,
    )
    config = PipelineConfig(archive, tmp_path / "processed")
    run_pipeline(config, limit=10, full=False)
    original = html_path.stat()
    os.utime(
        html_path,
        ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000),
    )

    retried = run_pipeline(config, limit=10, full=False)

    assert retried.changed == 1
    assert retried.metadata_only == 0
    assert retried.failed == 1
    with duckdb.connect(str(config.catalog_db), read_only=True) as connection:
        status = connection.execute(
            "SELECT status FROM source_manifest"
        ).fetchone()[0]
    assert status == "failed"


def test_stale_failed_source_requires_reprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "archive"
    write_article_fixture(
        archive,
        "2024/malformed.html",
        article_id="WP-STALE-FAILURE",
        published=None,
    )
    config = PipelineConfig(archive, tmp_path / "processed")
    run_pipeline(config, limit=10, full=False)
    with duckdb.connect(str(config.catalog_db)) as connection:
        connection.execute(
            "UPDATE source_manifest SET extractor_version = '1'"
        )
    real_extract = extract_article
    monkeypatch.setattr(
        "wsj_pipeline.pipeline.extract_article",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("stale failed source retried without --reprocess")
        ),
    )

    skipped = run_pipeline(config, limit=10, full=False)

    assert skipped.stale == 1
    assert skipped.failed == 0
    monkeypatch.setattr("wsj_pipeline.pipeline.extract_article", real_extract)
    reprocessed = run_pipeline(
        config,
        limit=10,
        full=False,
        reprocess=True,
    )
    assert reprocessed.reprocessed == 1
    assert reprocessed.failed == 1


def test_stale_extractor_is_marked_seen_and_requires_reprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "archive"
    write_article_fixture(archive, "2024/one.html", article_id="WP-STALE")
    config = PipelineConfig(archive, tmp_path / "processed")
    run_pipeline(config, limit=10, full=False)
    with duckdb.connect(str(config.catalog_db)) as connection:
        connection.execute(
            "UPDATE source_manifest SET extractor_version = '1'"
        )
    real_extract = extract_article
    monkeypatch.setattr(
        "wsj_pipeline.pipeline.extract_article",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("stale source was extracted without --reprocess")
        ),
    )

    skipped = run_pipeline(config, limit=10, full=False)

    with duckdb.connect(str(config.catalog_db), read_only=True) as connection:
        stored_version, seen_run_id = connection.execute(
            "SELECT extractor_version, last_seen_run_id FROM source_manifest"
        ).fetchone()
        skipped_run_id = connection.execute(
            "SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()[0]
    assert skipped.stale == 1
    assert skipped.succeeded == 0
    assert stored_version == "1"
    assert seen_run_id == skipped_run_id

    monkeypatch.setattr("wsj_pipeline.pipeline.extract_article", real_extract)
    reprocessed = run_pipeline(
        config,
        limit=10,
        full=False,
        reprocess=True,
    )

    assert reprocessed.reprocessed == 1
    assert reprocessed.succeeded == 1


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


def test_public_recompute_removes_article_and_generated_file_when_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, articles = _duplicate_articles(tmp_path)
    only_candidate = articles[0]
    real_transaction = Catalog.transaction
    transaction_observed = False

    with Catalog.open(config.catalog_db) as catalog:
        run_id = catalog.begin_run("run", "fixture")
        original = recompute_article(
            catalog,
            config,
            only_candidate.article_id,
            run_id,
            current_articles=(only_candidate,),
        )
        assert original is not None
        markdown_path = config.output_root / original.cleaned_markdown_path
        catalog.remove_candidate(only_candidate.relative_html_path)

        @contextmanager
        def observing_transaction(current_catalog: Catalog):
            nonlocal transaction_observed
            with real_transaction(current_catalog):
                yield
                transaction_observed = True
                assert current_catalog.article_for_id(only_candidate.article_id) is None
                assert markdown_path.is_file()

        monkeypatch.setattr(Catalog, "transaction", observing_transaction)
        removed = recompute_article(
            catalog,
            config,
            only_candidate.article_id,
            run_id,
            current_articles=(),
        )

        assert removed is None
        assert catalog.article_for_id(only_candidate.article_id) is None
    assert transaction_observed
    assert not markdown_path.exists()


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
