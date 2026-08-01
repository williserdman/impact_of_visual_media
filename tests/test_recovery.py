from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import date
from hashlib import sha256
from pathlib import Path

import duckdb
import pytest

import wsj_pipeline.discovery as discovery_module
from tests.fixtures import write_article_fixture
from wsj_pipeline.catalog import Catalog
from wsj_pipeline.config import PipelineConfig
from wsj_pipeline.pipeline import (
    PipelineLockedError,
    article_markdown_path,
    pipeline_lock,
    run_pipeline,
    run_validated_pipeline,
)


def _build_reconciliation_retry_fixture(tmp_path: Path) -> PipelineConfig:
    archive = tmp_path / "archive"
    write_article_fixture(
        archive,
        "2024/a-unique.html",
        article_id="WP-RETRY-UNIQUE",
    )
    write_article_fixture(
        archive,
        "2024/b-duplicate.html",
        article_id="WP-RETRY-DUPLICATE",
        paragraphs=("Longer synthetic duplicate winner body.",),
    )
    write_article_fixture(
        archive,
        "2024/c-duplicate.html",
        article_id="WP-RETRY-DUPLICATE",
        paragraphs=("Short body.",),
    )
    config = PipelineConfig(archive, tmp_path / "processed", batch_size=1)
    initial = run_validated_pipeline(config, limit=None, full=True)
    assert initial.articles == 2
    assert initial.duplicate_count == 1
    assert initial.validation_ok
    shutil.rmtree(archive)
    archive.mkdir()
    return config


def _assert_reconciliation_retry_complete(config: PipelineConfig) -> None:
    assert list(config.text_root.rglob("*.md")) == []
    with duckdb.connect(str(config.catalog_db), read_only=True) as connection:
        assert connection.execute(
            "SELECT status FROM source_manifest ORDER BY source_html_path"
        ).fetchall() == [("missing",), ("missing",), ("missing",)]
        assert connection.execute("SELECT count(*) FROM candidates").fetchone() == (
            0,
        )
        assert connection.execute("SELECT count(*) FROM articles").fetchone() == (
            0,
        )
        assert connection.execute("SELECT count(*) FROM duplicates").fetchone() == (
            0,
        )


def test_pipeline_lock_is_exclusive_and_owned_lock_is_cleaned(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "processed" / "pipeline.lock"

    with pipeline_lock(lock_path):
        assert lock_path.is_file()
        with (
            pytest.raises(PipelineLockedError, match="already running"),
            pipeline_lock(lock_path),
        ):
            raise AssertionError("unreachable")
        assert lock_path.is_file()

    assert not lock_path.exists()


def test_batches_use_one_catalog_transaction_each(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "archive"
    for number in range(3):
        write_article_fixture(
            archive,
            f"2024/{number}.html",
            article_id=f"WP-BATCH-{number}",
        )
    config = PipelineConfig(archive, tmp_path / "processed", batch_size=2)
    real_transaction = Catalog.transaction
    transaction_count = 0

    @contextmanager
    def counting_transaction(catalog: Catalog) -> Iterator[None]:
        nonlocal transaction_count
        transaction_count += 1
        with real_transaction(catalog):
            yield

    monkeypatch.setattr(Catalog, "transaction", counting_transaction)

    summary = run_pipeline(config, limit=10, full=False)

    assert summary.succeeded == 3
    assert transaction_count == 2


def test_full_reconciliation_processes_missing_sources_in_bounded_pages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "archive"
    for number in range(5):
        write_article_fixture(
            archive,
            f"2024/{number}.html",
            article_id=f"WP-REMOVED-PAGE-{number}",
        )
    config = PipelineConfig(archive, tmp_path / "processed", batch_size=2)
    run_pipeline(config, limit=None, full=True)
    shutil.rmtree(archive)
    archive.mkdir()
    real_reconcile = Catalog.reconcile_missing
    real_affected_ids = Catalog.affected_missing_article_ids
    page_sizes: list[int] = []
    identity_page_sizes: list[int] = []

    def recording_reconcile(
        catalog: Catalog,
        run_id: str,
        *,
        batch_size: int,
    ):
        missing_count = real_reconcile(
            catalog,
            run_id,
            batch_size=batch_size,
        )
        page_sizes.append(missing_count)
        return missing_count

    def recording_affected_ids(
        catalog: Catalog,
        run_id: str,
        *,
        after_article_id: str | None,
        batch_size: int,
    ) -> tuple[str, ...]:
        affected_ids = real_affected_ids(
            catalog,
            run_id,
            after_article_id=after_article_id,
            batch_size=batch_size,
        )
        identity_page_sizes.append(len(affected_ids))
        return affected_ids

    monkeypatch.setattr(Catalog, "reconcile_missing", recording_reconcile)
    monkeypatch.setattr(
        Catalog,
        "affected_missing_article_ids",
        recording_affected_ids,
    )

    summary = run_pipeline(config, limit=None, full=True)

    assert summary.removed == 5
    assert summary.articles == 0
    assert len([size for size in page_sizes if size]) == 3
    assert all(size <= config.batch_size for size in page_sizes)
    assert len([size for size in identity_page_sizes if size]) == 3
    assert all(size <= config.batch_size for size in identity_page_sizes)


def test_batched_reconciliation_removes_duplicate_identity_only_after_inventory(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    write_article_fixture(
        archive,
        "2024/a.html",
        article_id="WP-REMOVED-DUPLICATES",
        paragraphs=("Longer synthetic winner body for removal.",),
    )
    write_article_fixture(
        archive,
        "2024/b.html",
        article_id="WP-REMOVED-DUPLICATES",
        paragraphs=("Short body.",),
    )
    config = PipelineConfig(archive, tmp_path / "processed", batch_size=1)
    run_pipeline(config, limit=None, full=True)
    shutil.rmtree(archive)
    archive.mkdir()

    summary = run_pipeline(config, limit=None, full=True)

    assert summary.removed == 2
    assert summary.articles == 0
    with duckdb.connect(str(config.catalog_db), read_only=True) as connection:
        assert connection.execute(
            "SELECT status FROM source_manifest ORDER BY source_html_path"
        ).fetchall() == [("missing",), ("missing",)]
        assert connection.execute("SELECT count(*) FROM candidates").fetchone() == (
            0,
        )


def test_full_retry_resumes_after_first_missing_source_page_commits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_reconciliation_retry_fixture(tmp_path)
    real_reconcile = Catalog.reconcile_missing
    calls = 0

    def interrupt_after_first_page(
        catalog: Catalog,
        run_id: str,
        *,
        batch_size: int,
    ) -> int:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated failure after phase-one commit")
        return real_reconcile(catalog, run_id, batch_size=batch_size)

    monkeypatch.setattr(Catalog, "reconcile_missing", interrupt_after_first_page)

    with pytest.raises(RuntimeError, match="after phase-one commit"):
        run_pipeline(config, limit=None, full=True)

    with duckdb.connect(str(config.catalog_db), read_only=True) as connection:
        assert connection.execute(
            "SELECT status FROM source_manifest ORDER BY source_html_path"
        ).fetchall() == [("missing",), ("succeeded",), ("succeeded",)]
        assert connection.execute("SELECT count(*) FROM candidates").fetchone() == (
            2,
        )
        assert connection.execute("SELECT count(*) FROM articles").fetchone() == (
            2,
        )
    monkeypatch.setattr(Catalog, "reconcile_missing", real_reconcile)

    retry = run_validated_pipeline(config, limit=None, full=True)

    assert retry.removed == 2
    assert retry.articles == 0
    assert retry.duplicate_count == 0
    assert retry.validation_ok
    _assert_reconciliation_retry_complete(config)


@pytest.mark.parametrize(
    ("failure_point", "failed_articles", "failed_duplicates"),
    [
        ("between-phases", 2, 1),
        ("mid-phase-two", 1, 0),
    ],
)
def test_full_retry_resumes_pending_identity_pages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
    failed_articles: int,
    failed_duplicates: int,
) -> None:
    config = _build_reconciliation_retry_fixture(tmp_path)
    real_affected_ids = Catalog.affected_missing_article_ids
    calls = 0

    def interrupt_identity_pages(
        catalog: Catalog,
        run_id: str,
        *,
        after_article_id: str | None,
        batch_size: int,
    ) -> tuple[str, ...]:
        nonlocal calls
        calls += 1
        if failure_point == "between-phases" or calls == 2:
            raise RuntimeError(f"simulated {failure_point} failure")
        return real_affected_ids(
            catalog,
            run_id,
            after_article_id=after_article_id,
            batch_size=batch_size,
        )

    monkeypatch.setattr(
        Catalog,
        "affected_missing_article_ids",
        interrupt_identity_pages,
    )

    with pytest.raises(RuntimeError, match=failure_point):
        run_pipeline(config, limit=None, full=True)

    with duckdb.connect(str(config.catalog_db), read_only=True) as connection:
        assert connection.execute("SELECT count(*) FROM candidates").fetchone() == (
            0,
        )
        assert connection.execute("SELECT count(*) FROM articles").fetchone() == (
            failed_articles,
        )
        assert connection.execute("SELECT count(*) FROM duplicates").fetchone() == (
            failed_duplicates,
        )
    assert len(list(config.text_root.rglob("*.md"))) == failed_articles
    monkeypatch.setattr(
        Catalog,
        "affected_missing_article_ids",
        real_affected_ids,
    )

    retry = run_validated_pipeline(config, limit=None, full=True)

    assert retry.removed == 0
    assert retry.articles == 0
    assert retry.duplicate_count == 0
    assert retry.validation_ok
    _assert_reconciliation_retry_complete(config)


def test_file_before_catalog_interruption_is_repaired_on_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "archive"
    write_article_fixture(archive, "2024/one.html", article_id="WP-RECOVER")
    config = PipelineConfig(archive, tmp_path / "processed")
    real_transaction = Catalog.transaction
    fail_commit = True

    @contextmanager
    def interrupted_transaction(catalog: Catalog) -> Iterator[None]:
        nonlocal fail_commit
        catalog.connection.execute("BEGIN TRANSACTION")
        try:
            yield
            if fail_commit:
                fail_commit = False
                catalog.connection.execute("ROLLBACK")
                raise RuntimeError("simulated catalog commit interruption")
            catalog.connection.execute("COMMIT")
        except BaseException:
            with suppress(duckdb.TransactionException):
                catalog.connection.execute("ROLLBACK")
            raise

    monkeypatch.setattr(Catalog, "transaction", interrupted_transaction)

    with pytest.raises(RuntimeError, match="catalog commit interruption"):
        run_pipeline(config, limit=10, full=False)

    published = next(config.text_root.rglob("*.md"))
    with duckdb.connect(str(config.catalog_db), read_only=True) as connection:
        assert connection.execute(
            "SELECT count(*) FROM source_manifest"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT status FROM runs"
        ).fetchone()[0] == "failed"

    monkeypatch.setattr(Catalog, "transaction", real_transaction)
    repaired = run_pipeline(config, limit=10, full=False)

    with duckdb.connect(str(config.catalog_db), read_only=True) as connection:
        stored_hash = connection.execute(
            "SELECT cleaned_markdown_sha256 FROM articles"
        ).fetchone()[0]
        manifest_count = connection.execute(
            "SELECT count(*) FROM source_manifest"
        ).fetchone()[0]
    assert repaired.new == 1
    assert manifest_count == 1
    assert stored_hash == sha256(published.read_bytes()).hexdigest()


def test_interrupted_full_inventory_does_not_reconcile_unseen_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "archive"
    write_article_fixture(archive, "2024/a.html", article_id="WP-FIRST")
    removed_html = write_article_fixture(
        archive,
        "2024/b.html",
        article_id="WP-INTERRUPTED-UNSEEN",
    )
    config = PipelineConfig(archive, tmp_path / "processed", batch_size=1)
    run_pipeline(config, limit=None, full=True)
    removed_markdown = config.output_root / article_markdown_path(
        "wsj:WP-INTERRUPTED-UNSEEN",
        date(2024, 1, 2),
    )
    removed_html.unlink()
    removed_html.with_name("b_main_image.jpg").unlink()
    from wsj_pipeline.discovery import discover_sources as real_discover

    def interrupted_discovery(source_root: Path, limit: int | None = None):
        yield from real_discover(source_root, limit=limit)
        raise RuntimeError("simulated interrupted full inventory")

    monkeypatch.setattr(
        "wsj_pipeline.pipeline.discover_sources",
        interrupted_discovery,
    )

    with pytest.raises(RuntimeError, match="interrupted full inventory"):
        run_pipeline(config, limit=None, full=True)

    assert removed_markdown.is_file()
    with duckdb.connect(str(config.catalog_db), read_only=True) as connection:
        manifest_status = connection.execute(
            "SELECT status FROM source_manifest WHERE source_html_path = '2024/b.html'"
        ).fetchone()
        article = connection.execute(
            "SELECT source_html_path FROM articles "
            "WHERE article_id = 'wsj:WP-INTERRUPTED-UNSEEN'"
        ).fetchone()
        run_status = connection.execute(
            "SELECT status FROM runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()[0]
    assert manifest_status == ("succeeded",)
    assert article == ("2024/b.html",)
    assert run_status == "failed"


@pytest.mark.parametrize("source_failure", ["missing", "regular-file"])
def test_full_run_rejects_unavailable_source_before_mutating_existing_outputs(
    tmp_path: Path,
    source_failure: str,
) -> None:
    archive = tmp_path / "archive"
    write_article_fixture(
        archive,
        "2024/retained.html",
        article_id="WP-SOURCE-PREFLIGHT",
    )
    config = PipelineConfig(archive, tmp_path / "processed")
    run_pipeline(config, limit=None, full=True)
    markdown_path = config.output_root / article_markdown_path(
        "wsj:WP-SOURCE-PREFLIGHT",
        date(2024, 1, 2),
    )
    markdown_bytes = markdown_path.read_bytes()
    with duckdb.connect(str(config.catalog_db), read_only=True) as connection:
        run_count = connection.execute("SELECT count(*) FROM runs").fetchone()[0]

    if source_failure == "missing":
        archive.rename(tmp_path / "detached-archive")
    else:
        shutil.rmtree(archive)
        archive.write_text("not a directory", encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match="source root must be an accessible real directory",
    ):
        run_pipeline(config, limit=None, full=True)

    assert markdown_path.read_bytes() == markdown_bytes
    assert not config.lock_path.exists()
    with duckdb.connect(str(config.catalog_db), read_only=True) as connection:
        assert connection.execute("SELECT count(*) FROM runs").fetchone()[0] == (
            run_count
        )
        assert connection.execute(
            "SELECT status FROM source_manifest"
        ).fetchone() == ("succeeded",)
        assert connection.execute("SELECT count(*) FROM articles").fetchone() == (1,)


def test_full_run_does_not_reconcile_after_directory_traversal_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "archive"
    write_article_fixture(archive, "2024/a.html", article_id="WP-TRAVERSE-A")
    removed_html = write_article_fixture(
        archive,
        "2024/b.html",
        article_id="WP-TRAVERSE-B",
    )
    config = PipelineConfig(archive, tmp_path / "processed")
    run_pipeline(config, limit=None, full=True)
    removed_html.unlink()
    removed_html.with_name("b_main_image.jpg").unlink()
    real_scandir = discovery_module.os.scandir

    def denied_scandir(path):
        if Path(path) == archive / "2024":
            raise PermissionError("simulated traversal denial")
        return real_scandir(path)

    monkeypatch.setattr(discovery_module.os, "scandir", denied_scandir)

    with pytest.raises(PermissionError, match="simulated traversal denial"):
        run_pipeline(config, limit=None, full=True)

    with duckdb.connect(str(config.catalog_db), read_only=True) as connection:
        assert connection.execute(
            "SELECT status FROM source_manifest "
            "WHERE source_html_path = '2024/b.html'"
        ).fetchone() == ("succeeded",)
        assert connection.execute(
            "SELECT source_html_path FROM articles "
            "WHERE article_id = 'wsj:WP-TRAVERSE-B'"
        ).fetchone() == ("2024/b.html",)
        assert connection.execute(
            "SELECT status FROM runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone() == ("failed",)


@pytest.mark.parametrize(
    "legacy_marker",
    [
        "state/pipeline.duckdb",
        "parquet",
        "index/wsj.duckdb",
    ],
)
def test_legacy_output_is_refused_without_deletion(
    tmp_path: Path,
    legacy_marker: str,
) -> None:
    archive = tmp_path / "archive"
    archive.mkdir()
    config = PipelineConfig(archive, tmp_path / "processed")
    marker = config.output_root / legacy_marker
    if marker.suffix:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_bytes(b"legacy-derived-data")
    else:
        marker.mkdir(parents=True)
        (marker / "keep.txt").write_text("legacy-derived-data", encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match="move derived output aside or choose a fresh output root",
    ):
        run_pipeline(config, limit=10, full=False)

    assert marker.exists()
    assert not config.catalog_db.exists()
    if marker.is_dir():
        assert (marker / "keep.txt").read_text(encoding="utf-8") == (
            "legacy-derived-data"
        )
    else:
        assert marker.read_bytes() == b"legacy-derived-data"


def test_changed_date_moves_markdown_and_deletes_only_old_generated_file(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    source_path = "2024/one.html"
    write_article_fixture(
        archive,
        source_path,
        article_id="WP-DATE-MOVE",
        published="2024-01-02T15:30:00Z",
    )
    config = PipelineConfig(archive, tmp_path / "processed")
    run_pipeline(config, limit=10, full=False)
    old_path = config.output_root / article_markdown_path(
        "wsj:WP-DATE-MOVE",
        date(2024, 1, 2),
    )
    neighbor = old_path.with_name("not-generated-by-this-article.md")
    neighbor.write_text("keep", encoding="utf-8")
    write_article_fixture(
        archive,
        source_path,
        article_id="WP-DATE-MOVE",
        published="2024-02-03T15:30:00Z",
        paragraphs=("Changed date and content.",),
    )

    summary = run_pipeline(config, limit=10, full=False)

    new_path = config.output_root / article_markdown_path(
        "wsj:WP-DATE-MOVE",
        date(2024, 2, 3),
    )
    assert summary.changed == 1
    assert new_path.is_file()
    assert not old_path.exists()
    assert neighbor.read_text(encoding="utf-8") == "keep"


def test_changed_identity_releases_old_article_before_publishing_new_one(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    source_path = "2024/one.html"
    write_article_fixture(
        archive,
        source_path,
        article_id="WP-OLD-IDENTITY",
    )
    config = PipelineConfig(archive, tmp_path / "processed")
    run_pipeline(config, limit=10, full=False)
    old_path = config.output_root / article_markdown_path(
        "wsj:WP-OLD-IDENTITY",
        date(2024, 1, 2),
    )
    write_article_fixture(
        archive,
        source_path,
        article_id="WP-NEW-IDENTITY",
        paragraphs=("Content with a corrected stable identity.",),
    )

    summary = run_pipeline(config, limit=10, full=False)

    new_path = config.output_root / article_markdown_path(
        "wsj:WP-NEW-IDENTITY",
        date(2024, 1, 2),
    )
    assert summary.changed == 1
    assert summary.succeeded == 1
    assert not old_path.exists()
    assert "corrected stable identity" in new_path.read_text(encoding="utf-8")
    with duckdb.connect(str(config.catalog_db), read_only=True) as connection:
        article_rows = connection.execute(
            "SELECT article_id, source_html_path, cleaned_markdown_path FROM articles"
        ).fetchall()
        candidate_ids = connection.execute(
            "SELECT article_id FROM candidates"
        ).fetchall()
        manifest_id = connection.execute(
            "SELECT article_id FROM source_manifest"
        ).fetchone()[0]
    assert article_rows == [
        (
            "wsj:WP-NEW-IDENTITY",
            source_path,
            new_path.relative_to(config.output_root).as_posix(),
        )
    ]
    assert candidate_ids == [("wsj:WP-NEW-IDENTITY",)]
    assert manifest_id == "wsj:WP-NEW-IDENTITY"


def test_post_commit_cleanup_failure_retains_harmless_orphan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "archive"
    source_path = "2024/one.html"
    write_article_fixture(
        archive,
        source_path,
        article_id="WP-ORPHAN",
        published="2024-01-02T15:30:00Z",
    )
    config = PipelineConfig(archive, tmp_path / "processed")
    run_pipeline(config, limit=10, full=False)
    old_path = config.output_root / article_markdown_path(
        "wsj:WP-ORPHAN",
        date(2024, 1, 2),
    )
    write_article_fixture(
        archive,
        source_path,
        article_id="WP-ORPHAN",
        published="2024-02-03T15:30:00Z",
        paragraphs=("Changed date and content.",),
    )
    real_unlink = os.unlink

    def fail_old_cleanup(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *args,
        dir_fd: int | None = None,
        **kwargs,
    ) -> None:
        if Path(path).name == old_path.name and dir_fd is not None:
            raise OSError("simulated orphan cleanup failure")
        real_unlink(path, *args, dir_fd=dir_fd, **kwargs)

    monkeypatch.setattr(os, "unlink", fail_old_cleanup)

    summary = run_pipeline(config, limit=10, full=False)

    new_path = config.output_root / article_markdown_path(
        "wsj:WP-ORPHAN",
        date(2024, 2, 3),
    )
    assert summary.changed == 1
    assert new_path.is_file()
    assert old_path.is_file()
    with duckdb.connect(str(config.catalog_db), read_only=True) as connection:
        catalog_path = connection.execute(
            "SELECT cleaned_markdown_path FROM articles"
        ).fetchone()[0]
    assert catalog_path == new_path.relative_to(config.output_root).as_posix()


def test_orphan_cleanup_does_not_follow_symlinked_date_directory(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    source_path = "2024/one.html"
    write_article_fixture(
        archive,
        source_path,
        article_id="WP-CLEANUP-SYMLINK",
        published="2024-01-02T15:30:00Z",
    )
    config = PipelineConfig(archive, tmp_path / "processed")
    run_pipeline(config, limit=10, full=False)
    old_path = config.output_root / article_markdown_path(
        "wsj:WP-CLEANUP-SYMLINK",
        date(2024, 1, 2),
    )
    external_day = tmp_path / "external-day"
    external_day.mkdir()
    external_victim = external_day / old_path.name
    external_victim.write_text("external file must survive", encoding="utf-8")
    old_path.unlink()
    old_path.parent.rmdir()
    old_path.parent.symlink_to(external_day, target_is_directory=True)
    write_article_fixture(
        archive,
        source_path,
        article_id="WP-CLEANUP-SYMLINK",
        published="2024-02-03T15:30:00Z",
        paragraphs=("Changed date and content.",),
    )

    summary = run_pipeline(config, limit=10, full=False)

    assert summary.changed == 1
    assert external_victim.read_text(encoding="utf-8") == (
        "external file must survive"
    )


def test_orphan_cleanup_symlink_loop_does_not_fail_committed_run(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    source_path = "2024/one.html"
    write_article_fixture(
        archive,
        source_path,
        article_id="WP-CLEANUP-LOOP",
        published="2024-01-02T15:30:00Z",
    )
    config = PipelineConfig(archive, tmp_path / "processed")
    run_pipeline(config, limit=10, full=False)
    old_path = config.output_root / article_markdown_path(
        "wsj:WP-CLEANUP-LOOP",
        date(2024, 1, 2),
    )
    loop_directory = old_path.parent
    old_path.unlink()
    loop_directory.rmdir()
    loop_directory.symlink_to(loop_directory, target_is_directory=True)
    write_article_fixture(
        archive,
        source_path,
        article_id="WP-CLEANUP-LOOP",
        published="2024-02-03T15:30:00Z",
        paragraphs=("Changed date and content.",),
    )

    summary = run_pipeline(config, limit=10, full=False)

    new_path = config.output_root / article_markdown_path(
        "wsj:WP-CLEANUP-LOOP",
        date(2024, 2, 3),
    )
    assert summary.changed == 1
    assert new_path.is_file()
    assert loop_directory.is_symlink()
    assert loop_directory.readlink() == loop_directory
    with duckdb.connect(str(config.catalog_db), read_only=True) as connection:
        run_status = connection.execute(
            "SELECT status FROM runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()[0]
    assert run_status == "succeeded"
