from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import date
from hashlib import sha256
from pathlib import Path

import duckdb
import pytest

from tests.fixtures import write_article_fixture
from wsj_pipeline.catalog import Catalog
from wsj_pipeline.config import PipelineConfig
from wsj_pipeline.pipeline import (
    PipelineLockedError,
    article_markdown_path,
    pipeline_lock,
    run_pipeline,
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
