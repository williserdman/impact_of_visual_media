"""End-to-end orchestration independent of command-line presentation."""

from __future__ import annotations

import html
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from pathlib import Path

from wsj_pipeline.canonical import recompute_canonical
from wsj_pipeline.config import PipelineConfig
from wsj_pipeline.discovery import discover_sources
from wsj_pipeline.index import IndexSummary, build_index
from wsj_pipeline.publish import (
    PublishSummary,
    publish_audits,
    publish_partitions,
)
from wsj_pipeline.state import (
    PipelineState,
    ProcessSummary,
    process_candidates,
)
from wsj_pipeline.validate import ValidationReport, validate_outputs


class PipelineLockedError(RuntimeError):
    """Raised when another mutating pipeline process owns the lock."""


@dataclass(frozen=True, slots=True)
class InventorySummary:
    """Counts from a read-only source inventory."""

    discovered: int
    with_image: int
    missing_image: int


@dataclass(frozen=True, slots=True)
class RunSummary:
    """End-to-end results suitable for JSON output."""

    run_id: str
    processing: ProcessSummary
    partitions_written: int
    rows_written: int
    article_count: int
    validation_ok: bool
    validation_issues: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@contextmanager
def pipeline_lock(lock_path: Path) -> Iterator[None]:
    """Own an exclusive recoverable lock file for one mutating operation."""

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
        os.write(descriptor, f"{os.getpid()}\n".encode())
        os.close(descriptor)
        yield
    finally:
        with suppress(FileNotFoundError):
            lock_path.unlink()


def inventory_sources(
    config: PipelineConfig,
    *,
    limit: int | None = None,
) -> InventorySummary:
    candidates = list(discover_sources(config.source_root, limit=limit))
    with_image = sum(
        candidate.relative_image_path is not None for candidate in candidates
    )
    return InventorySummary(
        discovered=len(candidates),
        with_image=with_image,
        missing_image=len(candidates) - with_image,
    )


def run_incremental(
    config: PipelineConfig,
    *,
    limit: int | None,
    full: bool,
    reprocess: bool = False,
) -> RunSummary:
    """Run incremental extraction through validated publication."""

    _require_scope(limit, full)
    scope = "full" if full else f"limit:{limit}"
    if reprocess:
        scope = f"{scope},reprocess"
    lock_path = config.output_root / "state" / "pipeline.lock"
    with pipeline_lock(lock_path):
        with PipelineState.open(config.state_db) as state:
            run_id = state.begin_run("run", scope)
        try:
            processing = process_candidates(
                config,
                discover_sources(config.source_root, limit=limit),
                run_id,
                reprocess_stale=reprocess,
            )
            with PipelineState.open(config.state_db) as state:
                canonical = recompute_canonical(
                    state,
                    processing.affected_article_keys,
                    run_id,
                )
                publication = publish_partitions(
                    state,
                    config,
                    canonical.affected_partitions,
                    run_id,
                )
                publish_audits(state, config, run_id)
            index_summary = build_index(config, run_id)
            validation = validate_outputs(config)
            summary = _run_summary(
                run_id,
                processing,
                publication,
                index_summary,
                validation,
            )
            with PipelineState.open(config.state_db) as state:
                state.finish_run(
                    run_id,
                    status="succeeded" if validation.ok else "failed",
                    summary=summary.as_dict(),
                )
                publish_audits(state, config, run_id)
            return summary
        except Exception as error:
            with suppress(Exception), PipelineState.open(config.state_db) as state:
                state.finish_run(
                    run_id,
                    status="failed",
                    summary={"error_type": type(error).__name__},
                )
                publish_audits(state, config, run_id)
            raise


def process_only(
    config: PipelineConfig,
    *,
    limit: int | None,
    full: bool,
    reprocess: bool = False,
) -> ProcessSummary:
    """Run only incremental inventory/extraction state updates."""

    _require_scope(limit, full)
    scope = "full" if full else f"limit:{limit}"
    if reprocess:
        scope = f"{scope},reprocess"
    with pipeline_lock(config.output_root / "state" / "pipeline.lock"):
        with PipelineState.open(config.state_db) as state:
            run_id = state.begin_run("process", scope)
        try:
            summary = process_candidates(
                config,
                discover_sources(config.source_root, limit=limit),
                run_id,
                reprocess_stale=reprocess,
            )
        except Exception as error:
            with PipelineState.open(config.state_db) as state:
                state.finish_run(
                    run_id,
                    status="failed",
                    summary={"error_type": type(error).__name__},
                )
            raise
        with PipelineState.open(config.state_db) as state:
            state.finish_run(
                run_id,
                status="succeeded",
                summary=summary.as_dict(),
            )
        return summary


def publish_all(config: PipelineConfig) -> dict[str, object]:
    """Recompute and publish every article already present in state."""

    with pipeline_lock(config.output_root / "state" / "pipeline.lock"):
        with PipelineState.open(config.state_db) as state:
            run_id = state.begin_run("publish", "all-state")
            article_keys = [
                row[0]
                for row in state.connection.execute(
                    "SELECT DISTINCT article_key FROM extracted_sources"
                ).fetchall()
            ]
            canonical = recompute_canonical(state, article_keys, run_id)
            publication = publish_partitions(
                state,
                config,
                canonical.affected_partitions,
                run_id,
            )
            state.finish_run(
                run_id,
                status="succeeded",
                summary=asdict(publication),
            )
            publish_audits(state, config, run_id)
        return {
            "run_id": run_id,
            **asdict(publication),
            "duplicates_recorded": canonical.duplicates_recorded,
        }


def run_smoke() -> RunSummary:
    """Run the whole pipeline against a generated temporary archive."""

    with tempfile.TemporaryDirectory(prefix="wsj-pipeline-smoke-") as directory:
        root = Path(directory)
        archive = root / "archive"
        _write_smoke_article(
            archive / "2024" / "a.html",
            article_id="SMOKE-A",
            body="A longer canonical smoke article body.",
            with_image=True,
        )
        _write_smoke_article(
            archive / "2024" / "a-duplicate.html",
            article_id="SMOKE-A",
            body="Short.",
            with_image=True,
        )
        _write_smoke_article(
            archive / "2025" / "2025" / "b.html",
            article_id="SMOKE-B",
            body="A second smoke article.",
            with_image=False,
            published="2025-02-03T14:00:00Z",
        )
        config = PipelineConfig(
            source_root=archive,
            output_root=root / "processed",
            batch_size=2,
        )
        return run_incremental(config, limit=None, full=True)


def _run_summary(
    run_id: str,
    processing: ProcessSummary,
    publication: PublishSummary,
    index_summary: IndexSummary,
    validation: ValidationReport,
) -> RunSummary:
    return RunSummary(
        run_id=run_id,
        processing=processing,
        partitions_written=publication.partitions_written,
        rows_written=publication.rows_written,
        article_count=index_summary.article_count,
        validation_ok=validation.ok,
        validation_issues=tuple(issue.code for issue in validation.issues),
    )


def _require_scope(limit: int | None, full: bool) -> None:
    if full == (limit is not None):
        raise ValueError("choose exactly one of a positive limit or full mode")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")


def _write_smoke_article(
    html_path: Path,
    *,
    article_id: str,
    body: str,
    with_image: bool,
    published: str = "2024-01-02T15:30:00Z",
) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "@context": "https://schema.org",
        "@type": "NewsArticle",
        "headline": f"Headline {article_id}",
        "datePublished": published,
        "author": [{"@type": "Person", "name": "Smoke Reporter"}],
    }
    document = f"""<!doctype html>
    <html><head>
      <link rel="canonical" href="https://www.wsj.com/smoke/{article_id}">
      <meta name="article.id" content="{html.escape(article_id)}">
      <meta name="article.headline" content="Headline {html.escape(article_id)}">
      <meta name="article.published" content="{published}">
      <meta name="article.updated" content="{published}">
      <meta name="author" content="Smoke Reporter">
      <script type="application/ld+json">{json.dumps(metadata)}</script>
    </head><body><article>
      <p data-type="paragraph">{html.escape(body)}</p>
    </article></body></html>"""
    html_path.write_text(document, encoding="utf-8")
    if with_image:
        html_path.with_name(f"{html_path.stem}_main_image.jpg").write_bytes(
            b"\xff\xd8\xff\xe0smoke"
        )
