"""Stable discovery and pairing of raw WSJ archive files."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

from wsj_pipeline.models import SourceCandidate

EXCLUDED_DIRECTORIES = frozenset({"__MACOSX", "backup"})
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
_IMAGE_PRIORITY = {extension: index for index, extension in enumerate(IMAGE_EXTENSIONS)}


def discover_sources(
    source_root: Path,
    limit: int | None = None,
) -> Iterator[SourceCandidate]:
    """Yield HTML/image candidates in deterministic relative-path order."""

    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")

    html_paths = sorted(
        _walk_html(source_root),
        key=lambda path: path.relative_to(source_root).as_posix(),
    )
    selected = html_paths if limit is None else html_paths[:limit]
    for html_path in selected:
        yield _candidate_for(source_root, html_path)


def _walk_html(source_root: Path) -> Iterator[Path]:
    for directory, directory_names, file_names in os.walk(source_root):
        directory_names[:] = sorted(
            name
            for name in directory_names
            if not name.startswith(".") and name not in EXCLUDED_DIRECTORIES
        )
        root = Path(directory)
        for file_name in sorted(file_names):
            if file_name.lower().endswith(".html"):
                yield root / file_name


def _candidate_for(source_root: Path, html_path: Path) -> SourceCandidate:
    html_stat = html_path.stat()
    image_paths = _matching_images(html_path)
    warnings: list[str] = []

    if not image_paths:
        warnings.append("missing_image")
        image_path = None
        image_stat = None
    else:
        if len(image_paths) > 1:
            warnings.append("multiple_images")
        image_path = image_paths[0]
        image_stat = image_path.stat()

    return SourceCandidate(
        relative_html_path=html_path.relative_to(source_root).as_posix(),
        html_size=html_stat.st_size,
        html_mtime_ns=html_stat.st_mtime_ns,
        relative_image_path=(
            image_path.relative_to(source_root).as_posix() if image_path else None
        ),
        image_size=image_stat.st_size if image_stat else None,
        image_mtime_ns=image_stat.st_mtime_ns if image_stat else None,
        warnings=tuple(warnings),
    )


def _matching_images(html_path: Path) -> list[Path]:
    target_stem = f"{html_path.stem}_main_image".casefold()
    matches = [
        candidate
        for candidate in html_path.parent.iterdir()
        if candidate.is_file()
        and candidate.stem.casefold() == target_stem
        and candidate.suffix.casefold() in _IMAGE_PRIORITY
    ]
    return sorted(
        matches,
        key=lambda path: (
            _IMAGE_PRIORITY[path.suffix.casefold()],
            path.name.casefold(),
            path.name,
        ),
    )
