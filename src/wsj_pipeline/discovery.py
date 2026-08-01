"""Stable discovery and pairing of raw WSJ archive files."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

from wsj_pipeline.models import SourceCandidate

EXCLUDED_DIRECTORIES = frozenset({"__MACOSX"})
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
_IMAGE_PRIORITY = {extension: index for index, extension in enumerate(IMAGE_EXTENSIONS)}


def discover_sources(
    source_root: Path,
    limit: int | None = None,
) -> Iterator[SourceCandidate]:
    """Yield HTML/image candidates in deterministic relative-path order."""

    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")

    indexed_directory: Path | None = None
    image_index: dict[str, list[Path]] = {}
    for yielded, html_path in enumerate(
        _walk_html(source_root, top_level=True),
        start=1,
    ):
        if html_path.parent != indexed_directory:
            indexed_directory = html_path.parent
            image_index = _index_images(indexed_directory)
        yield _candidate_for(source_root, html_path, image_index)
        if limit is not None and yielded >= limit:
            return


def _walk_html(directory: Path, *, top_level: bool) -> Iterator[Path]:
    entries: list[tuple[str, Path, bool]] = []
    with os.scandir(directory) as directory_entries:
        for entry in directory_entries:
            is_directory = entry.is_dir(follow_symlinks=False)
            if is_directory and (
                entry.name.startswith(".")
                or entry.name in EXCLUDED_DIRECTORIES
                or (top_level and entry.name == "backup")
            ):
                continue
            entries.append(
                (
                    f"{entry.name}/" if is_directory else entry.name,
                    Path(entry.path),
                    is_directory,
                )
            )

    for _, path, is_directory in sorted(entries, key=lambda item: item[0]):
        if is_directory:
            yield from _walk_html(path, top_level=False)
        elif path.name.lower().endswith(".html"):
            yield path


def _candidate_for(
    source_root: Path,
    html_path: Path,
    image_index: dict[str, list[Path]],
) -> SourceCandidate:
    html_stat = html_path.stat()
    image_paths = _matching_images(html_path, image_index)
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
        relative_header_image_path=(
            image_path.relative_to(source_root).as_posix() if image_path else None
        ),
        header_image_size=image_stat.st_size if image_stat else None,
        header_image_mtime_ns=image_stat.st_mtime_ns if image_stat else None,
        warnings=tuple(warnings),
    )


def _matching_images(
    html_path: Path,
    image_index: dict[str, list[Path]],
) -> list[Path]:
    target_stem = f"{html_path.stem}_main_image".casefold()
    matches = image_index.get(target_stem, [])
    return sorted(
        matches,
        key=lambda path: (
            _IMAGE_PRIORITY[path.suffix.casefold()],
            path.name.casefold(),
            path.name,
        ),
    )


def _index_images(directory: Path) -> dict[str, list[Path]]:
    image_index: dict[str, list[Path]] = {}
    for candidate in directory.iterdir():
        if candidate.is_file() and candidate.suffix.casefold() in _IMAGE_PRIORITY:
            image_index.setdefault(candidate.stem.casefold(), []).append(candidate)
    return image_index
