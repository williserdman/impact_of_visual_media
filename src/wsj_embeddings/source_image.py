"""Stable no-follow reads of canonical local images below the archive root."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from wsj_pipeline.file_safety import open_relative_regular


class SourceImageError(RuntimeError):
    """A stable content-free failure to read one canonical local image leaf."""

    def __init__(self, status: str) -> None:
        self.status = status
        super().__init__(f"header image path is {status}")


def read_source_image(root: Path, relative_value: str) -> bytes:
    """Read one unchanged source-relative regular single-link image leaf."""

    if not is_safe_source_relative_path(relative_value):
        raise SourceImageError("unsafe")
    status, descriptor = open_relative_regular(root, relative_value)
    if status != "ok":
        raise SourceImageError(status)
    assert descriptor is not None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise SourceImageError("unsafe")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read()
        after = os.fstat(descriptor)
    except OSError as error:
        raise SourceImageError("unsafe") from error
    finally:
        os.close(descriptor)

    if _snapshot_identity(before) != _snapshot_identity(after):
        raise SourceImageError("unsafe")
    verification_status, verification_descriptor = open_relative_regular(
        root,
        relative_value,
    )
    if verification_status != "ok":
        raise SourceImageError("unsafe")
    assert verification_descriptor is not None
    try:
        current = os.fstat(verification_descriptor)
    finally:
        os.close(verification_descriptor)
    if _snapshot_identity(after) != _snapshot_identity(current):
        raise SourceImageError("unsafe")
    return data


def is_safe_source_relative_path(relative_value: object) -> bool:
    """Recognize one contained local path without resolving or opening it."""

    if not isinstance(relative_value, str) or not relative_value:
        return False
    if relative_value.lower().startswith(("http://", "https://")):
        return False
    relative = Path(relative_value)
    return (
        not relative.is_absolute()
        and bool(relative.parts)
        and all(
            part not in {"", ".", ".."} and Path(part).name == part
            for part in relative.parts
        )
    )


def _snapshot_identity(file_stat: os.stat_result) -> tuple[int, ...]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_mode,
        file_stat.st_nlink,
        file_stat.st_size,
        file_stat.st_mtime_ns,
    )
