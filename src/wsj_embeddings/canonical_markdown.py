"""Stable no-follow reads of canonical Markdown below preprocessing output."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from wsj_pipeline.file_safety import open_relative_regular


class CanonicalMarkdownError(RuntimeError):
    """A stable content-free failure to read one canonical Markdown leaf."""

    def __init__(self, status: str) -> None:
        self.status = status
        super().__init__(f"canonical Markdown path is {status}")


def read_canonical_markdown(root: Path, relative_value: str) -> bytes:
    """Read one unchanged regular single-link leaf through no-follow descriptors."""

    status, descriptor = open_relative_regular(root, relative_value)
    if status != "ok":
        raise CanonicalMarkdownError(status)
    assert descriptor is not None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise CanonicalMarkdownError("unsafe")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read()
        after = os.fstat(descriptor)
    except OSError as error:
        raise CanonicalMarkdownError("unsafe") from error
    finally:
        os.close(descriptor)

    if _snapshot_identity(before) != _snapshot_identity(after):
        raise CanonicalMarkdownError("unsafe")

    verification_status, verification_descriptor = open_relative_regular(
        root,
        relative_value,
    )
    if verification_status != "ok":
        raise CanonicalMarkdownError("unsafe")
    assert verification_descriptor is not None
    try:
        current = os.fstat(verification_descriptor)
    finally:
        os.close(verification_descriptor)
    if _snapshot_identity(after) != _snapshot_identity(current):
        raise CanonicalMarkdownError("unsafe")
    return data


def _snapshot_identity(file_stat: os.stat_result) -> tuple[int, ...]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_mode,
        file_stat.st_nlink,
        file_stat.st_size,
        file_stat.st_mtime_ns,
    )
