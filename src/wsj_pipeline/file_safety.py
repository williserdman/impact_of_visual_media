"""Descriptor-relative no-follow access to files inside trusted roots."""

from __future__ import annotations

import errno
import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path

READ_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
DIRECTORY_FLAGS = READ_FLAGS | os.O_DIRECTORY
_HASH_CHUNK_SIZE = 1024 * 1024


class RelativeFileError(RuntimeError):
    """A stable content-free failure to open a safe relative regular file."""

    def __init__(self, status: str) -> None:
        self.status = status
        self.code = f"{status}_source_path"
        super().__init__(f"source path is {status}")


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    """Bytes and stat values obtained from the same verified descriptor."""

    data: bytes
    size: int
    mtime_ns: int


def open_relative_regular(root: Path, relative_value: str) -> tuple[str, int | None]:
    """Open one relative regular single-link leaf without following links."""

    relative = Path(relative_value)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(not _safe_component(part) for part in relative.parts)
    ):
        return "unsafe", None

    try:
        root_descriptor = os.open(root, DIRECTORY_FLAGS)
    except FileNotFoundError:
        return "missing", None
    except OSError:
        return "unsafe", None

    parent_descriptor = os.dup(root_descriptor)
    file_descriptor: int | None = None
    try:
        for component in relative.parts[:-1]:
            try:
                next_descriptor = os.open(
                    component,
                    DIRECTORY_FLAGS,
                    dir_fd=parent_descriptor,
                )
            except FileNotFoundError:
                return "missing", None
            except OSError:
                return "unsafe", None
            os.close(parent_descriptor)
            parent_descriptor = next_descriptor

        try:
            file_descriptor = os.open(
                relative.parts[-1],
                READ_FLAGS,
                dir_fd=parent_descriptor,
            )
        except FileNotFoundError:
            return "missing", None
        except OSError as error:
            if error.errno == errno.ENOENT:
                return "missing", None
            return "unsafe", None
        file_stat = os.fstat(file_descriptor)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
            os.close(file_descriptor)
            file_descriptor = None
            return "unsafe", None
        return "ok", file_descriptor
    finally:
        os.close(parent_descriptor)
        os.close(root_descriptor)


def require_relative_regular(root: Path, relative_value: str) -> int:
    status, descriptor = open_relative_regular(root, relative_value)
    if status != "ok":
        raise RelativeFileError(status)
    assert descriptor is not None
    return descriptor


def stat_relative_regular(root: Path, relative_value: str) -> os.stat_result:
    descriptor = require_relative_regular(root, relative_value)
    try:
        return os.fstat(descriptor)
    finally:
        os.close(descriptor)


def read_relative_regular(root: Path, relative_value: str) -> FileSnapshot:
    descriptor = require_relative_regular(root, relative_value)
    try:
        file_stat = os.fstat(descriptor)
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read()
    finally:
        os.close(descriptor)
    return FileSnapshot(data, file_stat.st_size, file_stat.st_mtime_ns)


def sha256_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            while chunk := stream.read(_HASH_CHUNK_SIZE):
                digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _safe_component(component: str) -> bool:
    return (
        bool(component)
        and component not in {".", ".."}
        and Path(component).name == component
    )
