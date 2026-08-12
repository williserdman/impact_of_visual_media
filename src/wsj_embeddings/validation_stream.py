"""Bounded row streaming for read-only corpus validation."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from enum import Enum
from typing import Any

_DIGEST_MODULUS = 1 << 256


class OrderIndependentDigest:
    """A duplicate-sensitive commutative digest for streamed artifact records."""

    def __init__(self) -> None:
        self._count = 0
        self._xor = 0
        self._sum = 0

    def add(self, record: bytes) -> None:
        value = int.from_bytes(hashlib.sha256(record).digest(), "big")
        self._count += 1
        self._xor ^= value
        self._sum = (self._sum + value) % _DIGEST_MODULUS

    update = add

    def hexdigest(self) -> str:
        digest = hashlib.sha256()
        digest.update(self._count.to_bytes(16, "big"))
        digest.update(self._xor.to_bytes(32, "big"))
        digest.update(self._sum.to_bytes(32, "big"))
        return digest.hexdigest()


class RowKind(Enum):
    """Memory contract for one validation query result."""

    METADATA = 256
    SINGLE_VECTOR = 32
    TRIPLE_VECTOR = 8

    @property
    def page_size(self) -> int:
        return int(self.value)


def stream_rows(
    connection: Any,
    sql: str,
    parameters: object | None = None,
    *,
    kind: RowKind = RowKind.METADATA,
) -> Iterator[tuple[object, ...]]:
    """Yield a complete ordered query using bounded cursor pages."""

    cursor = (
        connection.execute(sql, parameters)
        if parameters is not None
        else connection.execute(sql)
    )
    page_size = kind.page_size
    while True:
        page = cursor.fetchmany(page_size)
        if len(page) > page_size:
            raise RuntimeError("validation cursor exceeded its page ceiling")
        if not page:
            return
        yield from page
