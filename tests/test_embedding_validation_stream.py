from __future__ import annotations

import pytest

from wsj_embeddings.validate import _rows_with_canonical_articles
from wsj_embeddings.validation_stream import (
    OrderIndependentDigest,
    RowKind,
    stream_rows,
)


class _GuardedCursor:
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self.rows = rows
        self.offset = 0
        self.requests: list[int] = []

    def fetchall(self):
        raise AssertionError("validation must not fetch all corpus rows")

    def fetchmany(self, size: int):
        self.requests.append(size)
        if size > RowKind.METADATA.page_size:
            raise AssertionError("validation exceeded the generated page ceiling")
        page = self.rows[self.offset : self.offset + size]
        self.offset += len(page)
        return page


class _GuardedConnection:
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self.cursor = _GuardedCursor(rows)

    def execute(self, _sql: str, _parameters=None):
        return self.cursor


def test_validation_row_stream_reads_every_page_without_fetchall() -> None:
    rows = [(index,) for index in range(RowKind.METADATA.page_size * 2 + 1)]
    connection = _GuardedConnection(rows)

    assert (
        list(stream_rows(connection, "SELECT generated", kind=RowKind.METADATA))
        == rows
    )
    assert connection.cursor.requests == [256, 256, 256, 256]


def test_validation_row_stream_rejects_a_page_larger_than_its_contract() -> None:
    class OversizedCursor(_GuardedCursor):
        def fetchmany(self, size: int):
            return [(1,)] * (size + 1)

    connection = _GuardedConnection([])
    connection.cursor = OversizedCursor([])

    with pytest.raises(RuntimeError, match="page ceiling"):
        list(stream_rows(connection, "SELECT generated", kind=RowKind.TRIPLE_VECTOR))


def test_artifact_digest_is_order_independent_and_duplicate_sensitive() -> None:
    forward = OrderIndependentDigest()
    reverse = OrderIndependentDigest()
    duplicated = OrderIndependentDigest()

    for record in (b"generated-a", b"generated-b"):
        forward.add(record)
    for record in (b"generated-b", b"generated-a"):
        reverse.add(record)
    for record in (b"generated-a", b"generated-b", b"generated-a"):
        duplicated.add(record)

    assert forward.hexdigest() == reverse.hexdigest()
    assert duplicated.hexdigest() != forward.hexdigest()


def test_canonical_metadata_lookup_uses_one_query_per_bounded_page() -> None:
    class CanonicalLookup:
        def __init__(self) -> None:
            self.requests: list[tuple[str, ...]] = []

        def get_many(self, article_ids):
            identifiers = tuple(str(value) for value in article_ids)
            self.requests.append(identifiers)
            return {identifier: identifier for identifier in identifiers}

    lookup = CanonicalLookup()
    rows = [(f"wsj:GENERATED-{index:04d}",) for index in range(513)]

    attached = list(_rows_with_canonical_articles(rows, lookup))  # type: ignore[arg-type]

    assert len(attached) == 513
    assert [len(request) for request in lookup.requests] == [256, 256, 1]
