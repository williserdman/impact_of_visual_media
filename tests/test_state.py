from pathlib import Path

import duckdb
import pytest

from wsj_pipeline.state import PipelineState, StateError

EXPECTED_TABLES = {
    "canonical_sources",
    "duplicates",
    "extracted_sources",
    "failures",
    "metadata",
    "runs",
    "source_manifest",
}


def test_open_creates_versioned_operational_schema(tmp_path: Path) -> None:
    database_path = tmp_path / "state" / "pipeline.duckdb"

    with PipelineState.open(database_path):
        pass

    with duckdb.connect(str(database_path), read_only=True) as connection:
        tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
        schema_version = connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()[0]

    assert tables == EXPECTED_TABLES
    assert schema_version == "1"


def test_run_lifecycle_records_scope_counts_and_status(tmp_path: Path) -> None:
    database_path = tmp_path / "pipeline.duckdb"

    with PipelineState.open(database_path) as state:
        run_id = state.begin_run("process", "limit:5")
        state.finish_run(
            run_id,
            status="succeeded",
            summary={"discovered": 5, "succeeded": 4, "failed": 1},
        )

    with duckdb.connect(str(database_path), read_only=True) as connection:
        row = connection.execute(
            """
            SELECT command, scope, status, summary_json, finished_at
            FROM runs
            WHERE run_id = ?
            """,
            [run_id],
        ).fetchone()

    assert row[0:3] == ("process", "limit:5", "succeeded")
    assert '"failed":1' in row[3]
    assert row[4] is not None


def test_open_rejects_incompatible_schema_version(tmp_path: Path) -> None:
    database_path = tmp_path / "pipeline.duckdb"
    with duckdb.connect(str(database_path)) as connection:
        connection.execute(
            "CREATE TABLE metadata (key VARCHAR PRIMARY KEY, value VARCHAR)"
        )
        connection.execute("INSERT INTO metadata VALUES ('schema_version', '99')")

    with pytest.raises(StateError, match="schema version 99"):
        PipelineState.open(database_path)


def test_context_manager_closes_connection(tmp_path: Path) -> None:
    state = PipelineState.open(tmp_path / "pipeline.duckdb")
    connection = state.connection

    with state:
        connection.execute("SELECT 1")

    with pytest.raises(duckdb.ConnectionException):
        connection.execute("SELECT 1")
