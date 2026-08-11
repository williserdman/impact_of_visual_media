from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import pytest

import wsj_embeddings.pipeline as embedding_pipeline_module
from wsj_embeddings import FakeEmbeddingAdapter
from wsj_embeddings.adapters import JinaHttpResponse
from wsj_embeddings.catalog import configuration_id
from wsj_embeddings.cli import main
from wsj_pipeline.catalog import ArticleRecord, Catalog


class RecordingFakeAdapter(FakeEmbeddingAdapter):
    """Record generated Markdown while retaining deterministic fake vectors."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed_text(self, text: str) -> tuple[float, ...]:
        self.texts.append(text)
        return super().embed_text(text)


def write_generated_cli_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_root = tmp_path / "source"
    preprocessing_root = tmp_path / "preprocessed"
    embedding_root = tmp_path / "embeddings"
    source_root.mkdir()
    header_path = source_root / "headers" / "a.jpg"
    header_path.parent.mkdir()
    header_path.write_bytes(b"generated CLI header image")
    article_rows = (
        ("wsj:B", "# Complete B\n\nSecond paragraph.\n", None),
        ("wsj:A", "# Complete A\n\nFirst paragraph.\n", "headers/a.jpg"),
    )
    with (
        Catalog.open(preprocessing_root / "catalog.duckdb") as catalog,
        catalog.transaction(),
    ):
        for index, (article_id, markdown, header_path) in enumerate(article_rows):
            markdown_path = preprocessing_root / "text" / f"article-{index}.md"
            markdown_path.parent.mkdir(parents=True, exist_ok=True)
            markdown_path.write_text(markdown, encoding="utf-8")
            catalog.replace_article(
                ArticleRecord(
                    article_id=article_id,
                    source_html_path=f"synthetic/article-{index}.html",
                    cleaned_markdown_path=markdown_path.relative_to(
                        preprocessing_root
                    ).as_posix(),
                    header_image_path=header_path,
                    inline_image_urls=(),
                    published_at_utc=datetime(
                        2024, 1, index + 2, 15, 30, tzinfo=UTC
                    ),
                    publication_date_new_york=date(2024, 1, index + 2),
                    cleaned_markdown_sha256=hashlib.sha256(
                        markdown.encode()
                    ).hexdigest(),
                )
            )
    return source_root, preprocessing_root, embedding_root


def root_arguments(
    source_root: Path,
    preprocessing_root: Path,
    embedding_root: Path,
) -> list[str]:
    return [
        "--source-root",
        str(source_root),
        "--preprocessing-output-root",
        str(preprocessing_root),
        "--embedding-output-root",
        str(embedding_root),
    ]


def test_embedding_smoke_is_fixture_safe_and_deterministic(capsys) -> None:
    exit_code = main(["smoke"])
    line = capsys.readouterr().out

    assert exit_code == 0
    assert json.loads(line) == {
        "articles": 1,
        "embeddings": 1,
        "validation_ok": True,
    }
    assert line == f"{json.dumps(json.loads(line), sort_keys=True)}\n"


def test_inventory_is_credential_free_read_only_and_never_constructs_adapter(
    tmp_path: Path,
    capsys,
) -> None:
    """Break caught: inventory requires a key, mutates output, or creates an adapter."""

    roots = write_generated_cli_fixture(tmp_path)

    def fail_if_constructed():
        raise AssertionError("inventory constructed an embedding adapter")

    exit_code = main(
        ["inventory", *root_arguments(*roots)],
        environment={},
        adapter_factory=fail_if_constructed,
    )
    line = capsys.readouterr().out

    assert exit_code == 0
    assert json.loads(line) == {
        "canonical_articles": 2,
        "eligible_text": 2,
        "header_absent": 1,
        "header_present": 1,
    }
    assert line == f"{json.dumps(json.loads(line), sort_keys=True)}\n"
    assert not roots[2].exists()


def test_run_refuses_without_affirmative_authorization_before_adapter_or_state(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    """Break caught: a credential alone bypasses licensed-processing consent."""

    roots = write_generated_cli_fixture(tmp_path)
    before = {
        path.relative_to(tmp_path).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for root in roots[:2]
        for path in root.rglob("*")
        if path.is_file()
    }

    def fail_if_constructed():
        raise AssertionError("unauthorized run constructed an embedding adapter")

    def fail_if_markdown_read(*_args, **_kwargs):
        raise AssertionError("unauthorized run read canonical Markdown")

    monkeypatch.setattr(
        embedding_pipeline_module,
        "read_canonical_markdown",
        fail_if_markdown_read,
    )

    exit_code = main(
        ["run", *root_arguments(*roots), "--limit", "1"],
        environment={"JINA_API_KEY": "synthetic-present-but-insufficient"},
        adapter_factory=fail_if_constructed,
    )
    line = capsys.readouterr().out

    assert exit_code == 1
    assert json.loads(line) == {"error": "hosted_processing_not_authorized"}
    assert not roots[2].exists()
    assert before == {
        path.relative_to(tmp_path).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for root in roots[:2]
        for path in root.rglob("*")
        if path.is_file()
    }


def test_authorized_limited_run_embeds_complete_lexical_winner_with_fake(
    tmp_path: Path,
    capsys,
) -> None:
    """Break caught: run selects catalog order or sends partial canonical Markdown."""

    roots = write_generated_cli_fixture(tmp_path)
    adapter = RecordingFakeAdapter()

    exit_code = main(
        [
            "run",
            *root_arguments(*roots),
            "--limit",
            "1",
            "--authorize-hosted-processing",
        ],
        environment={},
        adapter_factory=lambda: adapter,
    )
    line = capsys.readouterr().out

    assert exit_code == 0
    assert json.loads(line) == {
        "articles": 1,
        "attempted": 2,
        "configuration_id": configuration_id(FakeEmbeddingAdapter.profile),
        "embeddings": 3,
        "header_absent": 0,
        "header_failed": 0,
        "interrupted": 0,
        "retryable": 0,
        "reused": 0,
        "succeeded": 3,
        "terminal": 0,
    }
    assert line == f"{json.dumps(json.loads(line), sort_keys=True)}\n"
    assert adapter.texts == ["# Complete A\n\nFirst paragraph.\n"]
    with duckdb.connect(str(roots[2] / "catalog.duckdb"), read_only=True) as db:
        embedded = db.execute(
            """
            SELECT article_id, modality, published_at_utc,
                   publication_date_new_york
            FROM embeddings
            ORDER BY modality
            """
        ).fetchall()
    assert embedded == [
        (
            "wsj:A",
            "article_text",
            datetime(2024, 1, 3, 15, 30, tzinfo=UTC),
            date(2024, 1, 3),
        ),
        (
            "wsj:A",
            "header_image",
            datetime(2024, 1, 3, 15, 30, tzinfo=UTC),
            date(2024, 1, 3),
        ),
        (
            "wsj:A",
            "multimodal_article",
            datetime(2024, 1, 3, 15, 30, tzinfo=UTC),
            date(2024, 1, 3),
        ),
    ]


def test_authorized_reprocess_flag_regenerates_only_the_limited_cli_scope(
    tmp_path: Path,
    capsys,
) -> None:
    """Break caught: CLI reprocess is ignored or expands beyond its positive limit."""

    roots = write_generated_cli_fixture(tmp_path)
    assert (
        main(
            [
                "run",
                *root_arguments(*roots),
                "--limit",
                "2",
                "--authorize-hosted-processing",
            ],
            adapter_factory=RecordingFakeAdapter,
        )
        == 0
    )
    capsys.readouterr()
    adapter = RecordingFakeAdapter()

    exit_code = main(
        [
            "run",
            *root_arguments(*roots),
            "--limit",
            "1",
            "--reprocess",
            "--authorize-hosted-processing",
        ],
        adapter_factory=lambda: adapter,
    )
    result = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert result == {
        "articles": 1,
        "attempted": 1,
        "configuration_id": configuration_id(FakeEmbeddingAdapter.profile),
        "embeddings": 2,
        "header_absent": 0,
        "header_failed": 0,
        "interrupted": 0,
        "retryable": 0,
        "reused": 1,
        "succeeded": 2,
        "terminal": 0,
    }
    assert adapter.texts == ["# Complete A\n\nFirst paragraph.\n"]
    with duckdb.connect(str(roots[2] / "catalog.duckdb"), read_only=True) as db:
        assert db.execute("SELECT count(*) FROM embeddings").fetchone() == (4,)
        assert db.execute(
            """
            SELECT article_id, modality
            FROM embedding_generation_history
            ORDER BY modality
            """
        ).fetchall() == [
            ("wsj:A", "article_text"),
            ("wsj:A", "multimodal_article"),
        ]


@pytest.mark.parametrize(
    ("disposition", "expected_retryable", "expected_absent", "expected_failed"),
    (
        ("absent", 0, 1, 0),
        ("missing", 1, 0, 1),
    ),
)
def test_run_summary_distinguishes_absent_from_failed_header(
    tmp_path: Path,
    capsys,
    disposition: str,
    expected_retryable: int,
    expected_absent: int,
    expected_failed: int,
) -> None:
    """Break caught: CLI summaries merge no-header and failed-header coverage."""

    roots = write_generated_cli_fixture(tmp_path)
    if disposition == "absent":
        with (
            Catalog.open(roots[1] / "catalog.duckdb") as catalog,
            catalog.transaction(),
        ):
            catalog.connection.execute(
                "UPDATE articles SET header_image_path = NULL "
                "WHERE article_id = 'wsj:A'"
            )
    else:
        (roots[0] / "headers" / "a.jpg").unlink()

    exit_code = main(
        [
            "run",
            *root_arguments(*roots),
            "--limit",
            "1",
            "--authorize-hosted-processing",
        ],
        adapter_factory=RecordingFakeAdapter,
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload == {
        "articles": 1,
        "attempted": 1,
        "configuration_id": configuration_id(FakeEmbeddingAdapter.profile),
        "embeddings": 1,
        "header_absent": expected_absent,
        "header_failed": expected_failed,
        "interrupted": 0,
        "retryable": expected_retryable,
        "reused": 0,
        "succeeded": 1,
        "terminal": 0,
    }


def test_authorized_run_reports_missing_credential_without_output_state(
    tmp_path: Path,
    capsys,
) -> None:
    """Break caught: production adapter credential failure leaks or mutates state."""

    roots = write_generated_cli_fixture(tmp_path)

    exit_code = main(
        [
            "run",
            *root_arguments(*roots),
            "--limit",
            "1",
            "--authorize-hosted-processing",
        ],
        environment={},
    )
    line = capsys.readouterr().out

    assert exit_code == 1
    assert json.loads(line) == {"error": "missing_credential"}
    assert line == f"{json.dumps(json.loads(line), sort_keys=True)}\n"
    assert not roots[2].exists()


@pytest.mark.parametrize("catalog_state", ("missing", "malformed"))
def test_inventory_reports_catalog_failure_as_one_content_free_json_result(
    tmp_path: Path,
    capsys,
    catalog_state: str,
) -> None:
    """Break caught: expected inventory catalog failures escape as tracebacks."""

    roots = write_generated_cli_fixture(tmp_path)
    catalog_path = roots[1] / "catalog.duckdb"
    catalog_path.unlink()
    if catalog_state == "malformed":
        with duckdb.connect(str(catalog_path)) as db:
            db.execute("CREATE TABLE unsafe_secret_table(value VARCHAR)")

    exit_code = main(["inventory", *root_arguments(*roots)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == '{"error": "preprocessing_catalog"}\n'
    assert captured.err == ""
    assert str(tmp_path) not in captured.out
    assert "unsafe_secret_table" not in captured.out
    assert not roots[2].exists()


def test_inventory_reports_invalid_root_configuration_without_path_disclosure(
    tmp_path: Path,
    capsys,
) -> None:
    """Break caught: disjoint-root validation escapes with configured paths."""

    roots = write_generated_cli_fixture(tmp_path)

    exit_code = main(
        ["inventory", *root_arguments(roots[0], roots[1], roots[1])]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == '{"error": "invalid_configuration"}\n'
    assert captured.err == ""
    assert str(tmp_path) not in captured.out


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    (
        ("timeout", "timeout"),
        ("rate", "rate_limit"),
        ("rejection", "deterministic_request"),
    ),
)
def test_run_reports_post_construction_hosted_failure_without_unsafe_output(
    tmp_path: Path,
    capsys,
    failure: str,
    expected_code: str,
) -> None:
    """Break caught: hosted request failures leak their cause or request content."""

    class FailingTransport:
        def post(self, url, *, headers, body, timeout_seconds):
            del url, headers, body, timeout_seconds
            if failure == "timeout":
                raise TimeoutError("transport-secret")
            return JinaHttpResponse(
                status_code=429 if failure == "rate" else 413,
                headers={"Retry-After": "2"},
                body=b"response-body-secret",
            )

    roots = write_generated_cli_fixture(tmp_path)

    exit_code = main(
        [
            "run",
            *root_arguments(*roots),
            "--limit",
            "1",
            "--authorize-hosted-processing",
        ],
        environment={"JINA_API_KEY": "credential-secret"},
        transport=FailingTransport(),
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == f'{json.dumps({"error": expected_code}, sort_keys=True)}\n'
    assert captured.err == ""
    for forbidden in (
        str(tmp_path),
        "credential-secret",
        "transport-secret",
        "response-body-secret",
        "Complete A",
    ):
        assert forbidden not in captured.out
    expected_state = "retryable" if failure in {"timeout", "rate"} else "terminal"
    expected_status = (
        None if failure == "timeout" else (429 if failure == "rate" else 413)
    )
    with duckdb.connect(str(roots[2] / "catalog.duckdb"), read_only=True) as db:
        assert db.execute(
            """
            SELECT state, attempt_count, error_code, status_code
            FROM embedding_work_items
            """
        ).fetchone() == (expected_state, 1, expected_code, expected_status)
        assert db.execute("SELECT count(*) FROM embeddings").fetchone() == (0,)
    assert not (roots[2] / "pipeline.lock").exists()
    catalog_bytes = (roots[2] / "catalog.duckdb").read_bytes()
    for forbidden in (
        b"credential-secret",
        b"transport-secret",
        b"response-body-secret",
        b"Complete A",
    ):
        assert forbidden not in catalog_bytes


def test_run_reports_existing_lock_without_removing_or_disclosing_it(
    tmp_path: Path,
    capsys,
) -> None:
    """Break caught: a normal lock conflict raises a path-bearing traceback."""

    roots = write_generated_cli_fixture(tmp_path)
    roots[2].mkdir()
    lock_path = roots[2] / "pipeline.lock"
    lock_path.write_text("existing-lock-secret\n", encoding="utf-8")

    exit_code = main(
        [
            "run",
            *root_arguments(*roots),
            "--limit",
            "1",
            "--authorize-hosted-processing",
        ],
        adapter_factory=FakeEmbeddingAdapter,
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == '{"error": "embedding_pipeline_locked"}\n'
    assert captured.err == ""
    assert str(tmp_path) not in captured.out
    assert "existing-lock-secret" not in captured.out
    assert lock_path.read_text(encoding="utf-8") == "existing-lock-secret\n"
    assert not (roots[2] / "catalog.duckdb").exists()


def test_run_reports_unsafe_output_root_without_writing_through_it(
    tmp_path: Path,
    capsys,
) -> None:
    """Break caught: unsafe output rejection escapes or writes through a symlink."""

    roots = write_generated_cli_fixture(tmp_path)
    outside = tmp_path / "outside-secret"
    outside.mkdir()

    def replace_output_before_coordinator():
        roots[2].symlink_to(outside, target_is_directory=True)
        return FakeEmbeddingAdapter()

    exit_code = main(
        [
            "run",
            *root_arguments(*roots),
            "--limit",
            "1",
            "--authorize-hosted-processing",
        ],
        adapter_factory=replace_output_before_coordinator,
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == '{"error": "embedding_output_root"}\n'
    assert captured.err == ""
    assert str(tmp_path) not in captured.out
    assert list(outside.iterdir()) == []


def test_run_reports_malformed_embedding_catalog_and_cleans_owned_lock(
    tmp_path: Path,
    capsys,
) -> None:
    """Break caught: incompatible embedding output raises schema exception text."""

    roots = write_generated_cli_fixture(tmp_path)
    roots[2].mkdir()
    with duckdb.connect(str(roots[2] / "catalog.duckdb")) as db:
        db.execute("CREATE TABLE legacy_secret_table(value INTEGER)")

    exit_code = main(
        [
            "run",
            *root_arguments(*roots),
            "--limit",
            "1",
            "--authorize-hosted-processing",
        ],
        adapter_factory=FakeEmbeddingAdapter,
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == '{"error": "embedding_catalog"}\n'
    assert captured.err == ""
    assert str(tmp_path) not in captured.out
    assert "legacy_secret_table" not in captured.out
    assert not (roots[2] / "pipeline.lock").exists()


def test_run_reports_pipeline_failure_without_article_identity_or_path(
    tmp_path: Path,
    capsys,
) -> None:
    """Break caught: classified pipeline errors expose article or filesystem data."""

    roots = write_generated_cli_fixture(tmp_path)
    (roots[1] / "text" / "article-1.md").unlink()

    exit_code = main(
        [
            "run",
            *root_arguments(*roots),
            "--limit",
            "1",
            "--authorize-hosted-processing",
        ],
        adapter_factory=FakeEmbeddingAdapter,
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == '{"error": "read_markdown"}\n'
    assert captured.err == ""
    for forbidden in (str(tmp_path), "wsj:A", "article-1.md", "Complete A"):
        assert forbidden not in captured.out
    with duckdb.connect(str(roots[2] / "catalog.duckdb"), read_only=True) as db:
        assert db.execute(
            "SELECT state, attempt_count FROM embedding_work_items"
        ).fetchone() == ("in_progress", 1)
        assert db.execute("SELECT count(*) FROM embeddings").fetchone() == (0,)
    assert not (roots[2] / "pipeline.lock").exists()


def test_run_does_not_swallow_programmer_error_from_constructed_adapter(
    tmp_path: Path,
    capsys,
) -> None:
    """Break caught: adapter programming bugs become operational JSON errors."""

    roots = write_generated_cli_fixture(tmp_path)

    class BrokenAdapter(FakeEmbeddingAdapter):
        def embed_text(self, text: str) -> tuple[float, ...]:
            del text
            raise AssertionError("programmer error remains visible")

    with pytest.raises(AssertionError, match="programmer error remains visible"):
        main(
            [
                "run",
                *root_arguments(*roots),
                "--limit",
                "1",
                "--authorize-hosted-processing",
            ],
            adapter_factory=BrokenAdapter,
        )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
