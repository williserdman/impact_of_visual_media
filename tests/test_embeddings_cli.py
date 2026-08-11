from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb

import wsj_embeddings.pipeline as embedding_pipeline_module
from wsj_embeddings import FakeEmbeddingAdapter
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
    assert json.loads(line) == {"articles": 1, "embeddings": 1}
    assert line == f"{json.dumps(json.loads(line), sort_keys=True)}\n"
    assert adapter.texts == ["# Complete A\n\nFirst paragraph.\n"]
    with duckdb.connect(str(roots[2] / "catalog.duckdb"), read_only=True) as db:
        embedded = db.execute(
            """
            SELECT article_id, published_at_utc, publication_date_new_york
            FROM embeddings
            """
        ).fetchall()
    assert embedded == [
        (
            "wsj:A",
            datetime(2024, 1, 3, 15, 30, tzinfo=UTC),
            date(2024, 1, 3),
        )
    ]


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
