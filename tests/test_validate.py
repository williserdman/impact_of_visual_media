from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import duckdb
import pytest

from tests.fixtures import write_article_fixture
from wsj_pipeline import validate as validation
from wsj_pipeline.catalog import Catalog
from wsj_pipeline.config import PipelineConfig
from wsj_pipeline.pipeline import run_pipeline


def _build_outputs(
    tmp_path: Path,
    *,
    with_duplicate: bool = False,
) -> PipelineConfig:
    source_root = tmp_path / "archive"
    write_article_fixture(
        source_root,
        "2023/late-utc.html",
        article_id="WP-LATE-UTC",
        published="2024-01-01T02:30:00Z",
        inline_figures=(
            (
                "https://images.example.test/late.jpg",
                "Synthetic late image",
                "Synthetic caption",
                "Synthetic credit",
                "Synthetic creator",
            ),
        ),
    )
    write_article_fixture(
        source_root,
        "2024/january.html",
        article_id="WP-JANUARY",
        published="2024-01-15T15:30:00Z",
        inline_figures=(
            (
                "https://images.example.test/first.jpg",
                "Synthetic first image",
                "Synthetic caption",
                "Synthetic credit",
                "Synthetic creator",
            ),
            (
                "https://images.example.test/second.jpg",
                "Synthetic second image",
                "Synthetic caption",
                "Synthetic credit",
                "Synthetic creator",
            ),
        ),
    )
    if with_duplicate:
        write_article_fixture(
            source_root,
            "2024/january-copy.html",
            article_id="WP-JANUARY",
            published="2024-01-15T15:30:00Z",
            paragraphs=("Short synthetic losing duplicate.",),
        )
    config = PipelineConfig(source_root, tmp_path / "processed", batch_size=1)
    run_pipeline(config, limit=None, full=True)
    return config


def _article_rows(config: PipelineConfig) -> list[tuple[object, ...]]:
    with duckdb.connect(str(config.catalog_db), read_only=True) as connection:
        return connection.execute(
            """
            SELECT article_id, source_html_path, cleaned_markdown_path,
                   header_image_path, inline_image_urls, published_at_utc,
                   publication_date_new_york, cleaned_markdown_sha256
            FROM articles
            ORDER BY article_id
            """
        ).fetchall()


def _codes(config: PipelineConfig) -> list[str]:
    return [issue.code for issue in validation.validate_outputs(config).issues]


def test_clean_outputs_validate_successfully(tmp_path: Path) -> None:
    config = _build_outputs(tmp_path)

    report = validation.validate_outputs(config)

    assert report.ok
    assert report.issues == ()


def test_validation_detects_unsupported_catalog_version(tmp_path: Path) -> None:
    config = _build_outputs(tmp_path)
    with duckdb.connect(str(config.catalog_db)) as connection:
        connection.execute(
            "UPDATE metadata SET value = '999' WHERE key = 'schema_version'"
        )

    assert "unsupported_catalog_version" in _codes(config)


def test_validation_detects_unsupported_extractor_version(tmp_path: Path) -> None:
    config = _build_outputs(tmp_path)
    with duckdb.connect(str(config.catalog_db)) as connection:
        connection.execute(
            """
            UPDATE candidates
            SET extractor_version = '999'
            WHERE source_html_path = '2024/january.html'
            """
        )

    assert "unsupported_extractor_version" in _codes(config)


def test_validation_detects_missing_markdown(tmp_path: Path) -> None:
    config = _build_outputs(tmp_path)
    row = _article_rows(config)[0]
    (config.output_root / str(row[2])).unlink()

    assert "missing_markdown" in _codes(config)


def test_validation_detects_markdown_hash_mismatch_without_exposing_text(
    tmp_path: Path,
) -> None:
    config = _build_outputs(tmp_path)
    row = _article_rows(config)[0]
    sentinel = "SYNTHETIC PRIVATE SENTINEL"
    (config.output_root / str(row[2])).write_text(sentinel, encoding="utf-8")

    report = validation.validate_outputs(config)

    assert "markdown_hash_mismatch" in {issue.code for issue in report.issues}
    assert all(sentinel not in issue.message for issue in report.issues)


def test_validation_detects_missing_header_image(tmp_path: Path) -> None:
    config = _build_outputs(tmp_path)
    row = _article_rows(config)[0]
    (config.source_root / str(row[3])).unlink()

    assert "missing_header_image" in _codes(config)


@pytest.mark.parametrize(
    ("duplicate_column", "expected_code"),
    [
        ("article_id", "duplicate_article_id"),
        ("source_html_path", "duplicate_source_html_path"),
        ("cleaned_markdown_path", "duplicate_markdown_path"),
    ],
)
def test_validation_detects_duplicate_research_keys_if_constraints_are_bypassed(
    tmp_path: Path,
    duplicate_column: str,
    expected_code: str,
) -> None:
    config = _build_outputs(tmp_path)
    with duckdb.connect(str(config.catalog_db)) as connection:
        connection.execute("CREATE TABLE unconstrained AS SELECT * FROM articles")
        connection.execute("DROP TABLE articles")
        connection.execute("ALTER TABLE unconstrained RENAME TO articles")
        replacements = {
            "article_id": "wsj:SYNTHETIC-DUPLICATE",
            "source_html_path": "synthetic/duplicate.html",
            "cleaned_markdown_path": "text/2099/01/01/synthetic-duplicate.md",
        }
        replacements.pop(duplicate_column)
        connection.execute(
            f"""
            INSERT INTO articles
            SELECT
                {"?" if "article_id" in replacements else "article_id"},
                {
                    "?"
                    if "source_html_path" in replacements
                    else "source_html_path"
                },
                {
                    "?"
                    if "cleaned_markdown_path" in replacements
                    else "cleaned_markdown_path"
                },
                header_image_path, inline_image_urls, published_at_utc,
                publication_date_new_york, cleaned_markdown_sha256
            FROM articles
            ORDER BY article_id
            LIMIT 1
            """,
            list(replacements.values()),
        )

    assert expected_code in _codes(config)


def test_catalog_constraints_reject_duplicate_article_and_paths(
    tmp_path: Path,
) -> None:
    config = _build_outputs(tmp_path)
    row = _article_rows(config)[0]
    with duckdb.connect(str(config.catalog_db)) as connection:
        for index, duplicate_column in enumerate((0, 1, 2), start=1):
            duplicate = list(row)
            duplicate[0] = f"wsj:CONSTRAINT-{index}"
            duplicate[1] = f"synthetic/constraint-{index}.html"
            duplicate[2] = f"text/2099/01/0{index}/constraint-{index}.md"
            duplicate[duplicate_column] = row[duplicate_column]
            with pytest.raises(duckdb.ConstraintException):
                connection.execute(
                    "INSERT INTO articles VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    duplicate,
                )


def test_validation_detects_utc_new_york_date_mismatch(tmp_path: Path) -> None:
    config = _build_outputs(tmp_path)
    with duckdb.connect(str(config.catalog_db)) as connection:
        connection.execute(
            """
            UPDATE articles
            SET publication_date_new_york = DATE '2024-01-01'
            WHERE article_id = 'wsj:WP-LATE-UTC'
            """
        )

    assert "publication_date_mismatch" in _codes(config)


def test_validation_detects_date_path_mismatch(tmp_path: Path) -> None:
    config = _build_outputs(tmp_path)
    with duckdb.connect(str(config.catalog_db)) as connection:
        connection.execute(
            """
            UPDATE articles
            SET cleaned_markdown_path =
                'text/1999/12/31/' || sha256(article_id) || '.md'
            WHERE article_id = 'wsj:WP-JANUARY'
            """
        )

    assert "markdown_path_mismatch" in _codes(config)


def test_validation_detects_invalid_inline_image_url(tmp_path: Path) -> None:
    config = _build_outputs(tmp_path)
    with duckdb.connect(str(config.catalog_db)) as connection:
        connection.execute(
            """
            UPDATE articles
            SET inline_image_urls = ['file:///synthetic/private.jpg']
            WHERE article_id = 'wsj:WP-JANUARY'
            """
        )

    assert "invalid_inline_image_url" in _codes(config)


def test_validation_detects_unstable_inline_image_url_list(tmp_path: Path) -> None:
    config = _build_outputs(tmp_path)
    with duckdb.connect(str(config.catalog_db)) as connection:
        connection.execute(
            """
            UPDATE articles
            SET inline_image_urls = [
                'https://images.example.test/first.jpg',
                'https://images.example.test/first.jpg'
            ]
            WHERE article_id = 'wsj:WP-JANUARY'
            """
        )

    assert "unstable_inline_image_urls" in _codes(config)


def test_validation_detects_broken_duplicate_winner(tmp_path: Path) -> None:
    config = _build_outputs(tmp_path, with_duplicate=True)
    with duckdb.connect(str(config.catalog_db)) as connection:
        connection.execute(
            """
            UPDATE duplicates
            SET winner_source_html_path = 'synthetic/missing-winner.html'
            WHERE article_id = 'wsj:WP-JANUARY'
            """
        )

    assert "broken_duplicate_winner" in _codes(config)


def test_validation_detects_inconsistent_manifest_winner(tmp_path: Path) -> None:
    config = _build_outputs(tmp_path)
    with duckdb.connect(str(config.catalog_db)) as connection:
        connection.execute(
            """
            UPDATE source_manifest
            SET article_id = 'wsj:SYNTHETIC-WRONG-IDENTITY'
            WHERE source_html_path = '2024/january.html'
            """
        )

    assert "inconsistent_manifest_winner" in _codes(config)


def test_validation_detects_running_run(tmp_path: Path) -> None:
    config = _build_outputs(tmp_path)
    with Catalog.open(config.catalog_db) as catalog:
        catalog.begin_run("validate", "fixture")

    assert "running_run" in _codes(config)


@pytest.mark.parametrize("status", ["failed", "missing"])
def test_validation_rejects_failed_or_missing_winner(
    tmp_path: Path,
    status: str,
) -> None:
    config = _build_outputs(tmp_path)
    with duckdb.connect(str(config.catalog_db)) as connection:
        connection.execute(
            """
            UPDATE source_manifest
            SET status = ?
            WHERE source_html_path = '2024/january.html'
            """,
            [status],
        )

    assert "failed_or_missing_winner" in _codes(config)


def test_validation_detects_orphan_markdown(tmp_path: Path) -> None:
    config = _build_outputs(tmp_path)
    orphan = config.text_root / "2099/01/01/orphan.md"
    orphan.parent.mkdir(parents=True)
    orphan.write_text("Synthetic orphan.", encoding="utf-8")

    assert "orphan_markdown" in _codes(config)


def test_validation_rejects_output_path_traversal(tmp_path: Path) -> None:
    config = _build_outputs(tmp_path)
    outside = tmp_path / "outside.md"
    outside.write_text("Synthetic outside data.", encoding="utf-8")
    with duckdb.connect(str(config.catalog_db)) as connection:
        connection.execute(
            """
            UPDATE articles
            SET cleaned_markdown_path = '../outside.md',
                cleaned_markdown_sha256 = ?
            WHERE article_id = 'wsj:WP-JANUARY'
            """,
            [sha256(outside.read_bytes()).hexdigest()],
        )

    assert "unsafe_markdown_path" in _codes(config)


def test_validation_rejects_source_path_traversal(tmp_path: Path) -> None:
    config = _build_outputs(tmp_path)
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"synthetic outside image")
    with duckdb.connect(str(config.catalog_db)) as connection:
        connection.execute(
            """
            UPDATE articles
            SET header_image_path = '../outside.jpg'
            WHERE article_id = 'wsj:WP-JANUARY'
            """
        )

    assert "unsafe_header_image_path" in _codes(config)


def test_validation_does_not_follow_markdown_symlink(tmp_path: Path) -> None:
    config = _build_outputs(tmp_path)
    row = _article_rows(config)[0]
    markdown = config.output_root / str(row[2])
    outside = tmp_path / "outside.md"
    outside.write_bytes(markdown.read_bytes())
    markdown.unlink()
    markdown.symlink_to(outside)

    assert "unsafe_markdown_path" in _codes(config)


def test_validation_does_not_follow_header_directory_symlink(
    tmp_path: Path,
) -> None:
    config = _build_outputs(tmp_path)
    row = _article_rows(config)[0]
    header_path = Path(str(row[3]))
    real_directory = config.source_root / header_path.parent
    relocated = tmp_path / "relocated-source-directory"
    real_directory.rename(relocated)
    real_directory.symlink_to(relocated, target_is_directory=True)

    assert "unsafe_header_image_path" in _codes(config)


def test_validation_streams_each_catalog_file_before_requesting_the_next(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_outputs(tmp_path)
    real_iterator = validation._iter_catalog_files
    yielded_paths: list[str] = []

    def mutate_after_each_yield(connection):
        for item in real_iterator(connection):
            yielded_paths.append(item.cleaned_markdown_path)
            yield item
            (config.output_root / item.cleaned_markdown_path).write_text(
                "Synthetic mutation after validation.",
                encoding="utf-8",
            )

    monkeypatch.setattr(validation, "_iter_catalog_files", mutate_after_each_yield)

    report = validation.validate_outputs(config)

    assert report.ok
    assert len(yielded_paths) == 2


def test_validation_reports_issues_in_stable_order(tmp_path: Path) -> None:
    config = _build_outputs(tmp_path)
    orphan = config.text_root / "2099/01/01/orphan.md"
    orphan.parent.mkdir(parents=True)
    orphan.write_text("Synthetic orphan.", encoding="utf-8")
    row = _article_rows(config)[0]
    (config.output_root / str(row[2])).unlink()

    report = validation.validate_outputs(config)

    assert report.issues == tuple(
        sorted(report.issues, key=lambda issue: (issue.code, issue.message))
    )
