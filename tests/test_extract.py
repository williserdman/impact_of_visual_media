from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tests.fixtures import write_article_fixture
from wsj_pipeline.discovery import discover_sources
from wsj_pipeline.extract import EXTRACTOR_VERSION, ExtractionError, extract_article


def extract_only_article(archive: Path):
    [candidate] = list(discover_sources(archive))
    return extract_article(candidate, archive)


def test_extracts_normalized_metadata_body_and_image(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    html_path = write_article_fixture(
        archive,
        "2024/2024/story.html",
        paragraphs=(
            "First   paragraph with \u212b.",
            "Second paragraph.",
        ),
    )

    article = extract_only_article(archive)

    assert article.wsj_article_id == "WP-WSJ-TEST"
    assert article.canonical_url == "https://www.wsj.com/business/test-story"
    assert article.headline == "A Test Headline"
    assert article.original_headline == "The Original Test Headline"
    assert article.summary == "A useful summary."
    assert article.authors == ("Alice Reporter", "Bob Editor")
    assert article.keywords == ("Markets", "Testing")
    assert article.section == "Markets"
    assert article.page == "Stocks"
    assert article.article_type == "Market News"
    assert article.access == "paid"
    assert article.body_paragraphs == (
        "First paragraph with Å.",
        "Second paragraph.",
    )
    assert article.body_text == "First paragraph with Å.\n\nSecond paragraph."
    assert "Related card" not in article.body_text
    assert "Caption text" not in article.body_text
    assert "Author biography" not in article.body_text
    assert article.body_word_count == 6
    assert article.relative_image_path == "2024/2024/story_main_image.jpg"
    assert article.image_present is True
    assert article.image_media_type == "image/jpeg"
    assert article.image_size_bytes == len(b"\xff\xd8\xff\xe0test-jpeg")
    assert article.image_caption == "Traders on a test floor."
    assert article.image_credit == "TEST NEWS"
    assert article.image_creator == "Test Photographer"
    assert article.image_alt == "A test trading floor"
    assert article.image_width == 1280
    assert article.image_height == 720
    assert (
        article.source_html_sha256 == hashlib.sha256(html_path.read_bytes()).hexdigest()
    )
    assert article.extractor_version == EXTRACTOR_VERSION
    assert json.loads(article.raw_metadata_json)["article.id"] == "WP-WSJ-TEST"


def test_missing_image_is_preserved_as_warning(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    write_article_fixture(archive, "2016/story.html", with_image=False)

    article = extract_only_article(archive)

    assert article.image_present is False
    assert article.relative_image_path is None
    assert article.image_media_type is None
    assert "missing_image" in article.warnings


def test_declared_word_count_mismatch_is_warned(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    write_article_fixture(
        archive,
        "2016/story.html",
        paragraphs=("Only three words.",),
        declared_word_count=900,
    )

    article = extract_only_article(archive)

    assert article.body_word_count == 3
    assert "word_count_mismatch" in article.warnings


def test_json_ld_supplies_authors_when_meta_author_is_absent(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    html_path = write_article_fixture(archive, "2016/story.html")
    document = html_path.read_text(encoding="utf-8")
    document = document.replace(
        '<meta name="author" content="Alice Reporter|Bob Editor">', ""
    )
    html_path.write_text(document, encoding="utf-8")

    article = extract_only_article(archive)

    assert article.authors == ("Alice Reporter", "Bob Editor")


def test_missing_body_raises_content_free_error(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    html_path = write_article_fixture(
        archive,
        "2016/story.html",
        paragraphs=(),
        excluded_paragraphs=False,
    )
    secret_text = "licensed-secret-article-fragment"
    html_path.write_text(
        html_path.read_text(encoding="utf-8").replace(
            '<div class="article-body paywall">',
            f'<div class="article-body paywall">{secret_text}',
        ),
        encoding="utf-8",
    )

    with pytest.raises(ExtractionError) as exc:
        extract_only_article(archive)

    assert exc.value.code == "missing_body"
    assert secret_text not in str(exc.value)
