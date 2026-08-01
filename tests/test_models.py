from dataclasses import FrozenInstanceError, fields
from datetime import UTC, date, datetime

import pytest

from wsj_pipeline.models import ExtractedArticle, SourceCandidate


def test_source_candidate_uses_explicit_header_image_fields() -> None:
    candidate = SourceCandidate(
        relative_html_path="2024/story.html",
        html_size=123,
        html_mtime_ns=456,
        relative_header_image_path="2024/story_main_image.jpg",
        header_image_size=789,
        header_image_mtime_ns=101112,
    )

    assert candidate.relative_header_image_path == "2024/story_main_image.jpg"
    assert candidate.header_image_size == 789
    assert candidate.header_image_mtime_ns == 101112
    assert candidate.warnings == ()


def test_extracted_article_contract_is_immutable() -> None:
    published_at = datetime(2024, 1, 2, tzinfo=UTC)
    article = ExtractedArticle(
        article_id="wsj:example",
        relative_html_path="2024/story.html",
        source_html_sha256="a" * 64,
        extractor_version="3",
        canonical_url="https://example.com/story",
        published_at_utc=published_at,
        publication_date_new_york=date(2024, 1, 1),
        updated_at_utc=None,
        markdown_text="# Synthetic headline",
        editorial_character_count=21,
        editorial_block_count=1,
        metadata_completeness=4,
        relative_header_image_path="2024/story_main_image.jpg",
        inline_image_urls=("https://images.example.com/story.jpg",),
        warnings=("synthetic_warning",),
    )

    assert [field.name for field in fields(ExtractedArticle)] == [
        "article_id",
        "relative_html_path",
        "source_html_sha256",
        "extractor_version",
        "canonical_url",
        "published_at_utc",
        "publication_date_new_york",
        "updated_at_utc",
        "markdown_text",
        "editorial_character_count",
        "editorial_block_count",
        "metadata_completeness",
        "relative_header_image_path",
        "inline_image_urls",
        "warnings",
    ]
    assert article.inline_image_urls == ("https://images.example.com/story.jpg",)
    assert article.warnings == ("synthetic_warning",)
    assert ExtractedArticle.__dataclass_params__.frozen is True

    with pytest.raises(FrozenInstanceError):
        article.markdown_text = "changed"
