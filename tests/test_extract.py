from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from tests.fixtures import write_article_fixture
from wsj_pipeline import extract as extract_module
from wsj_pipeline.discovery import discover_sources


def extract_only_article(archive: Path):
    [candidate] = list(discover_sources(archive))
    return extract_module.extract_article(candidate, archive)


def extract_fixture_article(tmp_path: Path):
    archive = tmp_path / "archive"
    write_article_fixture(
        archive,
        "2024/story.html",
        paragraphs=(),
        editorial_links=(("editorial link", "https://example.com/source"),),
        headings=("Synthetic section",),
        list_items=("First item", "Second item"),
        quotes=("Synthetic pull quote",),
        excluded_page_elements=False,
    )
    return extract_only_article(archive)


def test_extracts_all_editorial_content_as_markdown(tmp_path: Path) -> None:
    article = extract_fixture_article(tmp_path)

    assert article.markdown_text.splitlines() == [
        "# Synthetic Headline",
        "",
        "Synthetic dek.",
        "",
        "By Test Reporter",
        "",
        "Opening paragraph with an [editorial link](https://example.com/source).",
        "",
        "## Synthetic section",
        "",
        "- First item",
        "- Second item",
        "",
        "> Synthetic pull quote",
    ]
    assert article.editorial_block_count == 7
    assert article.editorial_character_count == len(article.markdown_text)


def test_excludes_non_editorial_page_elements(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    write_article_fixture(archive, "2024/story.html")

    article = extract_only_article(archive)

    for excluded in (
        "advertisement",
        "subscription prompt",
        "related story",
        "author biography",
        "navigation",
        "excluded-ad.jpg",
        "excluded-related.jpg",
    ):
        assert excluded not in article.markdown_text.casefold()
    assert article.inline_image_urls == ()


def test_emits_metadata_once_normalizes_unicode_and_preserves_repeated_body(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    write_article_fixture(
        archive,
        "2024/story.html",
        headline="Synthetic \u212b Headline",
        dek="Synthetic \u212b dek.",
        paragraphs=("Repeated \u212b paragraph.", "Repeated \u212b paragraph."),
        excluded_page_elements=False,
    )

    article = extract_only_article(archive)

    assert "\u212b" not in article.markdown_text
    assert article.markdown_text.count("# Synthetic Å Headline") == 1
    assert article.markdown_text.count("Synthetic Å dek.") == 1
    assert article.markdown_text.count("By Test Reporter") == 1
    assert article.markdown_text.count("Repeated Å paragraph.") == 2


def test_preserves_distinct_metadata_and_uses_dom_fallbacks(tmp_path: Path) -> None:
    structured_archive = tmp_path / "structured"
    write_article_fixture(
        structured_archive,
        "2024/story.html",
        subtitle="Synthetic subtitle.",
        summary="Synthetic summary.",
        paragraphs=(),
        excluded_page_elements=False,
    )

    structured = extract_only_article(structured_archive)

    assert structured.markdown_text.splitlines()[:9] == [
        "# Synthetic Headline",
        "",
        "Synthetic subtitle.",
        "",
        "Synthetic dek.",
        "",
        "Synthetic summary.",
        "",
        "By Test Reporter",
    ]

    dom_archive = tmp_path / "dom"
    write_article_fixture(
        dom_archive,
        "2024/story.html",
        subtitle="DOM subtitle.",
        summary="DOM summary.",
        paragraphs=(),
        excluded_page_elements=True,
        editorial_metadata_in_dom_only=True,
    )
    dom_path = dom_archive / "2024/story.html"
    dom_document = dom_path.read_text(encoding="utf-8")
    dom_document = dom_document.replace(
        "Synthetic advertisement.",
        '<span class="article-summary">Advertisement summary leak.</span>',
    ).replace(
        "Synthetic subscription prompt.",
        '<span class="article-summary">Prompt summary leak.</span>',
    ).replace(
        "Synthetic related story.",
        '<span class="article-summary">Related summary leak.</span>',
    ).replace(
        '<aside class="advertisement">',
        '<div class="story-content"><span class="article-summary">'
        'Unmarked body summary leak.</span></div><aside class="advertisement">',
    )
    dom_path.write_text(dom_document, encoding="utf-8")

    dom = extract_only_article(dom_archive)

    assert dom.markdown_text.splitlines()[:9] == [
        "# Synthetic Headline",
        "",
        "DOM subtitle.",
        "",
        "Synthetic dek.",
        "",
        "DOM summary.",
        "",
        "By Test Reporter",
    ]
    assert dom.markdown_text.count("Unmarked body summary leak.") == 1
    for excluded_leak in (
        "Advertisement summary leak.",
        "Prompt summary leak.",
        "Related summary leak.",
    ):
        assert excluded_leak not in dom.markdown_text


def test_escapes_literal_markdown_block_markers_in_prose(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    write_article_fixture(
        archive,
        "2024/story.html",
        paragraphs=(
            "# Literal heading marker",
            "## Literal level-two heading marker",
            "### Literal level-three heading marker",
            "> Literal quote marker",
            ">> Literal nested quote marker",
            "- Literal list marker",
            "+ Literal plus-list marker",
            "1. Literal ordered-list marker",
            "---",
            "~~~",
            "#",
            ">",
        ),
        excluded_page_elements=False,
    )

    article = extract_only_article(archive)

    for literal in (
        r"\# Literal heading marker",
        r"\## Literal level-two heading marker",
        r"\### Literal level-three heading marker",
        r"\> Literal quote marker",
        r"\>> Literal nested quote marker",
        r"\- Literal list marker",
        r"\+ Literal plus-list marker",
        r"1\. Literal ordered-list marker",
        r"\---",
        r"\~~~",
        r"\#",
        r"\>",
    ):
        assert literal in article.markdown_text.splitlines()


def test_keeps_unlisted_inline_elements_in_one_mixed_content_block(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    html_path = write_article_fixture(
        archive,
        "2024/story.html",
        paragraphs=(),
        excluded_page_elements=False,
    )
    document = html_path.read_text(encoding="utf-8").replace(
        '            \n            \n          </div>',
        """<div class="article-interactive">Before <code>x</code>
        <time>noon</time> <abbr>WSJ</abbr> <mark>marked</mark>
        <label>label</label> after.</div></div>""",
    )
    html_path.write_text(document, encoding="utf-8")

    article = extract_only_article(archive)

    assert (
        "Before x noon WSJ marked label after."
        in article.markdown_text.splitlines()
    )


def test_retains_empty_editorial_content_with_warning(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    write_article_fixture(
        archive,
        "2024/story.html",
        headline="",
        dek=None,
        authors=(),
        paragraphs=(),
        excluded_page_elements=False,
    )

    article = extract_only_article(archive)

    assert article.markdown_text == ""
    assert article.editorial_character_count == 0
    assert article.editorial_block_count == 0
    assert "empty_content" in article.warnings


def test_renders_tables_interactives_and_escapes_markdown(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    html_path = write_article_fixture(
        archive,
        "2024/story.html",
        paragraphs=("Synthetic *literal* text.",),
        table_rows=(("Label", "Value"), ("Synthetic", "A | B")),
        interactive_text=("Synthetic interactive annotation.",),
        excluded_page_elements=False,
    )
    document = html_path.read_text(encoding="utf-8").replace(
        '<section class="article-interactive" data-type="interactive">',
        '<form class="article-interactive" data-type="interactive">',
    ).replace("</section>", "</form>", 1)
    document = document.replace(
        "          </div>\n        </article>",
        "          <footer><p data-type=\"paragraph\">"
        "Synthetic correction note.</p></footer></div>\n        </article>",
    )
    html_path.write_text(document, encoding="utf-8")

    article = extract_only_article(archive)

    assert "Synthetic \\*literal\\* text." in article.markdown_text
    assert "| Label | Value |" in article.markdown_text
    assert "| --- | --- |" in article.markdown_text
    assert "| Synthetic | A \\| B |" in article.markdown_text
    assert "Synthetic interactive annotation." in article.markdown_text
    assert "Synthetic correction note." in article.markdown_text
    assert "ignoredSyntheticCode" not in article.markdown_text


def test_extracts_ordered_deduplicated_http_inline_images(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    write_article_fixture(
        archive,
        "2024/story.html",
        paragraphs=(),
        inline_figures=(
            (
                "https://images.wsj.net/inline-one.jpg",
                "Synthetic alt",
                "Synthetic caption.",
                "TEST CREDIT.",
                "Test Creator.",
            ),
            (
                "//images.wsj.net/inline-two.png",
                "Second synthetic alt",
                "Second synthetic caption.",
                "SECOND CREDIT.",
                "Second Creator.",
            ),
            (
                "https://images.wsj.net/inline-one.jpg",
                "Duplicate alt",
                "Duplicate caption.",
                "DUPLICATE CREDIT.",
                "Duplicate Creator.",
            ),
            (
                "ftp://images.wsj.net/rejected.jpg",
                "Rejected alt",
                "Rejected caption.",
                "REJECTED CREDIT.",
                "Rejected Creator.",
            ),
        ),
    )

    article = extract_only_article(archive)

    assert article.inline_image_urls == (
        "https://images.wsj.net/inline-one.jpg",
        "https://images.wsj.net/inline-two.png",
    )
    assert (
        "![Synthetic alt](https://images.wsj.net/inline-one.jpg)"
        in article.markdown_text
    )
    assert "Synthetic caption. TEST CREDIT." in article.markdown_text
    assert article.markdown_text.count("inline-one.jpg") == 1
    assert "rejected.jpg" not in article.markdown_text
    assert "Duplicate caption. DUPLICATE CREDIT. Duplicate Creator." in (
        article.markdown_text
    )
    assert "Rejected caption. REJECTED CREDIT. Rejected Creator." in (
        article.markdown_text
    )


def test_extracts_nested_lazy_image_from_valid_fallback_source(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    html_path = write_article_fixture(
        archive,
        "2024/story.html",
        paragraphs=("Synthetic nested image marker.",),
        excluded_page_elements=False,
    )
    document = html_path.read_text(encoding="utf-8").replace(
        "Synthetic nested image marker.",
        'Before <img src="data:image/gif;base64,AAAA" data-src="/nested.png" '
        'alt="Nested synthetic alt"> after.',
    )
    html_path.write_text(document, encoding="utf-8")

    article = extract_only_article(archive)

    assert article.inline_image_urls == ("https://www.wsj.com/nested.png",)
    assert (
        "Before ![Nested synthetic alt](https://www.wsj.com/nested.png) after."
        in article.markdown_text
    )


def test_preserves_figure_and_interactive_dom_order_with_links_and_images(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    html_path = write_article_fixture(
        archive,
        "2024/story.html",
        paragraphs=(),
        inline_figures=(
            (
                "https://images.wsj.net/replace-me.jpg",
                "Replace me",
                "Replace caption.",
                "REPLACE CREDIT.",
                "Replace Creator.",
            ),
        ),
        interactive_text=("Replace interactive.",),
        excluded_page_elements=False,
    )
    document = html_path.read_text(encoding="utf-8")
    figure_start = document.index("<figure>")
    figure_end = document.index("</figure>", figure_start) + len("</figure>")
    document = (
        document[:figure_start]
        + """<figure><figcaption>Caption <a href="/caption-link">link</a>.
        <span class="image-credit">TEST CREDIT.
        <span class="image-creator">Test Creator.</span></span>
        <img src="/caption-image.png" alt="Caption image"> End.</figcaption></figure>"""
        + document[figure_end:]
    )
    document = document.replace(
        "<p>Replace interactive.</p><script>ignoredSyntheticCode()</script>",
        'Interactive <a href="/interactive-link">link</a> '
        '<img src="/interactive-image.png" alt="Interactive image"> after.',
    )
    html_path.write_text(document, encoding="utf-8")

    article = extract_only_article(archive)

    assert (
        "Caption [link](https://www.wsj.com/caption-link). TEST CREDIT. "
        "Test Creator. ![Caption image](https://www.wsj.com/caption-image.png) End."
        in article.markdown_text
    )
    assert article.markdown_text.count("Test Creator.") == 1
    assert (
        "Interactive [link](https://www.wsj.com/interactive-link) "
        "![Interactive image](https://www.wsj.com/interactive-image.png) after."
        in article.markdown_text
    )
    assert article.inline_image_urls == (
        "https://www.wsj.com/caption-image.png",
        "https://www.wsj.com/interactive-image.png",
    )


def test_preserves_nested_lists_and_multi_paragraph_quotes(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    html_path = write_article_fixture(
        archive,
        "2024/story.html",
        paragraphs=(),
        excluded_page_elements=False,
    )
    document = html_path.read_text(encoding="utf-8").replace(
        '            \n            \n          </div>',
        """<ul><li>Parent item<ul><li>Nested item</li></ul></li></ul>
        <ol><li>One</li><li>Two</li><li>Three</li><li>Four</li><li>Five</li>
        <li>Six</li><li>Seven</li><li>Eight</li><li>Nine</li>
        <li>Tenth<ul><li>Nested under tenth</li></ul></li></ol>
        <blockquote><p>First quoted paragraph.</p>
        <p>Second quoted paragraph.</p></blockquote></div>""",
    )
    html_path.write_text(document, encoding="utf-8")

    article = extract_only_article(archive)

    assert "- Parent item\n  - Nested item" in article.markdown_text
    assert "10. Tenth\n    - Nested under tenth" in article.markdown_text
    assert (
        "> First quoted paragraph.\n>\n> Second quoted paragraph."
        in article.markdown_text
    )


def test_resolves_relative_editorial_links_against_canonical_url(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    write_article_fixture(
        archive,
        "2024/story.html",
        canonical_url="HTTPS://WWW.WSJ.COM:443/business/story/?b=2&a=1#comments",
        paragraphs=(),
        editorial_links=(
            ("relative source", "/source"),
            ("complex source", "https://example.com/a_(synthetic)"),
            ("unsafe source", "javascript:synthetic()"),
        ),
        excluded_page_elements=False,
    )

    article = extract_only_article(archive)

    assert article.canonical_url == "https://www.wsj.com/business/story?a=1&b=2"
    assert "[relative source](https://www.wsj.com/source)" in article.markdown_text
    assert (
        "[complex source](https://example.com/a_%28synthetic%29)"
        in article.markdown_text
    )
    assert "unsafe source" in article.markdown_text
    assert "javascript:" not in article.markdown_text


def test_extracts_structured_identity_dates_and_provenance(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    html_path = write_article_fixture(archive, "2024/story.html")

    article = extract_only_article(archive)

    assert article.article_id == "wsj:WP-WSJ-TEST"
    assert article.relative_html_path == "2024/story.html"
    assert article.source_html_sha256 == hashlib.sha256(
        html_path.read_bytes()
    ).hexdigest()
    assert article.extractor_version == extract_module.EXTRACTOR_VERSION == "4"
    assert article.published_at_utc == datetime(2024, 1, 2, 15, 30, tzinfo=UTC)
    assert article.publication_date_new_york == date(2024, 1, 2)
    assert article.updated_at_utc == datetime(2024, 1, 2, 16, 0, tzinfo=UTC)
    assert article.relative_header_image_path == "2024/story_main_image.jpg"
    assert article.metadata_completeness > 0
    assert article.warnings == ()


def test_identity_prefers_wsj_id_then_normalized_url_then_html_hash() -> None:
    normalized_url = "https://www.wsj.com/business/story?a=1&b=2"
    expected_digest = hashlib.sha256(normalized_url.encode()).hexdigest()

    assert (
        extract_module.derive_article_id(
            " WP-WSJ-123 ", "https://www.wsj.com/ignored", "abc"
        )
        == "wsj:WP-WSJ-123"
    )
    assert extract_module.derive_article_id(
        None,
        "HTTPS://WWW.WSJ.COM:443/business/story/?b=2&a=1#fragment",
        "abc",
    ) == f"url:{expected_digest}"
    assert extract_module.derive_article_id(None, None, "abc") == "html:abc"


@pytest.mark.parametrize(
    ("published", "code"),
    [
        (None, "missing_publication_timestamp"),
        ("synthetic-invalid-timestamp", "invalid_timestamp"),
        ("2024-01-02T15:30:00", "naive_timestamp"),
    ],
)
def test_publication_timestamp_failures_are_content_free(
    tmp_path: Path,
    published: str | None,
    code: str,
) -> None:
    archive = tmp_path / "archive"
    write_article_fixture(archive, "2024/story.html", published=published)

    with pytest.raises(extract_module.ExtractionError) as caught:
        extract_only_article(archive)

    assert caught.value.code == code
    if published:
        assert published not in str(caught.value)


@pytest.mark.parametrize(
    "updated",
    ("malformed-optional-update", "2024-01-02T16:00:00"),
    ids=("malformed", "naive"),
)
def test_invalid_optional_update_is_dropped_with_content_free_warning(
    tmp_path: Path,
    updated: str,
) -> None:
    archive = tmp_path / "archive"
    write_article_fixture(
        archive,
        "2024/story.html",
        published="2024-01-02T15:30:00Z",
        updated=updated,
    )

    article = extract_only_article(archive)

    assert article.updated_at_utc is None
    assert article.warnings == ("invalid_optional_update",)
    assert updated not in repr(article.warnings)


def test_missing_header_image_is_retained_as_warning(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    write_article_fixture(archive, "2024/story.html", with_image=False)

    article = extract_only_article(archive)

    assert article.relative_header_image_path is None
    assert article.warnings == ("missing_image",)
