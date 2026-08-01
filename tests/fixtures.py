from __future__ import annotations

import html
import json
from pathlib import Path

FigureFixture = tuple[str, str, str, str, str]


def write_article_fixture(
    root: Path,
    relative_html_path: str,
    *,
    article_id: str | None = "WP-WSJ-TEST",
    canonical_url: str | None = "https://www.wsj.com/business/test-story",
    published: str | None = "2024-01-02T15:30:00Z",
    updated: str | None = "2024-01-02T16:00:00Z",
    created: str | None = "2024-01-02T15:00:00Z",
    last_published: str | None = "2024-01-02T16:05:00Z",
    headline: str = "Synthetic Headline",
    original_headline: str = "Original Synthetic Headline",
    subtitle: str | None = None,
    dek: str | None = "Synthetic dek.",
    summary: str | None = None,
    authors: tuple[str, ...] = ("Test Reporter",),
    keywords: tuple[str, ...] = ("Markets", "Testing"),
    paragraphs: tuple[str, ...] = ("First paragraph.", "Second paragraph."),
    headings: tuple[str, ...] = (),
    list_items: tuple[str, ...] = (),
    table_rows: tuple[tuple[str, ...], ...] = (),
    quotes: tuple[str, ...] = (),
    editorial_links: tuple[tuple[str, str], ...] = (),
    inline_figures: tuple[FigureFixture, ...] = (),
    interactive_text: tuple[str, ...] = (),
    declared_word_count: int | None = None,
    with_image: bool = True,
    excluded_page_elements: bool = True,
    editorial_metadata_in_dom_only: bool = False,
) -> Path:
    """Write one entirely synthetic article fixture in editorial DOM order."""

    html_path = root / relative_html_path
    html_path.parent.mkdir(parents=True, exist_ok=True)
    news_article = {
        "@context": "https://schema.org",
        "@type": "NewsArticle",
        "headline": None if editorial_metadata_in_dom_only else headline,
        "description": (
            None if editorial_metadata_in_dom_only else (summary or dek)
        ),
        "datePublished": published,
        "author": (
            []
            if editorial_metadata_in_dom_only
            else [{"@type": "Person", "name": author} for author in authors]
        ),
    }
    json_ld = json.dumps(news_article)

    meta_values = {
        "article.id": article_id,
        "article.headline": (
            None if editorial_metadata_in_dom_only else (headline or None)
        ),
        "article.origheadline": original_headline,
        "article.subtitle": None if editorial_metadata_in_dom_only else subtitle,
        "article.dek": None if editorial_metadata_in_dom_only else dek,
        "article.summary": None if editorial_metadata_in_dom_only else summary,
        "article.published": published,
        "article.updated": updated,
        "article.created": created,
        "dateLastPubbed": last_published,
        "article.section": "Markets",
        "article.page": "Stocks",
        "article.type": "Market News",
        "article.access": "paid",
        "author": (
            None if editorial_metadata_in_dom_only else ("|".join(authors) or None)
        ),
        "keywords": ",".join(keywords),
        "article:word_count": (
            str(declared_word_count) if declared_word_count is not None else None
        ),
    }
    meta_html = "\n".join(
        f'<meta name="{html.escape(name)}" content="{html.escape(value)}">'
        for name, value in meta_values.items()
        if value is not None
    )
    canonical_html = (
        f'<link rel="canonical" href="{html.escape(canonical_url)}">'
        if canonical_url
        else ""
    )

    metadata_html = ""
    if headline:
        metadata_html += f'<h1 class="article-headline">{html.escape(headline)}</h1>'
    if subtitle:
        metadata_html += (
            f'<p class="article-subtitle">{html.escape(subtitle)}</p>'
        )
    if dek:
        metadata_html += f'<p class="article-dek">{html.escape(dek)}</p>'
    if summary:
        metadata_html += f'<p class="article-summary">{html.escape(summary)}</p>'
    if authors:
        byline = _format_byline(authors)
        metadata_html += f'<div class="article-byline">{html.escape(byline)}</div>'

    editorial_html: list[str] = [
        f'<p data-type="paragraph">{html.escape(paragraph)}</p>'
        for paragraph in paragraphs
    ]
    editorial_html.extend(
        '<p data-type="paragraph">Opening paragraph with an '
        f'<a href="{html.escape(href)}">{html.escape(label)}</a>.</p>'
        for label, href in editorial_links
    )
    editorial_html.extend(f"<h2>{html.escape(value)}</h2>" for value in headings)
    if list_items:
        items = "".join(f"<li>{html.escape(item)}</li>" for item in list_items)
        editorial_html.append(f"<ul>{items}</ul>")
    if table_rows:
        rows = "".join(
            "<tr>"
            + "".join(
                f"<{('th' if row_index == 0 else 'td')}>{html.escape(cell)}</"
                f"{('th' if row_index == 0 else 'td')}>"
                for cell in row
            )
            + "</tr>"
            for row_index, row in enumerate(table_rows)
        )
        editorial_html.append(f"<table>{rows}</table>")
    editorial_html.extend(
        f"<blockquote>{html.escape(value)}</blockquote>" for value in quotes
    )
    editorial_html.extend(_figure_html(figure) for figure in inline_figures)
    editorial_html.extend(
        '<section class="article-interactive" data-type="interactive">'
        f"<p>{html.escape(value)}</p><script>ignoredSyntheticCode()</script></section>"
        for value in interactive_text
    )

    excluded_html = ""
    if excluded_page_elements:
        excluded_html = """
        <aside class="advertisement">
          <p data-type="paragraph">Synthetic advertisement.</p>
          <img src="https://images.wsj.net/excluded-ad.jpg" alt="Excluded ad">
        </aside>
        <section class="subscription-prompt">Synthetic subscription prompt.</section>
        <aside data-inset_type="related" class="related-card">
          <p data-type="paragraph">Synthetic related story.</p>
          <img src="https://images.wsj.net/excluded-related.jpg" alt="Excluded related">
        </aside>
        <section class="author-biography">Synthetic author biography.</section>
        <nav>Global synthetic navigation.</nav>
        """

    document = f"""<!doctype html>
    <html>
      <head>
        {canonical_html}
        {meta_html}
        <script type="application/ld+json">{html.escape(json_ld, quote=False)}</script>
      </head>
      <body>
        <article>
          {metadata_html}
          <div class="article-body paywall">
            {''.join(editorial_html)}
            {excluded_html}
          </div>
        </article>
      </body>
    </html>
    """
    html_path.write_text(document, encoding="utf-8")

    if with_image:
        html_path.with_name(f"{html_path.stem}_main_image.jpg").write_bytes(
            b"\xff\xd8\xff\xe0test-jpeg"
        )
    return html_path


def _figure_html(figure: FigureFixture) -> str:
    src, alt, caption, credit, creator = figure
    return (
        "<figure>"
        f'<img src="{html.escape(src)}" alt="{html.escape(alt)}">'
        f"<figcaption>{html.escape(caption)}</figcaption>"
        f'<span class="image-credit">{html.escape(credit)}</span>'
        f'<span class="image-creator">{html.escape(creator)}</span>'
        "</figure>"
    )


def _format_byline(authors: tuple[str, ...]) -> str:
    if len(authors) == 1:
        return f"By {authors[0]}"
    return f"By {', '.join(authors[:-1])} and {authors[-1]}"
