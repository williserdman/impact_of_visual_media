from __future__ import annotations

import html
import json
from pathlib import Path


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
    headline: str = "A Test Headline",
    original_headline: str = "The Original Test Headline",
    summary: str = "A useful summary.",
    authors: tuple[str, ...] = ("Alice Reporter", "Bob Editor"),
    keywords: tuple[str, ...] = ("Markets", "Testing"),
    paragraphs: tuple[str, ...] = ("First paragraph.", "Second paragraph."),
    declared_word_count: int | None = None,
    with_image: bool = True,
    excluded_paragraphs: bool = True,
) -> Path:
    html_path = root / relative_html_path
    html_path.parent.mkdir(parents=True, exist_ok=True)
    image_url = "https://images.wsj.net/im-123/social"
    image_object = {
        "@context": "https://schema.org",
        "@type": "ImageObject",
        "url": image_url,
        "caption": "Traders on a test floor.",
        "creditText": "TEST NEWS",
        "creator": {"@type": "Person", "name": "Test Photographer"},
        "width": 1280,
        "height": 720,
    }
    news_article = {
        "@context": "https://schema.org",
        "@type": "NewsArticle",
        "headline": headline,
        "datePublished": published,
        "author": [{"@type": "Person", "name": author} for author in authors],
    }
    json_ld = json.dumps([news_article, image_object])

    meta_values = {
        "article.id": article_id,
        "article.headline": headline,
        "article.origheadline": original_headline,
        "article.summary": summary,
        "article.published": published,
        "article.updated": updated,
        "article.created": created,
        "dateLastPubbed": last_published,
        "article.section": "Markets",
        "article.page": "Stocks",
        "article.type": "Market News",
        "article.access": "paid",
        "author": "|".join(authors),
        "keywords": ",".join(keywords),
        "twitter:image:alt": "A test trading floor",
        "og:image": image_url,
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
    paragraph_html = "\n".join(
        f'<p data-type="paragraph">{html.escape(paragraph)}</p>'
        for paragraph in paragraphs
    )
    excluded_html = ""
    if excluded_paragraphs:
        excluded_html = """
        <aside data-inset_type="related">
          <p data-type="paragraph">Related card text.</p>
        </aside>
        <figure>
          <p data-type="paragraph">Caption text.</p>
        </figure>
        <section class="author-module">
          <p data-type="paragraph">Author biography.</p>
        </section>
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
          <div class="article-body paywall">
            {paragraph_html}
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
