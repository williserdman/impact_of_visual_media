"""Versioned extraction of deterministic editorial Markdown."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import parse_qsl, quote, urlencode, urljoin, urlsplit, urlunsplit
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup, NavigableString, Tag

from wsj_pipeline.models import ExtractedArticle, SourceCandidate

EXTRACTOR_VERSION = "3"
NEW_YORK = ZoneInfo("America/New_York")

_EXCLUDED_TAGS = frozenset(
    {"button", "iframe", "nav", "noscript", "script", "style"}
)
_EXCLUDED_MARKER_WORDS = frozenset(
    {
        "ad",
        "advert",
        "advertisement",
        "author",
        "biography",
        "hidden",
        "login",
        "navigation",
        "promo",
        "prompt",
        "recommendation",
        "recommended",
        "related",
        "registration",
        "share",
        "sharing",
        "social",
        "subscription",
        "tracking",
    }
)
_METADATA_MARKER_WORDS = frozenset(
    {"byline", "dek", "headline", "subtitle", "summary"}
)
_MARKER_WORD_PATTERN = re.compile(r"[a-z0-9]+")
_MARKDOWN_ESCAPE_PATTERN = re.compile(r"([\\`*_[\]{}|])")


class ExtractionError(ValueError):
    """A stable, content-free article extraction failure."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


@dataclass(slots=True)
class _RenderState:
    canonical_url: str | None
    inline_image_urls: list[str] = field(default_factory=list)
    seen_image_urls: set[str] = field(default_factory=set)


def parse_wsj_timestamp(value: str | None) -> datetime | None:
    """Parse an optional ISO-8601 timestamp into aware UTC."""

    if value is None or not value.strip():
        return None
    normalized = value.strip()
    if normalized.endswith(("Z", "z")):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ExtractionError(
            "invalid_timestamp", "timestamp is not valid ISO-8601"
        ) from error
    if parsed.tzinfo is None:
        raise ExtractionError("naive_timestamp", "timestamp has no timezone")
    return parsed.astimezone(UTC)


def derive_publication_date_new_york(value: datetime) -> date:
    """Derive the America/New_York calendar date for an aware timestamp."""

    if value.tzinfo is None:
        raise ExtractionError("naive_timestamp", "timestamp has no timezone")
    return value.astimezone(NEW_YORK).date()


def derive_article_id(
    wsj_id: str | None,
    canonical_url: str | None,
    html_sha256: str,
) -> str:
    """Derive a stable namespaced identity using the approved fallback chain."""

    if wsj_id and wsj_id.strip():
        return f"wsj:{wsj_id.strip()}"
    if canonical_url and canonical_url.strip():
        normalized = _normalize_canonical_url(canonical_url)
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"url:{digest}"
    return f"html:{html_sha256}"


def extract_article(
    candidate: SourceCandidate,
    source_root: Path,
) -> ExtractedArticle:
    """Extract one source into structured metadata and editorial Markdown."""

    html_path = source_root / candidate.relative_html_path
    html_bytes = html_path.read_bytes()
    html_sha256 = hashlib.sha256(html_bytes).hexdigest()
    soup = BeautifulSoup(html_bytes, "lxml")
    metadata = _extract_metadata(soup)
    news_article = _first_typed_object(_json_ld_objects(soup), "NewsArticle")

    canonical_url = _canonical_url(soup)
    wsj_id = _first_value(metadata, "article.id")
    headline = (
        _first_value(metadata, "article.headline")
        or _json_string(news_article, "headline")
        or _first_dom_metadata_value(soup, "headline", tag_name="h1")
    )
    descriptive_metadata = _unique_texts(
        _first_value(metadata, "article.subtitle"),
        _first_value(metadata, "article.dek"),
        _first_value(metadata, "article.summary", "description"),
        _json_string(news_article, "description"),
        *_dom_metadata_values(soup, "subtitle"),
        *_dom_metadata_values(soup, "dek"),
        *_dom_metadata_values(soup, "summary"),
    )
    authors = _split_values(_first_value(metadata, "author"), separator="|")
    if not authors:
        authors = _json_ld_authors(news_article)
    if not authors:
        authors = _dom_authors(soup)

    published_at_utc = parse_wsj_timestamp(
        _first_value(metadata, "article.published", "datePublished")
        or _json_string(news_article, "datePublished")
    )
    if published_at_utc is None:
        raise ExtractionError(
            "missing_publication_timestamp",
            "article has no publication timestamp",
        )
    updated_at_utc = parse_wsj_timestamp(
        _first_value(metadata, "article.updated", "dateModified")
        or _json_string(news_article, "dateModified")
    )

    metadata_blocks = _metadata_blocks(headline, descriptive_metadata, authors)
    editorial_blocks, inline_image_urls = _extract_editorial_blocks(
        soup,
        canonical_url,
    )
    blocks = (*metadata_blocks, *editorial_blocks)
    markdown_text = "\n\n".join(blocks)
    warnings = list(candidate.warnings)
    if not markdown_text:
        warnings.append("empty_content")

    metadata_values = (
        wsj_id,
        canonical_url,
        headline,
        *descriptive_metadata,
        authors,
        published_at_utc,
        updated_at_utc,
    )
    return ExtractedArticle(
        article_id=derive_article_id(wsj_id, canonical_url, html_sha256),
        relative_html_path=candidate.relative_html_path,
        source_html_sha256=html_sha256,
        extractor_version=EXTRACTOR_VERSION,
        canonical_url=canonical_url,
        published_at_utc=published_at_utc,
        publication_date_new_york=derive_publication_date_new_york(
            published_at_utc
        ),
        updated_at_utc=updated_at_utc,
        markdown_text=markdown_text,
        editorial_character_count=len(markdown_text),
        editorial_block_count=len(blocks),
        metadata_completeness=sum(bool(value) for value in metadata_values),
        relative_header_image_path=candidate.relative_header_image_path,
        inline_image_urls=inline_image_urls,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _metadata_blocks(
    headline: str | None,
    descriptive_metadata: tuple[str, ...],
    authors: tuple[str, ...],
) -> tuple[str, ...]:
    candidates = (
        f"# {_escape_markdown(headline)}" if headline else None,
        *(_escape_markdown(value) for value in descriptive_metadata),
        _escape_markdown(_format_byline(authors)) if authors else None,
    )
    blocks: list[str] = []
    normalized_text: set[str] = set()
    for block in candidates:
        if not block:
            continue
        comparison = block.removeprefix("# ")
        if comparison in normalized_text:
            continue
        normalized_text.add(comparison)
        blocks.append(block)
    return tuple(blocks)


def _extract_editorial_blocks(
    soup: BeautifulSoup,
    canonical_url: str | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    root = soup.find("article")
    if not isinstance(root, Tag):
        root = soup.body if isinstance(soup.body, Tag) else soup
    state = _RenderState(canonical_url)
    blocks = tuple(_walk_blocks(root, state))
    return blocks, tuple(state.inline_image_urls)


def _walk_blocks(node: Tag, state: _RenderState) -> list[str]:
    blocks: list[str] = []
    for child in node.children:
        if not isinstance(child, Tag):
            continue
        if _is_excluded(child) or _is_metadata_element(child):
            continue
        if child.name in {"h2", "h3", "h4", "h5", "h6"}:
            heading = _render_inline(child, state)
            if heading:
                blocks.append(f"{'#' * int(child.name[1])} {heading}")
        elif child.name == "p":
            paragraph = _render_inline(child, state)
            if paragraph:
                blocks.append(paragraph)
        elif child.name in {"ol", "ul"}:
            rendered = _render_list(child, state)
            if rendered:
                blocks.append(rendered)
        elif child.name == "table":
            rendered = _render_table(child, state)
            if rendered:
                blocks.append(rendered)
        elif child.name == "blockquote":
            rendered = _render_quote(child, state)
            if rendered:
                blocks.append(rendered)
        elif child.name == "figure":
            blocks.extend(_render_figure(child, state))
        elif child.name == "img":
            image = _render_image(child, state)
            if image:
                blocks.append(image)
        elif _is_interactive(child):
            rendered = _render_visible_text(child)
            if rendered:
                blocks.append(_escape_markdown(rendered))
        else:
            blocks.extend(_walk_blocks(child, state))
    return blocks


def _render_list(tag: Tag, state: _RenderState) -> str:
    items: list[str] = []
    for index, item in enumerate(tag.find_all("li", recursive=False), start=1):
        text = _render_inline(item, state)
        if text:
            marker = f"{index}." if tag.name == "ol" else "-"
            items.append(f"{marker} {text}")
    return "\n".join(items)


def _render_table(tag: Tag, state: _RenderState) -> str:
    rows: list[list[str]] = []
    for row in tag.find_all("tr"):
        cells = [
            _render_inline(cell, state)
            for cell in row.find_all(["th", "td"], recursive=False)
        ]
        if cells:
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    padded_rows = [row + [""] * (width - len(row)) for row in rows]
    rendered = [_markdown_table_row(padded_rows[0])]
    rendered.append(_markdown_table_row(["---"] * width))
    rendered.extend(_markdown_table_row(row) for row in padded_rows[1:])
    return "\n".join(rendered)


def _markdown_table_row(cells: list[str]) -> str:
    return f"| {' | '.join(cells)} |"


def _render_quote(tag: Tag, state: _RenderState) -> str:
    text = _render_inline(tag, state)
    return "\n".join(f"> {line}" for line in text.splitlines()) if text else ""


def _render_figure(tag: Tag, state: _RenderState) -> list[str]:
    image = tag.find("img")
    image_markdown = _render_image(image, state) if isinstance(image, Tag) else ""

    caption_tag = tag.find("figcaption")
    caption = (
        _render_visible_text(caption_tag) if isinstance(caption_tag, Tag) else ""
    )
    credit_tag = tag.find(
        class_=lambda value: value and "credit" in _class_value(value).casefold()
    )
    credit = _render_visible_text(credit_tag) if isinstance(credit_tag, Tag) else ""
    creator_tag = tag.find(
        class_=lambda value: value and "creator" in _class_value(value).casefold()
    )
    creator = (
        _render_visible_text(creator_tag) if isinstance(creator_tag, Tag) else ""
    )
    caption_parts = list(
        dict.fromkeys(part for part in (caption, credit, creator) if part)
    )
    caption_block = " ".join(_escape_markdown(part) for part in caption_parts)
    return [block for block in (image_markdown, caption_block) if block]


def _render_image(tag: Tag, state: _RenderState) -> str:
    resolved_url = next(
        (
            resolved
            for attribute in ("src", "data-src", "data-original")
            if (
                resolved := _resolve_http_url(
                    _attribute(tag, attribute), state.canonical_url
                )
            )
        ),
        None,
    )
    if resolved_url is None or resolved_url in state.seen_image_urls:
        return ""
    state.seen_image_urls.add(resolved_url)
    state.inline_image_urls.append(resolved_url)
    alt = _escape_markdown(_clean_text(_attribute(tag, "alt")))
    return f"![{alt}]({resolved_url})"


def _render_inline(tag: Tag, state: _RenderState) -> str:
    return _clean_text(_render_inline_raw(tag, state))


def _render_inline_raw(node: Tag, state: _RenderState) -> str:
    parts: list[str] = []
    for child in node.children:
        if isinstance(child, NavigableString):
            parts.append(_escape_markdown(str(child), clean=False))
        elif isinstance(child, Tag):
            if child.name in _EXCLUDED_TAGS or _is_excluded(child):
                continue
            label = _render_inline_raw(child, state)
            if child.name == "img":
                parts.append(_render_image(child, state))
            elif child.name == "a":
                url = _resolve_http_url(
                    _attribute(child, "href"), state.canonical_url
                )
                parts.append(f"[{label}]({url})" if url and label else label)
            elif child.name == "br":
                parts.append("\n")
            else:
                parts.append(label)
    return "".join(parts)


def _render_visible_text(tag: Tag) -> str:
    parts: list[str] = []
    for child in tag.children:
        if isinstance(child, NavigableString):
            parts.append(str(child))
        elif isinstance(child, Tag) and not _is_excluded(child):
            parts.append(_render_visible_text(child))
    return _clean_text(" ".join(parts))


def _is_excluded(tag: Tag) -> bool:
    if tag.name in _EXCLUDED_TAGS or tag.has_attr("hidden"):
        return True
    if _attribute(tag, "aria-hidden").casefold() == "true":
        return True
    style = _attribute(tag, "style").replace(" ", "").casefold()
    if "display:none" in style or "visibility:hidden" in style:
        return True
    marker_words = _marker_words(tag)
    return bool(marker_words & _EXCLUDED_MARKER_WORDS)


def _is_metadata_element(tag: Tag) -> bool:
    if tag.name == "h1":
        return True
    return bool(_marker_words(tag) & _METADATA_MARKER_WORDS)


def _is_interactive(tag: Tag) -> bool:
    marker_words = _marker_words(tag)
    return (
        "interactive" in marker_words
        or _attribute(tag, "data-type") == "interactive"
    )


def _marker_words(tag: Tag) -> set[str]:
    marker = " ".join(
        (
            _attribute(tag, "id"),
            _class_value(tag.get("class")),
            _attribute(tag, "data-testid"),
            _attribute(tag, "data-inset_type"),
            _attribute(tag, "role"),
        )
    ).casefold()
    return set(_MARKER_WORD_PATTERN.findall(marker))


def _extract_metadata(soup: BeautifulSoup) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for tag in soup.find_all("meta"):
        key = _first_attribute(tag, "name", "property", "itemprop")
        content = _attribute(tag, "content")
        if key and content and key not in metadata:
            metadata[key] = _clean_text(content)
    return metadata


def _first_dom_metadata_value(
    soup: BeautifulSoup,
    marker_word: str,
    *,
    tag_name: str | None = None,
) -> str | None:
    values = _dom_metadata_values(soup, marker_word)
    if values:
        return values[0]
    if tag_name:
        tag = soup.find(tag_name)
        if isinstance(tag, Tag):
            value = _render_visible_text(tag)
            return value or None
    return None


def _dom_metadata_values(
    soup: BeautifulSoup,
    marker_word: str,
) -> tuple[str, ...]:
    values: list[str] = []
    for tag in soup.find_all(True):
        if marker_word not in _marker_words(tag):
            continue
        value = _render_visible_text(tag)
        if value and value not in values:
            values.append(value)
    return tuple(values)


def _dom_authors(soup: BeautifulSoup) -> tuple[str, ...]:
    byline = _first_dom_metadata_value(soup, "byline")
    if not byline:
        return ()
    without_prefix = re.sub(r"^by\s+", "", byline, flags=re.IGNORECASE)
    return tuple(
        value
        for item in re.split(r"\s+and\s+|\s*,\s*", without_prefix)
        if (value := _clean_text(item))
    )


def _canonical_url(soup: BeautifulSoup) -> str | None:
    for tag in soup.find_all("link"):
        rel = tag.get("rel")
        rel_values = rel if isinstance(rel, list) else str(rel or "").split()
        if "canonical" not in {str(value).casefold() for value in rel_values}:
            continue
        href = _attribute(tag, "href").strip()
        return _normalize_canonical_url(href) if href else None
    return None


def _normalize_canonical_url(value: str) -> str:
    stripped = value.strip()
    parts = urlsplit(stripped)
    scheme = parts.scheme.casefold()
    hostname = (parts.hostname or "").casefold()
    try:
        port = parts.port
    except ValueError:
        return stripped
    if port is not None and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        hostname = f"{hostname}:{port}"
    path = parts.path.rstrip("/")
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return urlunsplit((scheme, hostname, path, query, ""))


def _resolve_http_url(value: str, base_url: str | None) -> str | None:
    stripped = value.strip()
    if not stripped:
        return None
    resolved = urljoin(base_url or "", stripped)
    parts = urlsplit(resolved)
    if parts.scheme.casefold() not in {"http", "https"} or not parts.netloc:
        return None
    scheme = parts.scheme.casefold()
    hostname = (parts.hostname or "").casefold()
    try:
        port = parts.port
    except ValueError:
        return None
    if port is not None and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        hostname = f"{hostname}:{port}"
    path = quote(parts.path, safe="/:@!$&'*+,;=-._~%")
    query = quote(parts.query, safe="=&?/:;+,%@!$'*[]-._~")
    fragment = quote(parts.fragment, safe="?/:;+,%@!$&'*[]=-._~")
    return urlunsplit((scheme, hostname, path, query, fragment))


def _json_ld_objects(soup: BeautifulSoup) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    for script in soup.find_all("script", type="application/ld+json"):
        raw = script.string
        if not raw:
            continue
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        objects.extend(_flatten_json_ld(value))
    return objects


def _flatten_json_ld(value: object) -> list[dict[str, object]]:
    if isinstance(value, list):
        flattened: list[dict[str, object]] = []
        for item in value:
            flattened.extend(_flatten_json_ld(item))
        return flattened
    if not isinstance(value, dict):
        return []
    graph = value.get("@graph")
    if isinstance(graph, list):
        return [value, *_flatten_json_ld(graph)]
    return [value]


def _first_typed_object(
    objects: list[dict[str, object]],
    object_type: str,
) -> dict[str, object]:
    for item in objects:
        types = item.get("@type")
        if types == object_type or (
            isinstance(types, list) and object_type in types
        ):
            return item
    return {}


def _json_ld_authors(news_article: dict[str, object]) -> tuple[str, ...]:
    raw_authors = news_article.get("author")
    if isinstance(raw_authors, dict):
        raw_authors = [raw_authors]
    if not isinstance(raw_authors, list):
        return ()
    authors: list[str] = []
    for raw_author in raw_authors:
        if not isinstance(raw_author, dict):
            continue
        name = raw_author.get("name")
        if isinstance(name, str) and name.strip():
            authors.append(_clean_text(name))
    return tuple(authors)


def _format_byline(authors: tuple[str, ...]) -> str:
    if len(authors) == 1:
        return f"By {authors[0]}"
    return f"By {', '.join(authors[:-1])} and {authors[-1]}"


def _first_value(metadata: dict[str, str], *keys: str) -> str | None:
    for key in keys:
        value = metadata.get(key)
        if value:
            return value
    return None


def _split_values(value: str | None, *, separator: str) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(
        cleaned for item in value.split(separator) if (cleaned := _clean_text(item))
    )


def _unique_texts(*values: str | None) -> tuple[str, ...]:
    unique: list[str] = []
    for value in values:
        if value and (cleaned := _clean_text(value)) and cleaned not in unique:
            unique.append(cleaned)
    return tuple(unique)


def _json_string(value: dict[str, object], key: str) -> str | None:
    item = value.get(key)
    return _clean_text(item) if isinstance(item, str) and item.strip() else None


def _first_attribute(tag: Tag, *keys: str) -> str:
    for key in keys:
        value = _attribute(tag, key)
        if value:
            return value
    return ""


def _attribute(tag: Tag, key: str) -> str:
    value = tag.get(key)
    return value if isinstance(value, str) else ""


def _class_value(value: object) -> str:
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return value if isinstance(value, str) else ""


def _clean_text(value: str) -> str:
    return unicodedata.normalize("NFC", " ".join(value.split()))


def _escape_markdown(value: str, *, clean: bool = True) -> str:
    normalized = _clean_text(value) if clean else unicodedata.normalize("NFC", value)
    return _MARKDOWN_ESCAPE_PATTERN.sub(r"\\\1", normalized)
