"""Versioned extraction of normalized records from archived WSJ HTML."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup, Tag

from wsj_pipeline.models import ExtractedArticle, PublicationDates, SourceCandidate

EXTRACTOR_VERSION = "2"
NEW_YORK = ZoneInfo("America/New_York")
_WORD_PATTERN = re.compile(r"\b[\w\u2019'-]+\b", re.UNICODE)
_EXCLUDED_ANCESTOR_TOKENS = (
    "author",
    "caption",
    "related",
    "recommend",
    "advert",
)


class ExtractionError(ValueError):
    """A stable, content-free article extraction failure."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def parse_wsj_timestamp(value: str | None) -> datetime | None:
    """Parse a WSJ timestamp into an aware UTC datetime."""

    if not value:
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


def derive_publication_dates(published_at_utc: datetime) -> PublicationDates:
    """Derive unambiguous UTC and America/New_York calendar fields."""

    if published_at_utc.tzinfo is None:
        raise ExtractionError("naive_timestamp", "timestamp has no timezone")
    utc_value = published_at_utc.astimezone(UTC)
    new_york_value = utc_value.astimezone(NEW_YORK)
    return PublicationDates(
        published_at_utc=utc_value,
        publication_date_utc=utc_value.date(),
        published_at_new_york=new_york_value,
        publication_date_new_york=new_york_value.date(),
        publication_year_ny=new_york_value.year,
        publication_month_ny=new_york_value.month,
    )


def extract_article(
    candidate: SourceCandidate,
    source_root: Path,
) -> ExtractedArticle:
    """Extract one candidate without modifying its raw files."""

    html_path = source_root / candidate.relative_html_path
    html_bytes = html_path.read_bytes()
    source_hash = hashlib.sha256(html_bytes).hexdigest()
    soup = BeautifulSoup(html_bytes, "lxml")
    metadata = _extract_metadata(soup)
    json_ld = _json_ld_objects(soup)
    news_article = _first_typed_object(json_ld, "NewsArticle")
    image_object = _first_typed_object(json_ld, "ImageObject")

    paragraphs = _extract_body_paragraphs(soup)

    published = parse_wsj_timestamp(
        _first_value(metadata, "article.published", "datePublished")
        or _json_string(news_article, "datePublished")
    )
    if published is None:
        raise ExtractionError(
            "missing_publication_timestamp",
            "article has no publication timestamp",
        )
    dates = derive_publication_dates(published)

    body_text = "\n\n".join(paragraphs)
    body_word_count = len(_WORD_PATTERN.findall(body_text))
    warnings = list(candidate.warnings)
    if not paragraphs:
        warnings.append("empty_body")
    declared_word_count = _parse_int(_first_value(metadata, "article:word_count"))
    if declared_word_count is not None and abs(
        declared_word_count - body_word_count
    ) > max(10, declared_word_count // 10):
        warnings.append("word_count_mismatch")

    authors = _split_values(_first_value(metadata, "author"), separator="|")
    if not authors:
        authors = _json_ld_authors(news_article)

    image_creator = _image_creator(image_object)
    image_media_type = (
        mimetypes.guess_type(candidate.relative_image_path)[0]
        if candidate.relative_image_path
        else None
    )
    headline = _first_value(metadata, "article.headline") or _json_string(
        news_article, "headline"
    )
    canonical_url = _canonical_url(soup)
    core_values = (
        _first_value(metadata, "article.id"),
        canonical_url,
        headline,
        _first_value(metadata, "article.summary"),
        authors,
        _first_value(metadata, "article.section"),
        published,
        paragraphs,
        candidate.relative_image_path,
    )

    return ExtractedArticle(
        relative_html_path=candidate.relative_html_path,
        source_html_sha256=source_hash,
        extractor_version=EXTRACTOR_VERSION,
        wsj_article_id=_first_value(metadata, "article.id"),
        canonical_url=canonical_url,
        headline=headline,
        original_headline=_first_value(metadata, "article.origheadline"),
        summary=_first_value(metadata, "article.summary", "description"),
        authors=authors,
        keywords=_split_values(_first_value(metadata, "keywords"), separator=","),
        section=_first_value(metadata, "article.section"),
        page=_first_value(metadata, "article.page"),
        article_type=_first_value(metadata, "article.type"),
        access=_first_value(metadata, "article.access"),
        published_at_utc=dates.published_at_utc,
        publication_date_utc=dates.publication_date_utc,
        published_at_new_york=dates.published_at_new_york,
        publication_date_new_york=dates.publication_date_new_york,
        publication_year_ny=dates.publication_year_ny,
        publication_month_ny=dates.publication_month_ny,
        updated_at_utc=parse_wsj_timestamp(
            _first_value(metadata, "article.updated", "dateModified")
        ),
        created_at_utc=parse_wsj_timestamp(
            _first_value(metadata, "article.created", "dateCreated")
        ),
        last_published_at_utc=parse_wsj_timestamp(
            _first_value(metadata, "dateLastPubbed")
        ),
        body_paragraphs=paragraphs,
        body_text=body_text,
        body_word_count=body_word_count,
        relative_image_path=candidate.relative_image_path,
        image_present=candidate.relative_image_path is not None,
        image_media_type=image_media_type,
        image_size_bytes=candidate.image_size,
        image_caption=_json_string(image_object, "caption"),
        image_credit=_json_string(image_object, "creditText"),
        image_creator=image_creator,
        image_alt=_first_value(metadata, "twitter:image:alt", "og:image:alt"),
        image_width=_json_int(image_object, "width"),
        image_height=_json_int(image_object, "height"),
        metadata_completeness=sum(bool(value) for value in core_values),
        warnings=tuple(dict.fromkeys(warnings)),
        raw_metadata_json=json.dumps(
            metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def _extract_metadata(soup: BeautifulSoup) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for tag in soup.find_all("meta"):
        key = tag.get("name") or tag.get("property") or tag.get("itemprop")
        content = tag.get("content")
        if isinstance(key, str) and isinstance(content, str) and key not in metadata:
            metadata[key] = _clean_text(content)
    return metadata


def _canonical_url(soup: BeautifulSoup) -> str | None:
    tag = soup.find("link", rel=lambda value: value and "canonical" in value)
    if not isinstance(tag, Tag):
        return None
    href = tag.get("href")
    if not isinstance(href, str) or not href.strip():
        return None
    parts = urlsplit(href.strip())
    if not parts.scheme or not parts.netloc:
        return href.strip()
    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            parts.path,
            parts.query,
            "",
        )
    )


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
    objects: list[dict[str, object]], object_type: str
) -> dict[str, object]:
    for item in objects:
        types = item.get("@type")
        if types == object_type or (isinstance(types, list) and object_type in types):
            return item
    return {}


def _extract_body_paragraphs(soup: BeautifulSoup) -> tuple[str, ...]:
    root = soup.find("article")
    if not isinstance(root, Tag):
        root = soup
    paragraphs: list[str] = []
    for paragraph in root.select('p[data-type="paragraph"]'):
        if _is_excluded_paragraph(paragraph, root):
            continue
        text = _clean_text(paragraph.get_text(" ", strip=True))
        if text:
            paragraphs.append(text)
    return tuple(paragraphs)


def _is_excluded_paragraph(paragraph: Tag, root: Tag) -> bool:
    for ancestor in paragraph.parents:
        if ancestor is root:
            break
        if not isinstance(ancestor, Tag):
            continue
        if ancestor.name in {"aside", "figure", "figcaption"}:
            return True
        if ancestor.has_attr("data-inset_type"):
            return True
        marker = " ".join(
            [
                ancestor.get("id", ""),
                " ".join(ancestor.get("class", [])),
                ancestor.get("data-testid", ""),
            ]
        ).casefold()
        if any(token in marker for token in _EXCLUDED_ANCESTOR_TOKENS):
            return True
    return False


def _clean_text(value: str) -> str:
    return unicodedata.normalize("NFC", " ".join(value.split()))


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


def _json_string(value: dict[str, object], key: str) -> str | None:
    item = value.get(key)
    return _clean_text(item) if isinstance(item, str) and item.strip() else None


def _json_int(value: dict[str, object], key: str) -> int | None:
    item = value.get(key)
    return int(item) if isinstance(item, (int, float)) else None


def _json_ld_authors(news_article: dict[str, object]) -> tuple[str, ...]:
    raw_authors = news_article.get("author")
    if isinstance(raw_authors, dict):
        raw_authors = [raw_authors]
    if not isinstance(raw_authors, list):
        return ()
    authors: list[str] = []
    for raw_author in raw_authors:
        if isinstance(raw_author, dict):
            name = raw_author.get("name")
            if isinstance(name, str) and name.strip():
                authors.append(_clean_text(name))
    return tuple(authors)


def _image_creator(image_object: dict[str, object]) -> str | None:
    creator = image_object.get("creator")
    if isinstance(creator, dict):
        name = creator.get("name")
        if isinstance(name, str) and name.strip():
            return _clean_text(name)
    if isinstance(creator, str) and creator.strip():
        return _clean_text(creator)
    return None


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
