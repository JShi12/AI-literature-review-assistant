"""PyMuPDF-based page text and metadata extraction, plus filename-derived metadata fallback."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import fitz


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str


@dataclass(frozen=True)
class PaperMetadata:
    title: str | None = None
    authors: list[str] | None = None
    year: int | None = None


def extract_pages(pdf_path: str | Path) -> list[PageText]:
    """Extract page text exactly once so downstream offsets refer to stored text."""
    path = Path(pdf_path)
    pages: list[PageText] = []
    with fitz.open(path) as doc:
        for index, page in enumerate(doc, start=1):
            pages.append(PageText(page_number=index, text=page.get_text("text")))
    return pages


def extract_pdf_metadata(pdf_path: str | Path) -> PaperMetadata:
    """Combine PDF document metadata with metadata inferred from the first page's text."""
    path = Path(pdf_path)
    with fitz.open(path) as doc:
        metadata = doc.metadata or {}
        first_page_text = doc[0].get_text("text") if len(doc) else ""

    document_metadata = PaperMetadata(
        title=clean_metadata_text(metadata.get("title")),
        authors=parse_authors(metadata.get("author")),
        year=parse_year(" ".join(str(value) for value in metadata.values() if value)),
    )
    text_metadata = infer_paper_metadata_from_first_page(first_page_text)
    return PaperMetadata(
        title=best_title(document_metadata.title, text_metadata.title),
        authors=document_metadata.authors or text_metadata.authors,
        year=text_metadata.year or document_metadata.year,
    )


def infer_paper_metadata_from_first_page(text: str) -> PaperMetadata:
    lines = [
        line for line in (clean_metadata_text(normalize_pdf_text(raw_line)) for raw_line in text.splitlines()) if line
    ]
    if not lines:
        return PaperMetadata()

    year = parse_year(text)
    author_index = next((index for index, line in enumerate(lines[:12]) if looks_like_author_line(line)), None)
    if author_index is None or author_index == 0:
        return PaperMetadata(year=year)

    title = clean_metadata_text(" ".join(lines[:author_index]))
    author_lines: list[str] = []
    for line in lines[author_index : author_index + 4]:
        if author_lines and looks_like_affiliation_or_body_start(line):
            break
        if looks_like_affiliation_or_body_start(line):
            break
        author_lines.append(line)
        if " and " in line.casefold() and not line.rstrip().endswith(","):
            break

    return PaperMetadata(title=title, authors=parse_authors(" ".join(author_lines)), year=year)


def infer_paper_metadata_from_name(name: str | Path) -> PaperMetadata:
    """Infer title, authors, and year from a filename, as a fallback when PDF metadata is missing."""
    raw_name = str(name)
    suffix = Path(raw_name).suffix.lower()
    stem = Path(raw_name).stem if suffix in {".pdf", ".txt", ".md"} else raw_name
    normalized = clean_metadata_text(stem.replace("_", " "))
    if not normalized:
        return PaperMetadata()

    year_match = re.search(r"(?:^|[\s-])((?:19|20)\d{2})(?:[\s-]|$)", normalized)
    if year_match is None:
        return PaperMetadata(title=normalized)

    year = int(year_match.group(1))
    author_part = normalized[: year_match.start(1)].strip(" -_,")
    title_part = normalized[year_match.end(1) :].strip(" -_,")
    return PaperMetadata(
        title=title_part or normalized,
        authors=parse_authors(author_part),
        year=year,
    )


def clean_metadata_text(value: object) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value).strip())
    return text or None


def normalize_pdf_text(text: str) -> str:
    replacements = {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\u2212": "-",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def looks_like_author_line(line: str) -> bool:
    lowered = line.casefold()
    if looks_like_affiliation_or_body_start(line):
        return False
    if len(line) > 250:
        return False
    if re.search(r"\babstract\b|\bkeywords\b|\bintroduction\b", lowered):
        return False
    if re.search(r"\b[A-Z]\.\s*(?:[A-Z]\.\s*)?[A-Z][A-Za-z-]+", line):
        return True
    return _looks_like_name_list(line)


def _looks_like_name_list(line: str) -> bool:
    """Detect a comma/'and'-separated list of proper names (e.g. full-name author bylines).

    A bare comma is too weak a signal on its own -- ordinary sentences have commas too -- so this
    requires every comma/'and'-separated segment to look like a short run of capitalized name words.
    """
    cleaned = re.sub(r"[*†‡§¶#0-9]+", "", line).strip(" ,")
    if not cleaned:
        return False
    tokens = [token.strip() for token in re.split(r"\s*(?:,|;|\band\b|&)\s*", cleaned, flags=re.IGNORECASE)]
    tokens = [token for token in tokens if token]
    if len(tokens) < 2:
        return False
    for token in tokens:
        words = token.split()
        if not (1 <= len(words) <= 3):
            return False
        if not all(re.fullmatch(r"[A-Z][A-Za-z'-]*\.?", word) for word in words):
            return False
    return True


def looks_like_affiliation_or_body_start(line: str) -> bool:
    lowered = line.casefold()
    affiliation_markers = [
        "abstract",
        "department",
        "university",
        "institute",
        "college",
        "school",
        "state key laboratory",
        "cnrs",
        "supporting information",
        "received",
        "revised",
        "keywords",
    ]
    return any(marker in lowered for marker in affiliation_markers) or line.strip() in {"*", "■"}


def parse_authors(value: object) -> list[str]:
    text = clean_metadata_text(value)
    if not text:
        return []

    text = normalize_pdf_text(text)
    text = re.sub(r"[*†‡§¶#]+", "", text)
    text = re.sub(r"(?<=[A-Za-z])\d+\b", "", text)
    text = re.sub(r"([A-Z]\.)(?=[A-Z][a-z])", r"\1 ", text)
    text = text.strip(" -_,")
    if not text:
        return []
    if re.search(r"\bet\s+al\.?\b", text, flags=re.IGNORECASE):
        return [text]

    parts = [part.strip(" -_,") for part in re.split(r";|,|\s+(?:and|&)\s+", text) if part.strip(" -_,")]
    if len(parts) > 1:
        return parts
    if "," not in text and "." not in text:
        tokens = text.split()
        if 1 < len(tokens) <= 4 and all(token[:1].isupper() for token in tokens):
            return tokens
    return [text]


def parse_year(value: object) -> int | None:
    text = clean_metadata_text(value)
    if not text:
        return None
    match = re.search(r"\b((?:19|20)\d{2})\b", text)
    return int(match.group(1)) if match else None


def best_title(document_title: str | None, text_title: str | None) -> str | None:
    if text_title and (not document_title or looks_like_internal_pdf_title(document_title)):
        return text_title
    return document_title or text_title


_KNOWN_JUNK_TITLES = {
    "no job name",
    "untitled",
    "untitled document",
    "untitled-1",
    "(no title)",
    "adobe pdf",
}
_JUNK_TITLE_PREFIXES = ("microsoft word - ",)


def looks_like_internal_pdf_title(title: str) -> bool:
    normalized = title.strip()
    lowered = normalized.casefold()
    if lowered in _KNOWN_JUNK_TITLES:
        return True
    if lowered.startswith(_JUNK_TITLE_PREFIXES):
        return True
    if re.search(r"\d+\.\.\d+", normalized):
        return True
    return bool(re.fullmatch(r"[a-z]{1,4}\d+[a-z]?(?:\s+\d+\.\.\d+)?", normalized, flags=re.IGNORECASE))
