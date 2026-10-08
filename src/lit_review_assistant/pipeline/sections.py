"""Heuristics for detecting academic paper sections (abstract, methods, results, ...) from page text.

Works on plain extracted text (no font information), so headings are recognized by shape:

- a known section name on its own line ("Introduction", "References"), optionally numbered;
- a numbered heading on one line ("3.1 Input Output Representation", "S.2 Driving Systems");
- a section number on its own line followed by the title on the next ("5.1" / "Data Selection"),
  which is how PyMuPDF extracts many LaTeX-typeset papers;
- a Roman-numbered all-caps heading ("II. MATERIALS AND METHODS", IEEE/ACS style);
- a black-square bullet heading ("■RESULTS AND DISCUSSION", ACS journals);
- an inline "Abstract: ..." paragraph opener.

Custom titles are typed by keyword ("... Evaluation" -> results); a numbered subsection without a
telling keyword inherits its parent section's type, and an unrecognized top-level section defaults
to methods -- in practice that's what custom body sections between the introduction and results are.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from lit_review_assistant.pipeline.pdf import PageText

SECTION_TYPES = {
    "abstract": "abstract",
    "introduction": "introduction",
    "related work": "related_work",
    "background": "related_work",
    "methods": "methods",
    "method": "methods",
    "methodology": "methods",
    "materials and methods": "methods",
    "experiments": "methods",
    "experimental setup": "methods",
    "results": "results",
    "results and discussion": "results",
    "findings": "results",
    "discussion": "discussion",
    "limitations": "limitations",
    "limitation": "limitations",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "references": "other",
    "bibliography": "other",
    "acknowledgments": "other",
    "acknowledgements": "other",
    "acknowledgment": "other",
    "acknowledgement": "other",
    "appendix": "other",
    "supplementary material": "other",
    "supplementary materials": "other",
}

# Keyword rules for custom titles, checked in order; the first match wins.
TITLE_KEYWORDS: list[tuple[str, str]] = [
    (r"\babstract\b", "abstract"),
    (r"\bintroduction\b", "introduction"),
    (r"\b(related work|prior work|background|literature)\b", "related_work"),
    (r"\bconclu", "conclusion"),
    (r"\b(limitation|failure)", "limitations"),
    (r"\b(experimental setup|implementation|experiments?|methods?|methodology|materials)\b", "methods"),
    (r"\b(results?|evaluation|ablation|tests?|findings)\b", "results"),
    (r"\bdiscussion\b", "discussion"),
    (
        r"\b(references|bibliography|acknowledge?ments?|appendix|supplementary|associated content"
        r"|author information|funding|conflicts? of interest|notes)\b",
        "other",
    ),
]

SECTION_NUMBER = r"(?:[A-Z]\.)?\d{1,2}(?:\.\d{1,2}){0,3}"
HEADING_RE = re.compile(
    rf"^\s*(?:{SECTION_NUMBER}\.?\s+)?(?P<title>"
    + "|".join(re.escape(title) for title in sorted(SECTION_TYPES, key=len, reverse=True))
    + r")\s*$",
    re.IGNORECASE,
)
NUMBER_ONLY_RE = re.compile(rf"^(?P<number>{SECTION_NUMBER})\.?$")
NUMBERED_HEADING_RE = re.compile(rf"^(?P<number>{SECTION_NUMBER})\.?\s+(?P<title>\S.*)$")
INLINE_ABSTRACT_RE = re.compile(r"^abstract\s*[:.—–-]\s*\S", re.IGNORECASE)
# Roman numerals only with an all-caps title: "V. Koltun" in a byline must not look like a heading.
ROMAN_HEADING_RE = re.compile(r"^(?P<number>[IVX]{1,5})\.\s+(?P<title>[A-Z][A-Z0-9 ,&:/\-–]+)$")
BULLET_RE = re.compile(r"^[■▪]\s*")

# Section numbers above this are far more likely to be table cells or page numbers than headings.
MAX_TOP_LEVEL_NUMBER = 20
# A line appearing on this many pages is a running header/footer, never a heading.
RUNNING_HEADER_MIN_PAGES = 3


@dataclass(frozen=True)
class DetectedSection:
    title: str
    normalized_type: str
    page_start: int
    page_end: int
    start_char: int
    end_char: int
    confidence: float


@dataclass(frozen=True)
class HeadingHit:
    title: str
    normalized_type: str
    page_number: int
    start_char: int
    confidence: float = 0.75


def normalize_section_title(title: str) -> str:
    key = re.sub(r"\s+", " ", title.strip().lower())
    return SECTION_TYPES.get(key, "other")


def classify_title(title: str, parent_type: str | None = None) -> str:
    """Type a heading by its title: exact known name, then keywords, then the parent's type, then methods."""
    key = re.sub(r"\s+", " ", title.strip().lower())
    if key in SECTION_TYPES:
        return SECTION_TYPES[key]
    for pattern, section_type in TITLE_KEYWORDS:
        if re.search(pattern, key):
            return section_type
    return parent_type or "methods"


def looks_like_title(text: str) -> bool:
    letters = sum(char.isalpha() for char in text)
    return (
        0 < len(text) <= 90
        and 1 <= len(text.split()) <= 12
        and text[0].isupper()
        and not text.endswith((".", ",", ";", ":"))
        and letters >= 0.6 * len(text.replace(" ", ""))
    )


def _looks_like_body_or_heading(line: str | None) -> bool:
    if line is None:
        return False
    if len(line.split()) >= 5:
        return True
    numbered = NUMBER_ONLY_RE.match(line) or NUMBERED_HEADING_RE.match(line)
    return bool(numbered and _number_is_plausible(numbered.group("number")))


def _number_is_plausible(number: str) -> bool:
    top_level = re.sub(r"^[A-Z]\.", "", number).split(".")[0]
    return 0 < int(top_level) <= MAX_TOP_LEVEL_NUMBER


def _running_lines(pages: list[PageText]) -> set[str]:
    counts: Counter[str] = Counter()
    for page in pages:
        counts.update({line.strip() for line in page.text.splitlines() if line.strip()})
    return {line for line, count in counts.items() if count >= RUNNING_HEADER_MIN_PAGES}


def _find_heading_hits(pages: list[PageText]) -> list[HeadingHit]:
    running = _running_lines(pages)
    hits: list[HeadingHit] = []
    # Section number -> type, so a subsection that doesn't name its own role inherits its parent's.
    type_by_number: dict[str, str] = {}

    for page in pages:
        lines = page.text.splitlines(keepends=True)
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line))
        stripped = [line.strip() for line in lines]

        for index, line in enumerate(stripped):
            following = next((line for line in stripped[index + 1 :] if line), None)
            hit: HeadingHit | None = None
            number: str | None = None
            bulleted = bool(BULLET_RE.match(line))
            plain = BULLET_RE.sub("", line)

            known = HEADING_RE.match(plain)
            roman = ROMAN_HEADING_RE.match(plain)
            numbered = NUMBERED_HEADING_RE.match(plain)
            number_only = NUMBER_ONLY_RE.match(plain)
            if known:
                title = known.group("title").strip()
                hit = HeadingHit(title, normalize_section_title(title), page.page_number, offsets[index], 0.75)
                number = numbered.group("number") if numbered else None
            elif INLINE_ABSTRACT_RE.match(plain):
                hit = HeadingHit("Abstract", "abstract", page.page_number, offsets[index], 0.7)
            elif bulleted and looks_like_title(plain):
                hit = HeadingHit(plain, "", page.page_number, offsets[index], 0.7)
            elif roman and _looks_like_body_or_heading(following):
                number = roman.group("number")
                hit = HeadingHit(roman.group("title").strip(), "", page.page_number, offsets[index], 0.6)
            elif (
                numbered
                and _number_is_plausible(numbered.group("number"))
                and looks_like_title(numbered.group("title"))
                and _looks_like_body_or_heading(following)
            ):
                number = numbered.group("number")
                title = numbered.group("title").strip()
                hit = HeadingHit(title, "", page.page_number, offsets[index], 0.6)
            elif number_only and _number_is_plausible(number_only.group("number")) and index + 1 < len(lines):
                title = stripped[index + 1]
                after_title = next((line for line in stripped[index + 2 :] if line), None)
                if title and looks_like_title(title) and _looks_like_body_or_heading(after_title):
                    number = number_only.group("number")
                    # The heading starts at the title line, which is the line that names the section.
                    hit = HeadingHit(title, "", page.page_number, offsets[index + 1], 0.6)

            heading_line = stripped[index + 1] if hit is not None and hit.start_char != offsets[index] else line
            if hit is None or heading_line in running:
                continue
            parent_type = None
            if number is not None and "." in re.sub(r"^[A-Z]\.", "", number):
                parent_type = type_by_number.get(number.rsplit(".", 1)[0])
            section_type = hit.normalized_type or classify_title(hit.title, parent_type)
            if number is not None:
                type_by_number[number] = section_type
            hits.append(HeadingHit(hit.title, section_type, hit.page_number, hit.start_char, hit.confidence))

    # A number-only line and its title can both match (e.g. "1" then "Introduction"); keep one per position.
    unique: dict[tuple[int, int], HeadingHit] = {}
    for hit in hits:
        unique.setdefault((hit.page_number, hit.start_char), hit)
    return list(unique.values())


def detect_sections(pages: list[PageText]) -> list[DetectedSection]:
    """Detect academic sections across pages from heading-like lines (see the module docstring)."""
    hits = _find_heading_hits(pages)

    if not hits:
        return [
            DetectedSection(
                title="Full Text",
                normalized_type="other",
                page_start=pages[0].page_number if pages else 1,
                page_end=pages[-1].page_number if pages else 1,
                start_char=0,
                end_char=len(pages[-1].text) if pages else 0,
                confidence=0.3,
            )
        ]

    sections: list[DetectedSection] = []
    page_by_number = {page.page_number: page for page in pages}
    for index, hit in enumerate(hits):
        next_hit = hits[index + 1] if index + 1 < len(hits) else None
        page_end = next_hit.page_number if next_hit else pages[-1].page_number
        end_char = (
            next_hit.start_char
            if next_hit and next_hit.page_number == hit.page_number
            else len(page_by_number[page_end].text)
        )
        sections.append(
            DetectedSection(
                title=hit.title,
                normalized_type=hit.normalized_type,
                page_start=hit.page_number,
                page_end=page_end,
                start_char=hit.start_char,
                end_char=end_char,
                confidence=hit.confidence,
            )
        )
    return sections
