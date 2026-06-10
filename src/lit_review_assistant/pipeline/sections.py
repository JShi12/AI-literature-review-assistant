from __future__ import annotations

import re
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
    "findings": "results",
    "discussion": "discussion",
    "limitations": "limitations",
    "limitation": "limitations",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
}

HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?(?P<title>"
    + "|".join(re.escape(title) for title in sorted(SECTION_TYPES, key=len, reverse=True))
    + r")\s*$",
    re.IGNORECASE,
)


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


def normalize_section_title(title: str) -> str:
    key = re.sub(r"\s+", " ", title.strip().lower())
    return SECTION_TYPES.get(key, "other")


def detect_sections(pages: list[PageText]) -> list[DetectedSection]:
    hits: list[HeadingHit] = []
    for page in pages:
        cursor = 0
        for line in page.text.splitlines(keepends=True):
            plain = line.strip()
            match = HEADING_RE.match(plain)
            if match:
                title = match.group("title").strip()
                hits.append(
                    HeadingHit(
                        title=title,
                        normalized_type=normalize_section_title(title),
                        page_number=page.page_number,
                        start_char=cursor,
                    )
                )
            cursor += len(line)

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
        end_char = next_hit.start_char if next_hit and next_hit.page_number == hit.page_number else len(page_by_number[page_end].text)
        sections.append(
            DetectedSection(
                title=hit.title,
                normalized_type=hit.normalized_type,
                page_start=hit.page_number,
                page_end=page_end,
                start_char=hit.start_char,
                end_char=end_char,
                confidence=0.75,
            )
        )
    return sections
