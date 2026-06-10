from __future__ import annotations

from dataclasses import dataclass

from lit_review_assistant.pipeline.pdf import PageText
from lit_review_assistant.pipeline.sections import DetectedSection


@dataclass(frozen=True)
class TextChunk:
    page_start: int
    page_end: int
    start_char: int
    end_char: int
    text: str
    section_title: str | None = None
    section_type: str | None = None


def chunk_page_text(page: PageText, max_chars: int = 1_500, overlap: int = 150) -> list[TextChunk]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be non-negative and smaller than max_chars")

    text = page.text
    if not text:
        return []

    chunks: list[TextChunk] = []
    start = 0
    while start < len(text):
        hard_end = min(start + max_chars, len(text))
        end = _prefer_boundary(text, start, hard_end)
        if end <= start:
            end = hard_end
        chunk_text = text[start:end]
        if chunk_text.strip():
            chunks.append(
                TextChunk(
                    page_start=page.page_number,
                    page_end=page.page_number,
                    start_char=start,
                    end_char=end,
                    text=chunk_text,
                )
            )
        if end >= len(text):
            break
        start = max(end - overlap, 0)
    return chunks


def chunk_pages_with_sections(
    pages: list[PageText],
    sections: list[DetectedSection],
    max_chars: int = 1_500,
    overlap: int = 150,
) -> list[TextChunk]:
    section_lookup = {(section.page_start, section.start_char): section for section in sections}
    chunks: list[TextChunk] = []
    for page in pages:
        page_chunks = chunk_page_text(page, max_chars=max_chars, overlap=overlap)
        for chunk in page_chunks:
            section = _section_for_chunk(chunk, sections)
            chunks.append(
                TextChunk(
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    text=chunk.text,
                    section_title=section.title if section else None,
                    section_type=section.normalized_type if section else None,
                )
            )
    return chunks


def _prefer_boundary(text: str, start: int, hard_end: int) -> int:
    if hard_end == len(text):
        return hard_end
    window = text[start:hard_end]
    for delimiter in ("\n\n", ". ", "\n", " "):
        index = window.rfind(delimiter)
        if index > max(80, len(window) // 2):
            return start + index + len(delimiter)
    return hard_end


def _section_for_chunk(chunk: TextChunk, sections: list[DetectedSection]) -> DetectedSection | None:
    candidates = [
        section
        for section in sections
        if section.page_start <= chunk.page_start <= section.page_end
    ]
    if not candidates:
        return None
    same_page = [
        section
        for section in candidates
        if section.page_start == chunk.page_start and section.start_char <= chunk.start_char
    ]
    if same_page:
        return max(same_page, key=lambda section: section.start_char)
    return max(candidates, key=lambda section: section.page_start)
