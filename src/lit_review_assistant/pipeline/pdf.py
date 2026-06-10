from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str


def extract_pages(pdf_path: str | Path) -> list[PageText]:
    """Extract page text exactly once so downstream offsets refer to stored text."""
    path = Path(pdf_path)
    pages: list[PageText] = []
    with fitz.open(path) as doc:
        for index, page in enumerate(doc, start=1):
            pages.append(PageText(page_number=index, text=page.get_text("text")))
    return pages
