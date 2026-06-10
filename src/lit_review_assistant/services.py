from __future__ import annotations

import hashlib
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from lit_review_assistant.db.models import Claim, Chunk, Page, Paper, Section, Synthesis, SynthesisClaim
from lit_review_assistant.pipeline.chunking import chunk_pages_with_sections
from lit_review_assistant.pipeline.pdf import extract_pages
from lit_review_assistant.pipeline.sections import detect_sections


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def next_paper_key(session: Session) -> str:
    count = session.scalar(select(func.count(Paper.id))) or 0
    return f"P{count + 1:03d}"


def ingest_pdf(
    session: Session,
    pdf_path: str | Path,
    file_name: str | None = None,
    chunk_max_chars: int = 1_500,
    chunk_overlap: int = 150,
) -> Paper:
    path = Path(pdf_path)
    file_hash = sha256_file(path)
    existing = session.scalar(select(Paper).where(Paper.file_sha256 == file_hash))
    if existing:
        return existing

    pages = extract_pages(path)
    detected_sections = detect_sections(pages)
    chunks = chunk_pages_with_sections(pages, detected_sections, max_chars=chunk_max_chars, overlap=chunk_overlap)

    paper = Paper(
        paper_key=next_paper_key(session),
        title=path.stem,
        authors=[],
        file_name=file_name or path.name,
        file_sha256=file_hash,
    )
    session.add(paper)
    session.flush()

    for page in pages:
        session.add(Page(paper_id=paper.id, page_number=page.page_number, text=page.text))

    section_models: list[Section] = []
    for section in detected_sections:
        model = Section(
            paper_id=paper.id,
            title=section.title,
            normalized_type=section.normalized_type,
            page_start=section.page_start,
            page_end=section.page_end,
            start_char=section.start_char,
            end_char=section.end_char,
            confidence=section.confidence,
        )
        session.add(model)
        section_models.append(model)
    session.flush()

    for chunk in chunks:
        section_id = _match_section_id(chunk.section_title, chunk.section_type, chunk.page_start, section_models)
        session.add(
            Chunk(
                paper_id=paper.id,
                section_id=section_id,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                text=chunk.text,
            )
        )

    session.flush()
    return paper


def recalculate_synthesis_support_counts(session: Session, synthesis_id: str) -> Synthesis:
    synthesis = session.get(Synthesis, synthesis_id)
    if synthesis is None:
        raise ValueError(f"Synthesis not found: {synthesis_id}")

    rows = session.execute(
        select(SynthesisClaim.claim_id, Claim.paper_id)
        .join(Claim, Claim.id == SynthesisClaim.claim_id)
        .where(SynthesisClaim.synthesis_id == synthesis_id)
    ).all()

    claim_ids = {row[0] for row in rows}
    paper_ids = {row[1] for row in rows if row[1] is not None}
    synthesis.supporting_claims = len(claim_ids)
    synthesis.supporting_papers = len(paper_ids)
    session.flush()
    return synthesis


def _match_section_id(
    section_title: str | None,
    section_type: str | None,
    page_start: int,
    sections: list[Section],
) -> str | None:
    if not section_title and not section_type:
        return None
    candidates = [
        section
        for section in sections
        if section.page_start <= page_start <= section.page_end
        and (section.title == section_title or section.normalized_type == section_type)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda section: section.start_char or 0).id
