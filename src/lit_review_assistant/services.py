"""Application-level services: PDF ingestion orchestration, hashing, section matching, and metadata backfill."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from lit_review_assistant.db.models import Chunk, Claim, Page, Paper, Section, Synthesis, SynthesisClaim
from lit_review_assistant.pipeline.chunking import chunk_pages_with_sections
from lit_review_assistant.pipeline.pdf import (
    PaperMetadata,
    extract_pages,
    extract_pdf_metadata,
    infer_paper_metadata_from_name,
)
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
    """Ingest a PDF into pages, sections, and chunks, deduplicating by file hash.

    If a paper with the same SHA-256 content hash already exists, the existing record is
    returned unchanged rather than re-ingesting the file.
    """
    path = Path(pdf_path)
    file_hash = sha256_file(path)
    existing = session.scalar(select(Paper).where(Paper.file_sha256 == file_hash))
    if existing:
        return existing

    pdf_metadata = extract_pdf_metadata(path)
    filename_metadata = infer_paper_metadata_from_name(file_name or path.name)
    pages = extract_pages(path)
    detected_sections = detect_sections(pages)
    chunks = chunk_pages_with_sections(pages, detected_sections, max_chars=chunk_max_chars, overlap=chunk_overlap)

    paper = Paper(
        paper_key=next_paper_key(session),
        title=pdf_metadata.title or filename_metadata.title or path.stem,
        authors=pdf_metadata.authors or filename_metadata.authors or [],
        year=pdf_metadata.year or filename_metadata.year,
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


def select_chunks_for_claim_extraction(session: Session, paper_id: str | None, limit: int) -> list[Chunk]:
    """Pick up to `limit` chunks that don't have claims yet, round-robin across papers.

    A plain "order by paper, then page" selection exhausts one paper's chunks before ever
    reaching the next, so a modest limit across a multi-paper corpus can silently only ever
    touch the alphabetically-first paper. Round-robining keeps every paper represented.
    """
    query = (
        select(Chunk)
        .where(~Chunk.claims.any())
        .order_by(Chunk.paper_id.asc(), Chunk.page_start.asc(), Chunk.start_char.asc())
    )
    if paper_id is not None:
        query = query.where(Chunk.paper_id == paper_id)
    candidates = session.scalars(query).all()
    return round_robin_by_paper(candidates, limit)


def round_robin_by_paper(chunks: Sequence[Chunk], limit: int) -> list[Chunk]:
    """Interleave chunks by paper_id so a limited selection spreads evenly across papers.

    Assumes `chunks` is already ordered per-paper (e.g. by page/offset) -- that per-paper
    order is preserved, only the across-paper interleaving changes.
    """
    by_paper: dict[str, list[Chunk]] = {}
    paper_order: list[str] = []
    for chunk in chunks:
        if chunk.paper_id not in by_paper:
            by_paper[chunk.paper_id] = []
            paper_order.append(chunk.paper_id)
        by_paper[chunk.paper_id].append(chunk)

    selected: list[Chunk] = []
    index = 0
    while len(selected) < limit:
        progressed = False
        for paper_id in paper_order:
            bucket = by_paper[paper_id]
            if index < len(bucket):
                selected.append(bucket[index])
                progressed = True
                if len(selected) == limit:
                    break
        if not progressed:
            break
        index += 1
    return selected


def backfill_paper_metadata(session: Session, upload_dir: str | Path = "data/uploads") -> int:
    base_dir = Path(upload_dir)
    updated = 0
    papers = session.scalars(select(Paper).order_by(Paper.created_at.asc())).all()
    for paper in papers:
        metadata = infer_paper_metadata_from_name(paper.file_name)
        pdf_path = base_dir / paper.file_name
        if pdf_path.exists():
            pdf_metadata = extract_pdf_metadata(pdf_path)
            metadata = _merge_metadata(pdf_metadata, metadata)

        changed = False
        if metadata.authors and not paper.authors:
            paper.authors = metadata.authors
            changed = True
        if metadata.year is not None and paper.year is None:
            paper.year = metadata.year
            changed = True
        if metadata.title and _title_is_missing_or_filename_derived(paper.title, paper.file_name):
            paper.title = metadata.title
            changed = True
        if changed:
            updated += 1

    session.flush()
    return updated


def _merge_metadata(primary: PaperMetadata, fallback: PaperMetadata) -> PaperMetadata:
    """Merge two metadata records, preferring values from `primary` and falling back to `fallback`."""
    return type(primary)(
        title=primary.title or fallback.title,
        authors=primary.authors or fallback.authors,
        year=primary.year or fallback.year,
    )


def _title_is_missing_or_filename_derived(title: str | None, file_name: str) -> bool:
    if not title:
        return True
    normalized_title = title.replace("_", " ").strip()
    normalized_stem = Path(file_name).stem.replace("_", " ").strip()
    return normalized_title == normalized_stem


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
