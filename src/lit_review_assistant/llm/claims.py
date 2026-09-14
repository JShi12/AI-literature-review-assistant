"""Extract source-grounded claims from paper chunks using structured LLM output."""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel
from sqlalchemy.orm import Session

from lit_review_assistant.db.models import Chunk, Claim
from lit_review_assistant.llm.client import OpenAIStructuredLLM, StructuredLLM, create_llm_run
from lit_review_assistant.llm.embeddings import embed_and_persist_claims
from lit_review_assistant.schemas import ExtractedClaim

PROMPT_VERSION = "claims.v1"


class ExtractedClaimsBatch(BaseModel):
    claims: list[ExtractedClaim]


def extract_claims_for_chunk(
    session: Session,
    chunk: Chunk,
    llm: StructuredLLM | None = None,
    temperature: float = 0.1,
) -> list[Claim]:
    """Extract and persist source-grounded claims from a single chunk's text."""
    llm = llm or OpenAIStructuredLLM()
    result = llm.parse(
        text_format=ExtractedClaimsBatch,
        prompt_version=PROMPT_VERSION,
        instructions=CLAIM_EXTRACTION_INSTRUCTIONS,
        input_text=build_claim_extraction_input(chunk),
        temperature=temperature,
    )
    run = create_llm_run(session, result)
    claims = persist_extracted_claims(session, result.parsed.claims, run.id, chunk=chunk)
    embed_and_persist_claims(session, claims)
    session.flush()
    return claims


def persist_extracted_claims(
    session: Session,
    extracted_claims: list[ExtractedClaim],
    run_id: str,
    chunk: Chunk | None = None,
) -> list[Claim]:
    claims: list[Claim] = []
    for extracted in extracted_claims:
        paper_id = extracted.paper_id
        chunk_id = extracted.chunk_id
        section_id = extracted.section_id
        if chunk is not None:
            _validate_claim_location_against_chunk(extracted, chunk)
            paper_id = chunk.paper_id
            chunk_id = chunk.id
            section_id = chunk.section_id
        claim = Claim(
            claim_text=extracted.claim_text,
            claim_type=extracted.claim_type,
            normalized_text=extracted.normalized_text,
            paper_id=paper_id,
            chunk_id=chunk_id,
            section_id=section_id,
            page=extracted.page,
            start_char=extracted.start_char,
            end_char=extracted.end_char,
            confidence=Decimal(str(extracted.confidence)),
            run_id=run_id,
        )
        session.add(claim)
        claims.append(claim)
    session.flush()
    return claims


def build_claim_extraction_input(chunk: Chunk) -> str:
    section_label = chunk.section.normalized_type if chunk.section else "unknown"
    return (
        "Extract source-grounded atomic claims from this paper chunk.\n"
        "Return only claims that are directly supported by the text. Use page-level character offsets.\n\n"
        f"paper_id: {chunk.paper_id}\n"
        f"chunk_id: {chunk.id}\n"
        f"section_id: {chunk.section_id or ''}\n"
        f"section_type: {section_label}\n"
        f"page_start: {chunk.page_start}\n"
        f"page_end: {chunk.page_end}\n"
        f"chunk_start_char: {chunk.start_char}\n"
        f"chunk_end_char: {chunk.end_char}\n\n"
        "Chunk text:\n"
        f"{chunk.text}"
    )


def _validate_claim_location_against_chunk(extracted: ExtractedClaim, chunk: Chunk) -> None:
    if extracted.start_char < chunk.start_char or extracted.end_char > chunk.end_char:
        raise ValueError("Extracted claim offsets must fall within the source chunk offsets.")
    if not (chunk.page_start <= extracted.page <= chunk.page_end):
        raise ValueError("Extracted claim page must fall within the source chunk page range.")


CLAIM_EXTRACTION_INSTRUCTIONS = """You extract evidence claims from academic paper chunks.

Rules:
- Extract atomic claims only; one finding, method, dataset, metric, limitation, or future-work point per claim.
- Preserve the source meaning. Do not add interpretation.
- Include an optional normalized_text when a concise canonical form is obvious.
- Each claim must include paper_id, chunk_id, section_id, page, start_char, end_char, and confidence.
- Copy paper_id, chunk_id, and section_id exactly from the chunk metadata. Do not shorten, rewrite, or invent IDs.
- Character offsets must refer to the page text coordinates provided by the chunk metadata.
- If the chunk has no direct claim, return an empty claims list.
"""
