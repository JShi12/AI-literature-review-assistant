"""Extract source-grounded claims from paper chunks using structured LLM output."""

from __future__ import annotations

import logging
from decimal import Decimal

from pydantic import BaseModel
from sqlalchemy.orm import Session

from lit_review_assistant.db.models import Chunk, Claim
from lit_review_assistant.llm.client import LLMResult, OpenAIStructuredLLM, StructuredLLM, create_llm_run
from lit_review_assistant.llm.embeddings import embed_and_persist_claims
from lit_review_assistant.pipeline.quotes import find_quote_span, strip_control_chars
from lit_review_assistant.schemas import ClaimCandidate, ExtractedClaim

logger = logging.getLogger(__name__)

PROMPT_VERSION = "claims.v2"
DEFAULT_TEMPERATURE = 0.1


class ExtractedClaimsBatch(BaseModel):
    claims: list[ClaimCandidate]


def extract_claims_for_chunk(
    session: Session,
    chunk: Chunk,
    llm: StructuredLLM | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
) -> list[Claim]:
    """Extract and persist source-grounded claims from a single chunk's text."""
    result = request_claims(chunk, llm=llm, temperature=temperature)
    located, unlocated = locate_claims(result.parsed.claims, chunk)
    if unlocated:
        logger.warning(
            "Dropped %d of %d claim(s) from chunk %s: source quote not found in the chunk text",
            len(unlocated),
            len(result.parsed.claims),
            chunk.id,
        )
    run = create_llm_run(session, result)
    claims = persist_extracted_claims(session, located, run.id, chunk=chunk)
    embed_and_persist_claims(session, claims)
    session.flush()
    return claims


def request_claims(
    chunk: Chunk,
    llm: StructuredLLM | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
) -> LLMResult[ExtractedClaimsBatch]:
    """Ask the LLM for a chunk's claims without persisting anything (shared with the eval harness)."""
    llm = llm or OpenAIStructuredLLM()
    return llm.parse(
        text_format=ExtractedClaimsBatch,
        prompt_version=PROMPT_VERSION,
        instructions=CLAIM_EXTRACTION_INSTRUCTIONS,
        input_text=build_claim_extraction_input(chunk),
        temperature=temperature,
    )


def locate_claims(candidates: list[ClaimCandidate], chunk: Chunk) -> tuple[list[ExtractedClaim], list[ClaimCandidate]]:
    """Resolve each candidate's page and page-level offsets from its source quote.

    Returns (located claims, candidates whose quote couldn't be found in the chunk). The latter are
    dropped rather than stored with a made-up location: a quote that isn't in the source is also a
    grounding red flag. Chunks never span pages, so offsets are relative to chunk.page_start.
    """
    located: list[ExtractedClaim] = []
    unlocated: list[ClaimCandidate] = []
    for candidate in candidates:
        span = find_quote_span(chunk.text, candidate.source_quote)
        if span is None:
            unlocated.append(candidate)
            continue
        located.append(
            ExtractedClaim(
                claim_text=candidate.claim_text,
                claim_type=candidate.claim_type,
                normalized_text=candidate.normalized_text,
                paper_id=chunk.paper_id,
                chunk_id=chunk.id,
                section_id=chunk.section_id,
                page=chunk.page_start,
                start_char=chunk.start_char + span[0],
                end_char=chunk.start_char + span[1],
                confidence=candidate.confidence,
            )
        )
    return located, unlocated


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
            validate_claim_location_against_chunk(extracted, chunk)
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
        "Return only claims that are directly supported by the text, each with its verbatim source_quote.\n\n"
        f"section_type: {section_label}\n\n"
        "Chunk text:\n"
        # PDF math fonts leave control characters (e.g. \r) in the text; asked to quote verbatim, the model
        # can degenerate into emitting control characters until it runs out of tokens. Quotes are still
        # located in the unmodified chunk text, since find_quote_span ignores control characters.
        f"{strip_control_chars(chunk.text)}"
    )


def validate_claim_location_against_chunk(extracted: ExtractedClaim, chunk: Chunk) -> None:
    if extracted.start_char < chunk.start_char or extracted.end_char > chunk.end_char:
        raise ValueError("Extracted claim offsets must fall within the source chunk offsets.")
    if not (chunk.page_start <= extracted.page <= chunk.page_end):
        raise ValueError("Extracted claim page must fall within the source chunk page range.")


CLAIM_EXTRACTION_INSTRUCTIONS = """You extract evidence claims from academic paper chunks.

Rules:
- Extract atomic claims only; one finding, method, dataset, metric, limitation, or future-work point per claim.
- Preserve the source meaning. Do not add interpretation.
- Include an optional normalized_text when a concise canonical form is obvious.
- source_quote must be copied verbatim from the chunk text: the shortest contiguous passage that states \
the claim, usually one sentence or less. Do not paraphrase it, fix its typos, or join separate passages.
- Include confidence for every claim.
- If the chunk has no direct claim, return an empty claims list.
"""
