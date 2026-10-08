"""Pydantic schemas used as OpenAI structured-output response formats."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Confidence = float


ClaimType = Literal["finding", "method", "dataset", "metric", "limitation", "future_work", "background", "other"]


class ClaimCandidate(BaseModel):
    """One claim as the LLM returns it: the claim plus the verbatim passage it rests on.

    The model is not asked for character offsets (it can't count them reliably); they're computed by
    locating source_quote in the chunk text.
    """

    claim_text: str
    claim_type: ClaimType
    normalized_text: str | None = None
    source_quote: str
    confidence: Confidence = Field(ge=0, le=1)


class ExtractedClaim(BaseModel):
    """A claim with its resolved location in the paper, ready to persist."""

    claim_text: str
    claim_type: ClaimType
    normalized_text: str | None = None
    paper_id: str
    chunk_id: str
    section_id: str | None = None
    page: int
    start_char: int
    end_char: int
    confidence: Confidence = Field(ge=0, le=1)


class GeneratedSynthesis(BaseModel):
    synthesis_type: Literal["theme", "contradiction", "gap", "method_comparison", "insight"]
    title: str
    body: str
    supporting_claim_ids: list[str]
    supporting_papers: int = Field(ge=0)
    supporting_claims: int = Field(ge=0)
    confidence: Confidence = Field(ge=0, le=1)


class ReviewSentencePayload(BaseModel):
    section_title: str
    sentence_index: int = Field(ge=0)
    sentence_text: str
    supporting_claim_ids: list[str]
    is_supported: bool


class ReviewDraftPayload(BaseModel):
    title: str
    outline: list[str]
    markdown: str
    sentences: list[ReviewSentencePayload]
    confidence: Confidence = Field(ge=0, le=1)
