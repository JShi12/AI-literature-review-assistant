from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Confidence = float


class ExtractedClaim(BaseModel):
    claim_text: str
    claim_type: Literal["finding", "method", "dataset", "metric", "limitation", "future_work", "background", "other"]
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
