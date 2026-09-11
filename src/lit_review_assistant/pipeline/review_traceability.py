"""Helpers for mapping review draft sentences back to their supporting claim citations."""

from __future__ import annotations

from dataclasses import dataclass

from lit_review_assistant.schemas import ReviewSentencePayload


@dataclass(frozen=True)
class SentenceTrace:
    section_title: str
    sentence_index: int
    sentence_text: str
    supporting_claim_ids: list[str]
    is_supported: bool


def normalize_sentence_support(payload: ReviewSentencePayload) -> SentenceTrace:
    """Deduplicate and sort a sentence's supporting claim IDs, deriving its supported status."""
    claim_ids = sorted(set(payload.supporting_claim_ids))
    return SentenceTrace(
        section_title=payload.section_title,
        sentence_index=payload.sentence_index,
        sentence_text=payload.sentence_text,
        supporting_claim_ids=claim_ids,
        is_supported=bool(claim_ids),
    )


def unsupported_sentence_indexes(sentences: list[ReviewSentencePayload]) -> list[int]:
    """Return the indexes of sentences with no supporting claims."""
    return [sentence.sentence_index for sentence in sentences if not normalize_sentence_support(sentence).is_supported]


def claim_support_map(sentences: list[ReviewSentencePayload]) -> dict[int, list[str]]:
    """Map each sentence index to its deduplicated list of supporting claim IDs."""
    return {
        sentence.sentence_index: normalize_sentence_support(sentence).supporting_claim_ids for sentence in sentences
    }
