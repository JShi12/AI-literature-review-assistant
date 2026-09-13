"""Helpers for counting distinct supporting claims and papers behind a synthesis."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClaimSupport:
    claim_id: str
    paper_id: str


@dataclass(frozen=True)
class SupportCounts:
    supporting_papers: int
    supporting_claims: int


def calculate_support_counts(claims: list[ClaimSupport]) -> SupportCounts:
    """Count the distinct papers and claims backing a synthesis."""
    return SupportCounts(
        supporting_papers=len({claim.paper_id for claim in claims}),
        supporting_claims=len({claim.claim_id for claim in claims}),
    )


def sentence_support_status(claim_ids: list[str]) -> bool:
    """Return whether a sentence has at least one supporting claim."""
    return len(set(claim_ids)) > 0
