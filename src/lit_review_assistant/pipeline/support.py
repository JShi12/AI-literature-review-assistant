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
    return SupportCounts(
        supporting_papers=len({claim.paper_id for claim in claims}),
        supporting_claims=len({claim.claim_id for claim in claims}),
    )


def sentence_support_status(claim_ids: list[str]) -> bool:
    return len(set(claim_ids)) > 0
