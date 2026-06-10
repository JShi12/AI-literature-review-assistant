from __future__ import annotations

from lit_review_assistant.pipeline.support import ClaimSupport, calculate_support_counts, sentence_support_status


def test_support_counts_deduplicate_claims_and_papers() -> None:
    counts = calculate_support_counts(
        [
            ClaimSupport(claim_id="CL001", paper_id="P001"),
            ClaimSupport(claim_id="CL001", paper_id="P001"),
            ClaimSupport(claim_id="CL002", paper_id="P001"),
            ClaimSupport(claim_id="CL003", paper_id="P002"),
        ]
    )

    assert counts.supporting_claims == 3
    assert counts.supporting_papers == 2


def test_sentence_support_requires_at_least_one_claim() -> None:
    assert sentence_support_status(["CL001", "CL002"])
    assert not sentence_support_status([])
