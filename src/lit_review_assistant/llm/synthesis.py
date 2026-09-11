"""Generate claim-backed syntheses: themes, contradictions, gaps, method comparisons, and insights."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from lit_review_assistant.db.models import Claim, Synthesis, SynthesisClaim
from lit_review_assistant.llm.client import OpenAIStructuredLLM, StructuredLLM, create_llm_run
from lit_review_assistant.pipeline.support import ClaimSupport, calculate_support_counts
from lit_review_assistant.schemas import GeneratedSynthesis

PROMPT_VERSION = "synthesis.v1"
SynthesisType = Literal["theme", "contradiction", "gap", "method_comparison", "insight"]


class GeneratedSynthesesBatch(BaseModel):
    syntheses: list[GeneratedSynthesis]


def generate_syntheses(
    session: Session,
    claim_ids: list[str],
    synthesis_type: SynthesisType,
    llm: StructuredLLM | None = None,
    temperature: float = 0.2,
) -> list[Synthesis]:
    """Generate and persist syntheses of a single type from the given claims."""
    claims = session.scalars(select(Claim).where(Claim.id.in_(claim_ids))).all()
    if not claims:
        return []

    llm = llm or OpenAIStructuredLLM()
    result = llm.parse(
        text_format=GeneratedSynthesesBatch,
        prompt_version=PROMPT_VERSION,
        instructions=SYNTHESIS_INSTRUCTIONS,
        input_text=build_synthesis_input(claims, synthesis_type),
        temperature=temperature,
    )
    run = create_llm_run(session, result)
    syntheses = persist_generated_syntheses(session, result.parsed.syntheses, run.id)
    session.flush()
    return syntheses


def persist_generated_syntheses(
    session: Session,
    generated_syntheses: list[GeneratedSynthesis],
    run_id: str,
) -> list[Synthesis]:
    persisted: list[Synthesis] = []
    for generated in generated_syntheses:
        claims = session.scalars(select(Claim).where(Claim.id.in_(generated.supporting_claim_ids))).all()
        counts = calculate_support_counts(
            [ClaimSupport(claim_id=claim.id, paper_id=claim.paper_id) for claim in claims]
        )
        synthesis = Synthesis(
            synthesis_type=generated.synthesis_type,
            title=generated.title,
            body=generated.body,
            supporting_papers=counts.supporting_papers,
            supporting_claims=counts.supporting_claims,
            confidence=Decimal(str(generated.confidence)),
            run_id=run_id,
        )
        session.add(synthesis)
        session.flush()
        for claim in claims:
            session.add(SynthesisClaim(synthesis_id=synthesis.id, claim_id=claim.id))
        persisted.append(synthesis)
    session.flush()
    return persisted


def build_synthesis_input(claims: Sequence[Claim], synthesis_type: SynthesisType) -> str:
    claim_lines = []
    for claim in claims:
        text = claim.normalized_text or claim.claim_text
        claim_lines.append(
            f"- claim_id={claim.id}; paper_id={claim.paper_id}; type={claim.claim_type}; "
            f"confidence={claim.confidence}; text={text}"
        )
    return (
        f"Create {synthesis_type} syntheses from the claims below.\n"
        "Every synthesis must cite the supporting_claim_ids it actually uses.\n\n" + "\n".join(claim_lines)
    )


SYNTHESIS_INSTRUCTIONS = """You synthesize academic evidence claims.

Rules:
- Create only syntheses supported by the provided claims.
- synthesis_type must be one of: theme, contradiction, gap, method_comparison, insight.
- Do not cite claims that do not support the synthesis.
- For contradictions, describe the exact disagreement dimension.
- For gaps, tie the gap to observed limitations, missing evidence, or unresolved conflicts.
- Include confidence for every synthesis.
"""
