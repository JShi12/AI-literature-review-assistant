"""Generate claim-backed syntheses: themes, contradictions, gaps, method comparisons, and insights."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from lit_review_assistant.db.models import Claim, Synthesis, SynthesisClaim
from lit_review_assistant.llm.client import LLMResult, OpenAIStructuredLLM, StructuredLLM, create_llm_run
from lit_review_assistant.llm.embeddings import embed_and_persist_syntheses
from lit_review_assistant.pipeline.support import ClaimSupport, calculate_support_counts
from lit_review_assistant.schemas import GeneratedSynthesis

PROMPT_VERSION = "synthesis.v2"
DEFAULT_TEMPERATURE = 0.2
SynthesisType = Literal["theme", "contradiction", "gap", "method_comparison", "insight"]


class GeneratedSynthesesBatch(BaseModel):
    syntheses: list[GeneratedSynthesis]


def generate_syntheses(
    session: Session,
    claim_ids: list[str],
    synthesis_type: SynthesisType,
    llm: StructuredLLM | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
) -> list[Synthesis]:
    """Generate and persist syntheses of a single type from the given claims."""
    claims = session.scalars(select(Claim).where(Claim.id.in_(claim_ids))).all()
    if not claims:
        return []

    result = request_syntheses(claims, synthesis_type, llm=llm, temperature=temperature)
    run = create_llm_run(session, result)
    syntheses = persist_generated_syntheses(session, result.parsed.syntheses, run.id)
    embed_and_persist_syntheses(session, syntheses)
    session.flush()
    return syntheses


def request_syntheses(
    claims: Sequence[Claim],
    synthesis_type: SynthesisType,
    llm: StructuredLLM | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
) -> LLMResult[GeneratedSynthesesBatch]:
    """Ask the LLM for syntheses without persisting anything (shared with the eval harness)."""
    llm = llm or OpenAIStructuredLLM()
    return llm.parse(
        text_format=GeneratedSynthesesBatch,
        prompt_version=PROMPT_VERSION,
        instructions=SYNTHESIS_INSTRUCTIONS,
        input_text=build_synthesis_input(claims, synthesis_type),
        temperature=temperature,
    )


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
    # Papers get short labels (A, B, ...) so the model can tell them apart and attribute findings
    # correctly; with only paper UUIDs it routinely credited one paper's result to another.
    labels: dict[str, str] = {}
    paper_lines = []
    for claim in claims:
        if claim.paper_id not in labels:
            labels[claim.paper_id] = _paper_label(len(labels))
            paper_lines.append(f"[{labels[claim.paper_id]}] {_paper_title(claim)}")
    claim_lines = []
    for claim in claims:
        text = claim.normalized_text or claim.claim_text
        claim_lines.append(
            f"- claim_id={claim.id}; paper={labels[claim.paper_id]}; type={claim.claim_type}; "
            f"confidence={claim.confidence}; text={text}"
        )
    return (
        f"Create {synthesis_type} syntheses from the claims below, or none if the claims don't support one.\n"
        "Every synthesis must cite the supporting_claim_ids it actually uses.\n\n"
        "Papers:\n" + "\n".join(paper_lines) + "\n\nClaims:\n" + "\n".join(claim_lines)
    )


def _paper_label(index: int) -> str:
    return chr(ord("A") + index) if index < 26 else f"P{index + 1}"


def _paper_title(claim: Claim) -> str:
    paper = getattr(claim, "paper", None)
    return (paper.title or paper.paper_key) if paper is not None else claim.paper_id


SYNTHESIS_INSTRUCTIONS = """You synthesize academic evidence claims.

Rules:
- Create only syntheses supported by the provided claims. If the claims don't support any synthesis of the
  requested type, return an empty list -- never force one.
- synthesis_type must be the requested type, and must genuinely fit its definition:
  - theme: an idea that recurs in claims from at least two different papers.
  - contradiction: two cited claims that actually disagree; state the exact dimension. Different designs,
    settings, or focuses are not a contradiction.
  - gap: a missing or unresolved area that cited limitations, missing evidence, or conflicts point to.
  - method_comparison: methods compared on a dimension the claims report for each of them.
  - insight: a non-obvious takeaway that follows from the claims, not a restatement of one paper's
    description of its own work.
- Every statement in the title and body must be backed by the cited claims:
  - Do not add details, items, numbers, or qualifiers the claims don't state ("widely used", "exact",
    "always", "studies show" when only one paper says it).
  - Attribute each finding to the paper whose claim states it; say "both papers" or "across papers" only
    when claims from each of those papers state it.
  - Do not assert causes, mechanisms, or links between findings that no claim states.
- Cite every claim the synthesis relies on, and no claim that doesn't support it.
- Include confidence for every synthesis.
"""
