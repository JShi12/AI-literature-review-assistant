"""LLM-as-judge checks for what deterministic metrics can't see: is the content actually supported?

Three judges, each with a versioned rubric (the version is part of the recording key, so editing a
rubric means fresh judgments, never stale replays):

- claim grounding:       is each extracted claim supported by the chunk it came from?
- synthesis faithfulness: is every statement backed by the cited claims, and is it the requested type?
- review citations:      do each sentence's cited claims support the sentence?

The judge sees only the evidence the pipeline gave the model, never the rest of the paper, so
"unsupported" means "not supported by what it was allowed to rely on". Judges run at temperature 0
and put their reasoning before the verdict; the synthesis and review judges check statement by statement
and their verdicts are derived from those checks. Calibrate before trusting them: python -m evals.calibrate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from lit_review_assistant.llm.client import StructuredLLM

JUDGE_TEMPERATURE = 0.0
DEFAULT_JUDGE_MODEL = "gpt-4.1"

Support = Literal["supported", "partially_supported", "unsupported"]
Faithfulness = Literal["faithful", "partially_faithful", "unfaithful"]


@dataclass(frozen=True)
class CitedClaim:
    paper: str
    text: str


# --- Claim grounding ------------------------------------------------------------------------------

CLAIM_GROUNDING_VERSION = "judge.claim_grounding.v1"
CLAIM_GROUNDING_RUBRIC = """You check whether claims extracted from a paper are grounded in their source passage.

Judge each numbered claim ONLY against the passage. Outside knowledge does not count, even if true.
- supported: everything the claim asserts is stated in, or directly entailed by, the passage.
- partially_supported: the core is supported, but the claim adds, drops, or changes something material:
  a detail not in the passage, an overgeneralization, a removed hedge or qualifier, a changed scope.
- unsupported: the claim contradicts the passage or asserts something the passage does not say.
Paraphrase is fine; check meaning, numbers, entities, direction of comparisons, and hedges.
Return one verdict per claim, using the claim's number, with a brief reason before the verdict.
"""


class ClaimJudgment(BaseModel):
    claim_number: int
    reason: str
    verdict: Support


class ClaimGroundingJudgments(BaseModel):
    judgments: list[ClaimJudgment]


def judge_claims(passage: str, claims: Sequence[str], llm: StructuredLLM) -> list[Support | None]:
    """Judge every claim against its passage in one call; None where the judge skipped a claim."""
    if not claims:
        return []
    numbered = "\n".join(f"{number}. {claim}" for number, claim in enumerate(claims, start=1))
    result = llm.parse(
        text_format=ClaimGroundingJudgments,
        prompt_version=CLAIM_GROUNDING_VERSION,
        instructions=CLAIM_GROUNDING_RUBRIC,
        input_text=f"Passage:\n{passage}\n\nClaims:\n{numbered}",
        temperature=JUDGE_TEMPERATURE,
    )
    by_number = {judgment.claim_number: judgment.verdict for judgment in result.parsed.judgments}
    return [by_number.get(number) for number in range(1, len(claims) + 1)]


# --- Shared: statement-level checks --------------------------------------------------------------

# Judges first list each statement with its verdict; the overall verdict is derived from those checks
# in code. Asking for one holistic verdict made the judge lenient on exactly the subtle overstatements
# (an extra list item, an intensifier, a misattributed finding) that matter most.
UNSUPPORTED_DETAILS = """A statement is NOT supported if any part of it is missing from the cited claims, including:
- an extra item, number, or detail the claims don't mention;
- an intensifier or generalization the claims don't make ("widely used", "exact", "always", "studies show"
  when only one paper says it);
- a finding attributed to the wrong paper, or presented as shared when only one paper states it;
- a cause, mechanism, or link between findings that no claim states.
Judge ONLY against the cited claims (each labelled with its paper). Outside knowledge does not count."""


class StatementCheck(BaseModel):
    statement: str
    supporting_claims: list[int]
    supported: bool


def support_from_checks(checks: Sequence[StatementCheck]) -> Support:
    supported = sum(check.supported for check in checks)
    if checks and supported == len(checks):
        return "supported"
    return "unsupported" if supported == 0 else "partially_supported"


# --- Synthesis faithfulness -----------------------------------------------------------------------

SYNTHESIS_VERSION = "judge.synthesis.v2"
SYNTHESIS_RUBRIC = f"""You check a literature-review synthesis against the evidence claims it cites.

Work statement by statement. Split the title and body into their individual factual statements: each item
in a list is its own statement, and so is any statement that something holds across papers or that papers
agree or disagree. For each, give the numbers of the cited claims that back it and whether it is fully
supported.

{UNSUPPORTED_DETAILS}

Then judge the synthesis type. First write type_evidence: the specific cited claims that make it this type,
or "none". type_appropriate is true only if those claims satisfy the definition:
- theme: the same idea appears in claims from at least two different papers.
- contradiction: two cited claims actually disagree on a stated dimension (different designs or settings
  are not a disagreement).
- gap: a missing or unresolved area that cited limitations or evidence point to.
- method_comparison: methods compared on a dimension the claims report for each of them.
- insight: a non-obvious takeaway that follows from the cited evidence, not a restatement of one paper's
  description of itself.
"""


class SynthesisJudgment(BaseModel):
    statements: list[StatementCheck]
    type_evidence: str
    type_appropriate: bool

    @property
    def faithfulness(self) -> Faithfulness:
        """All statements supported -> faithful; at least half unsupported -> unfaithful; else partial."""
        unsupported = sum(not check.supported for check in self.statements)
        if not self.statements or unsupported * 2 >= len(self.statements):
            return "unfaithful"
        return "faithful" if unsupported == 0 else "partially_faithful"

    @property
    def unsupported_statements(self) -> list[str]:
        return [check.statement for check in self.statements if not check.supported]


def judge_synthesis(
    synthesis_type: str, title: str, body: str, cited: Sequence[CitedClaim], llm: StructuredLLM
) -> SynthesisJudgment:
    evidence = "\n".join(f"{number}. [{claim.paper}] {claim.text}" for number, claim in enumerate(cited, start=1))
    return llm.parse(
        text_format=SynthesisJudgment,
        prompt_version=SYNTHESIS_VERSION,
        instructions=SYNTHESIS_RUBRIC,
        input_text=(
            f"Synthesis type: {synthesis_type}\nTitle: {title}\nBody: {body}\n\nCited claims:\n{evidence or '(none)'}"
        ),
        temperature=JUDGE_TEMPERATURE,
    ).parsed


# --- Review citations -----------------------------------------------------------------------------

REVIEW_CITATIONS_VERSION = "judge.review_citations.v2"
REVIEW_CITATIONS_RUBRIC = f"""You check the citations in a literature-review draft.

For each numbered sentence you get the evidence claims it cites (each labelled with its paper). Split the
sentence into its individual assertions -- each listed item, number, qualifier, comparison, or causal link
is its own assertion -- and mark each as supported only if the sentence's own cited claims back it. Other
sentences' citations do not count. Give the numbers of the sentence's cited claims that back each assertion.

{UNSUPPORTED_DETAILS}

Return one entry per sentence, using the sentence's number.
"""


class SentenceCheck(BaseModel):
    sentence_number: int
    assertions: list[StatementCheck]

    @property
    def verdict(self) -> Support:
        return support_from_checks(self.assertions)


class ReviewCitationJudgments(BaseModel):
    judgments: list[SentenceCheck]


def judge_review_citations(
    sentences: Sequence[tuple[str, Sequence[CitedClaim]]], llm: StructuredLLM
) -> list[Support | None]:
    """Judge each (sentence, cited claims) pair in one call; None where the judge skipped a sentence."""
    if not sentences:
        return []
    blocks = []
    for number, (sentence, cited) in enumerate(sentences, start=1):
        evidence = "\n".join(f"   {index}. [{claim.paper}] {claim.text}" for index, claim in enumerate(cited, start=1))
        blocks.append(f"{number}. Sentence: {sentence}\n   Cited claims:\n{evidence}")
    result = llm.parse(
        text_format=ReviewCitationJudgments,
        prompt_version=REVIEW_CITATIONS_VERSION,
        instructions=REVIEW_CITATIONS_RUBRIC,
        input_text="\n\n".join(blocks),
        temperature=JUDGE_TEMPERATURE,
    )
    by_number = {judgment.sentence_number: judgment.verdict for judgment in result.parsed.judgments}
    return [by_number.get(number) for number in range(1, len(sentences) + 1)]
