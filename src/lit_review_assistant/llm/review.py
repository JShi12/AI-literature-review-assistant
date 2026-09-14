"""Generate Markdown literature review drafts with sentence-level citation traceability back to source claims."""

from __future__ import annotations

import re
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from lit_review_assistant.db.models import Claim, ReviewDraft, ReviewSentence, ReviewSentenceClaim, Synthesis
from lit_review_assistant.llm.client import OpenAIStructuredLLM, StructuredLLM, create_llm_run
from lit_review_assistant.pipeline.pdf import infer_paper_metadata_from_name
from lit_review_assistant.pipeline.review_traceability import normalize_sentence_support
from lit_review_assistant.schemas import ReviewDraftPayload, ReviewSentencePayload

PROMPT_VERSION = "review.v1"
UUID_PATTERN = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")


def generate_review_draft(
    session: Session,
    topic: str,
    synthesis_ids: list[str],
    llm: StructuredLLM | None = None,
    temperature: float = 0.2,
) -> ReviewDraft | None:
    """Generate, citation-clean, and persist a review draft from the given syntheses."""
    syntheses = session.scalars(select(Synthesis).where(Synthesis.id.in_(synthesis_ids))).all()
    if not syntheses:
        return None

    llm = llm or OpenAIStructuredLLM()
    result = llm.parse(
        text_format=ReviewDraftPayload,
        prompt_version=PROMPT_VERSION,
        instructions=REVIEW_INSTRUCTIONS,
        input_text=build_review_input(topic, syntheses),
        temperature=temperature,
    )
    extra_claims = find_claims_for_citation_cleanup(session, result.parsed, syntheses)
    parsed = apply_academic_citations(result.parsed, syntheses, extra_claims=extra_claims)
    run = create_llm_run(session, result)
    draft = persist_review_draft(session, parsed, run.id)
    session.flush()
    return draft


def persist_review_draft(session: Session, payload: ReviewDraftPayload, run_id: str) -> ReviewDraft:
    draft = ReviewDraft(
        title=payload.title,
        outline=payload.outline,
        markdown=payload.markdown,
        confidence=Decimal(str(payload.confidence)),
        run_id=run_id,
    )
    session.add(draft)
    session.flush()

    for sentence_payload in payload.sentences:
        sentence = persist_review_sentence(session, draft.id, sentence_payload)
        for claim_id in sorted(set(sentence_payload.supporting_claim_ids)):
            if session.get(Claim, claim_id) is not None:
                session.add(ReviewSentenceClaim(review_sentence_id=sentence.id, claim_id=claim_id))
    session.flush()
    return draft


def persist_review_sentence(
    session: Session,
    review_draft_id: str,
    payload: ReviewSentencePayload,
) -> ReviewSentence:
    normalized = normalize_sentence_support(payload)
    sentence = ReviewSentence(
        review_draft_id=review_draft_id,
        section_title=normalized.section_title,
        sentence_index=normalized.sentence_index,
        sentence_text=normalized.sentence_text,
        is_supported=normalized.is_supported,
    )
    session.add(sentence)
    session.flush()
    return sentence


def build_review_input(topic: str, syntheses: Sequence[Synthesis]) -> str:
    citation_by_paper_id = build_citation_map(syntheses)
    lines = [
        f"Topic: {topic}",
        "",
        "Paper references:",
    ]
    for paper_id, citation_number in citation_by_paper_id.items():
        lines.append(f"[{citation_number}] {format_reference(paper_id, syntheses)}")

    lines.extend(["", "Use these syntheses and their claim-backed evidence:"])
    for synthesis in syntheses:
        claim_refs = [
            f"{claim.id} -> [{citation_by_paper_id[claim.paper_id]}]"
            for claim in synthesis.claims
            if claim.paper_id in citation_by_paper_id
        ]
        lines.append(
            f"- synthesis_id={synthesis.id}; type={synthesis.synthesis_type}; title={synthesis.title}; "
            f"supporting_claim_refs={claim_refs}; body={synthesis.body}"
        )
    return "\n".join(lines)


def find_claims_for_citation_cleanup(
    session: Session,
    payload: ReviewDraftPayload,
    syntheses: Sequence[Synthesis],
) -> list[Claim]:
    text_parts = [payload.markdown]
    text_parts.extend(sentence.sentence_text for sentence in payload.sentences)
    for synthesis in syntheses:
        text_parts.extend([synthesis.title, synthesis.body])
    candidate_ids = sorted(set(UUID_PATTERN.findall("\n".join(text_parts))))
    if not candidate_ids:
        return []
    return list(session.scalars(select(Claim).where(Claim.id.in_(candidate_ids))).all())


def apply_academic_citations(
    payload: ReviewDraftPayload,
    syntheses: Sequence[Synthesis],
    extra_claims: list[Claim] | None = None,
) -> ReviewDraftPayload:
    claim_citations = build_claim_citation_map(syntheses, extra_claims=extra_claims)
    citation_by_paper_id = build_citation_map(syntheses, extra_claims=extra_claims)
    markdown = remove_repeated_title_headings(payload.markdown, payload.title)
    markdown = replace_claim_id_citations(markdown, claim_citations)
    markdown = remove_repeated_title_headings(markdown, payload.title)
    markdown = rebuild_references_section(markdown, citation_by_paper_id, syntheses, extra_claims=extra_claims)
    return payload.model_copy(update={"markdown": markdown})


def build_citation_map(
    syntheses: Sequence[Synthesis],
    extra_claims: list[Claim] | None = None,
) -> dict[str, int]:
    paper_ids: list[str] = []
    for claim in collect_support_claims(syntheses, extra_claims=extra_claims):
        if claim.paper_id not in paper_ids:
            paper_ids.append(claim.paper_id)
    return {paper_id: index for index, paper_id in enumerate(paper_ids, start=1)}


def build_claim_citation_map(
    syntheses: Sequence[Synthesis],
    extra_claims: list[Claim] | None = None,
) -> dict[str, int]:
    citation_by_paper_id = build_citation_map(syntheses, extra_claims=extra_claims)
    claim_citations: dict[str, int] = {}
    for claim in collect_support_claims(syntheses, extra_claims=extra_claims):
        citation_number = citation_by_paper_id.get(claim.paper_id)
        if citation_number is not None:
            claim_citations[claim.id] = citation_number
    return claim_citations


def collect_support_claims(
    syntheses: Sequence[Synthesis],
    extra_claims: list[Claim] | None = None,
) -> list[Claim]:
    claims: list[Claim] = []
    seen: set[str] = set()
    for synthesis in syntheses:
        for claim in synthesis.claims:
            if claim.id not in seen:
                claims.append(claim)
                seen.add(claim.id)
    for claim in extra_claims or []:
        if claim.id not in seen:
            claims.append(claim)
            seen.add(claim.id)
    return claims


def format_reference(
    paper_id: str,
    syntheses: Sequence[Synthesis],
    extra_claims: list[Claim] | None = None,
) -> str:
    for claim in collect_support_claims(syntheses, extra_claims=extra_claims):
        if claim.paper_id != paper_id:
            continue
        paper = getattr(claim, "paper", None)
        if paper is None:
            return f"paper_id={paper_id}"
        file_name = getattr(paper, "file_name", None)
        fallback = infer_paper_metadata_from_name(file_name or getattr(paper, "title", None) or "")
        title = paper.title or fallback.title or paper.file_name or paper.paper_key
        if (
            fallback.title
            and file_name
            and clean_reference_title(str(title)) == clean_reference_title(Path(file_name).stem)
        ):
            title = fallback.title
        authors = getattr(paper, "authors", None) or fallback.authors or []
        author_text = format_author_text(authors)
        year = paper.year if paper.year is not None else fallback.year
        year_text = year if year is not None else "n.d."
        title = clean_reference_title(title)
        return f"{author_text}. ({year_text}). {title}."
    return f"paper_id={paper_id}"


def clean_reference_title(title: str) -> str:
    return re.sub(r"\s+", " ", title.replace("_", " ").strip())


def format_author_text(authors: list[str]) -> str:
    if not authors:
        return "Unknown authors"
    return ", ".join(authors).rstrip(".")


def remove_repeated_title_headings(markdown: str, title: str) -> str:
    lines = markdown.splitlines()
    normalized_title = normalize_heading(title)
    cleaned: list[str] = []
    title_seen = False
    for line in lines:
        if normalize_heading(line) == normalized_title:
            if title_seen:
                continue
            title_seen = True
            cleaned.append(line)
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def normalize_heading(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lstrip("#").strip()).casefold()


def replace_claim_id_citations(markdown: str, claim_citations: dict[str, int]) -> str:
    if not claim_citations:
        return markdown

    def replace_bracket(match: re.Match[str]) -> str:
        body = match.group(1)
        citation_numbers = [
            citation_number for claim_id, citation_number in claim_citations.items() if claim_id in body
        ]
        if not citation_numbers:
            return match.group(0)
        unique_numbers = sorted(set(citation_numbers))
        return "[" + ", ".join(str(number) for number in unique_numbers) + "]"

    return re.sub(r"\[([^\]]+)\]", replace_bracket, markdown)


def rebuild_references_section(
    markdown: str,
    citation_by_paper_id: dict[str, int],
    syntheses: Sequence[Synthesis],
    extra_claims: list[Claim] | None = None,
) -> str:
    if not citation_by_paper_id:
        return markdown

    body = re.split(r"(?im)^\s{0,3}#{0,6}\s*references\s*$", markdown, maxsplit=1)[0].rstrip()
    reference_lines = [
        f"[{citation_number}] {format_reference(paper_id, syntheses, extra_claims=extra_claims)}"
        for paper_id, citation_number in citation_by_paper_id.items()
    ]
    return body + "\n\n## References\n\n" + "\n".join(reference_lines)


def normalize_references_for_markdown(markdown: str) -> str:
    """Tidy up the References section for display as a plain, bullet-free numbered list --
    like a published paper's reference list -- regardless of whether the stored markdown has
    "[N]" entries already on separate lines, crammed onto one line, or "-" bullet-prefixed
    from an older draft format.
    """
    parts = re.split(r"(?im)^\s{0,3}#{0,6}\s*references\s*$", markdown, maxsplit=1)
    if len(parts) != 2:
        return markdown

    body, references = parts
    references = re.sub(r"\s+(?=\[\d+\]\s)", "\n", references.strip())
    reference_lines = []
    for line in references.splitlines():
        line = re.sub(r"^-\s*", "", line.strip()).strip()
        if not line:
            continue
        if re.match(r"^\[\d+\]\s", line):
            reference_lines.append(improve_unknown_reference_line(line))
        else:
            reference_lines.append(line)
    if not reference_lines:
        return body.rstrip()
    return body.rstrip() + "\n\n## References\n\n" + "\n\n".join(reference_lines)


def improve_unknown_reference_line(line: str) -> str:
    match = re.match(r"^(-\s*)?(\[\d+\]\s+)Unknown authors\. \(n\.d\.\)\. (.+?)\.?$", line)
    if match is None:
        return line

    bullet = match.group(1) or ""
    citation = match.group(2)
    raw_title = match.group(3)
    metadata = infer_paper_metadata_from_name(raw_title)
    if not metadata.authors and metadata.year is None:
        return line
    author_text = format_author_text(metadata.authors or [])
    year_text = metadata.year if metadata.year is not None else "n.d."
    title = clean_reference_title(metadata.title or raw_title)
    return f"{bullet}{citation}{author_text}. ({year_text}). {title}."


REVIEW_INSTRUCTIONS = """You write evidence-grounded literature review drafts.

Rules:
- Generate a concise Markdown literature review.
- Every substantive sentence in the Markdown must include bracketed numeric citations such as [1] or [1, 2].
- Use only citation numbers listed in Paper references.
- Never cite claim IDs, UUIDs, synthesis IDs, or paper database IDs in the Markdown.
- Add a Markdown References section at the end using the same bracketed numbers.
- Every substantive sentence payload must include supporting_claim_ids.
- Mark is_supported true only when at least one supporting claim is listed.
- Sentences with no direct support should be marked is_supported false.
- Do not invent claims, papers, methods, datasets, or results beyond the provided syntheses.
- The output must include title, outline, markdown, sentences, and confidence.
"""
