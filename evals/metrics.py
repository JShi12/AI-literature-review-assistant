"""Deterministic (no-LLM) quality metrics for each pipeline stage's output.

Every function here is pure -- it scores an LLM output against its input -- so metrics are cheap,
reproducible, and unit-testable. LLM-as-judge metrics (grounding, faithfulness) are a later phase.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from difflib import SequenceMatcher

from lit_review_assistant.db.models import Chunk
from lit_review_assistant.llm.claims import validate_claim_location_against_chunk
from lit_review_assistant.llm.review import UUID_PATTERN
from lit_review_assistant.pipeline.review_traceability import normalize_sentence_support
from lit_review_assistant.schemas import ExtractedClaim, GeneratedSynthesis, ReviewDraftPayload

# A claim's [start_char, end_char) span "matches" its text when they're at least this similar.
SPAN_MATCH_THRESHOLD = 0.8
# A structured review sentence "appears" in the Markdown body when some body sentence is this similar.
SENTENCE_ALIGNMENT_THRESHOLD = 0.85
# Body sentences shorter than this (in words) are treated as connective tissue, not substantive claims.
SUBSTANTIVE_MIN_WORDS = 8

CITATION_RE = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]")
REFERENCES_SPLIT_RE = re.compile(r"(?im)^\s{0,3}#{0,6}\s*references\s*$")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b), autojunk=False).ratio()


def lexical_coverage(claim_text: str, source_text: str) -> float:
    """Share of the claim's characters that appear, in order, in the source text.

    A rough proxy for extractiveness: 1.0 means the claim is quoted verbatim, low values mean it is
    heavily paraphrased (or not in the source at all). Not a substitute for an entailment judge.
    """
    claim = normalize_text(claim_text)
    if not claim:
        return 0.0
    matcher = SequenceMatcher(None, claim, normalize_text(source_text), autojunk=False)
    return sum(block.size for block in matcher.get_matching_blocks()) / len(claim)


# --- Claims ---------------------------------------------------------------------------------------


def claim_metrics(claim: ExtractedClaim, chunk: Chunk, pages: Mapping[int, str]) -> dict[str, object]:
    """Score one extracted claim against the chunk it came from and the page text its offsets point into."""
    try:
        validate_claim_location_against_chunk(claim, chunk)
        location_valid = True
    except ValueError:
        location_valid = False

    page_text = pages.get(claim.page, "")
    span = page_text[claim.start_char : claim.end_char] if 0 <= claim.start_char < claim.end_char else ""
    span_similarity = similarity(claim.claim_text, span) if span else 0.0
    claim_text = normalize_text(claim.claim_text)

    return {
        "claim_text": claim.claim_text,
        "claim_type": claim.claim_type,
        "confidence": claim.confidence,
        "page": claim.page,
        "start_char": claim.start_char,
        "end_char": claim.end_char,
        "span_text": span,
        "location_valid": location_valid,
        "span_similarity": round(span_similarity, 3),
        "span_match": span_similarity >= SPAN_MATCH_THRESHOLD,
        "verbatim": bool(claim_text) and claim_text in normalize_text(chunk.text),
        "lexical_coverage": round(lexical_coverage(claim.claim_text, chunk.text), 3),
        "ids_copied": (
            claim.paper_id == chunk.paper_id
            and claim.chunk_id == chunk.id
            and (claim.section_id or None) == (chunk.section_id or None)
        ),
    }


def chunk_claims_metrics(claims: list[ExtractedClaim], chunk: Chunk, pages: Mapping[int, str]) -> dict[str, object]:
    per_claim = [claim_metrics(claim, chunk, pages) for claim in claims]
    return {
        "n_claims": len(claims),
        "empty": not claims,
        # Production raises on the first claim with an out-of-bounds location, losing every claim in the chunk.
        "rejected": any(not metrics["location_valid"] for metrics in per_claim),
        "claims": per_claim,
    }


# --- Syntheses ------------------------------------------------------------------------------------


def synthesis_metrics(
    synthesis: GeneratedSynthesis,
    requested_type: str,
    paper_id_by_claim_id: Mapping[str, str],
) -> dict[str, object]:
    """Score one synthesis's citations against the claims it was actually given."""
    cited = list(dict.fromkeys(synthesis.supporting_claim_ids))
    valid = [claim_id for claim_id in cited if claim_id in paper_id_by_claim_id]
    papers = {paper_id_by_claim_id[claim_id] for claim_id in valid}
    return {
        "n_cited": len(cited),
        "n_invented": len(cited) - len(valid),
        "unsupported": not valid,
        "type_matches": synthesis.synthesis_type == requested_type,
        "supporting_papers": len(papers),
        "multi_paper": len(papers) >= 2,
        # The model's self-reported counts disagreeing with its own citation list is a hallucination
        # signal (production discards these and recomputes them).
        "count_mismatch": (synthesis.supporting_claims != len(valid) or synthesis.supporting_papers != len(papers)),
    }


# --- Review drafts --------------------------------------------------------------------------------


def split_body_and_references(markdown: str) -> tuple[str, str | None]:
    parts = REFERENCES_SPLIT_RE.split(markdown, maxsplit=1)
    return (parts[0], parts[1]) if len(parts) == 2 else (markdown, None)


def body_sentences(body: str) -> list[str]:
    sentences: list[str] = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^(?:[-*+]|\d+\.)\s+", "", line)
        sentences.extend(part.strip() for part in SENTENCE_SPLIT_RE.split(line) if part.strip())
    return sentences


def cited_numbers(text: str) -> set[int]:
    numbers: set[int] = set()
    for match in CITATION_RE.finditer(text):
        for part in re.split(r"\s*,\s*", match.group(1)):
            bounds = [int(value) for value in re.split(r"\s*[–-]\s*", part)]
            numbers.update(range(bounds[0], bounds[-1] + 1))
    return numbers


def strip_citations(text: str) -> str:
    return UUID_PATTERN.sub("", CITATION_RE.sub("", text))


def review_metrics(
    raw: ReviewDraftPayload,
    final_markdown: str,
    citation_numbers: Iterable[int],
    known_claim_ids: set[str],
) -> dict[str, object]:
    """Score a review draft.

    `raw` is the LLM's structured output (sentence-level traceability); `final_markdown` is the Markdown
    after production's citation cleanup, i.e. what a user actually reads.
    """
    valid_numbers = set(citation_numbers)

    # Sentence-level traceability. Production derives is_supported from *any* cited id, then silently
    # drops ids that don't exist when linking claims -- so a sentence citing only invented ids is stored
    # as supported with nothing behind it ("phantom" support).
    n_sentences = len(raw.sentences)
    cited_ids = [claim_id for sentence in raw.sentences for claim_id in set(sentence.supporting_claim_ids)]
    invented_ids = [claim_id for claim_id in cited_ids if claim_id not in known_claim_ids]
    supported = 0
    phantom = 0
    flag_mismatches = 0
    for sentence in raw.sentences:
        trace = normalize_sentence_support(sentence)
        linked = bool(set(trace.supporting_claim_ids) & known_claim_ids)
        supported += linked
        phantom += trace.is_supported and not linked
        flag_mismatches += sentence.is_supported != linked

    # Markdown body, as rendered.
    body, references = split_body_and_references(final_markdown)
    sentences = body_sentences(body)
    substantive = [
        sentence for sentence in sentences if len(strip_citations(sentence).split()) >= SUBSTANTIVE_MIN_WORDS
    ]
    body_cited = cited_numbers(body)
    reference_lines = [line for line in (references or "").splitlines() if re.match(r"^\s*\[\d+\]\s", line)]

    plain_body_sentences = [strip_citations(sentence) for sentence in sentences]
    aligned = sum(
        1
        for sentence in raw.sentences
        if any(
            similarity(strip_citations(sentence.sentence_text), body_sentence) >= SENTENCE_ALIGNMENT_THRESHOLD
            for body_sentence in plain_body_sentences
        )
    )

    return {
        "n_sentences": n_sentences,
        "sentence_supported_rate": _rate(supported, n_sentences),
        "sentence_phantom_support_rate": _rate(phantom, n_sentences),
        "sentence_invented_id_rate": _rate(len(invented_ids), len(cited_ids)),
        "sentence_flag_mismatch_rate": _rate(flag_mismatches, n_sentences),
        "sentence_alignment_rate": _rate(aligned, n_sentences),
        "n_body_sentences": len(sentences),
        "body_citation_coverage": _rate(
            sum(1 for sentence in substantive if CITATION_RE.search(sentence)), len(substantive)
        ),
        "invalid_citation_numbers": sorted(body_cited - valid_numbers),
        "uncited_references": sorted(valid_numbers - body_cited),
        "leaked_ids": len(UUID_PATTERN.findall(final_markdown)),
        "references_complete": references is not None and len(reference_lines) == len(valid_numbers),
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 3) if denominator else None
