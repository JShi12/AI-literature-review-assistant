"""Deterministic (no-LLM) quality metrics for each pipeline stage's output.

Every function here is pure -- it scores an LLM output against its input -- so metrics are cheap,
reproducible, and unit-testable. LLM-as-judge metrics (grounding, faithfulness) are a later phase.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any

from lit_review_assistant.db.models import Chunk
from lit_review_assistant.llm.claims import validate_claim_location_against_chunk
from lit_review_assistant.llm.review import UUID_PATTERN
from lit_review_assistant.pipeline.chunking import TextChunk
from lit_review_assistant.pipeline.review_traceability import normalize_sentence_support
from lit_review_assistant.pipeline.sections import DetectedSection
from lit_review_assistant.schemas import ExtractedClaim, GeneratedSynthesis, ReviewDraftPayload

if TYPE_CHECKING:
    from evals.gold import GoldHeading

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
    }


def chunk_claims_metrics(claims: list[ExtractedClaim], chunk: Chunk, pages: Mapping[int, str]) -> dict[str, Any]:
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


# --- Gold claims ----------------------------------------------------------------------------------

# A predicted claim and a gold claim refer to the same evidence when their chunk spans overlap by at
# least this share of the shorter span.
SPAN_OVERLAP_THRESHOLD = 0.5
# Matching blocks shorter than this are ignored when locating a claim (they're mostly common words).
LOCATE_MIN_BLOCK = 8


def locate_quote(chunk_text: str, quote: str) -> tuple[int, int] | None:
    """Return the quote's [start, end) in *normalized* chunk text, if it appears verbatim."""
    needle = normalize_text(quote)
    start = normalize_text(chunk_text).find(needle) if needle else -1
    return None if start < 0 else (start, start + len(needle))


def locate_claim(claim_text: str, chunk_text: str) -> tuple[int, int] | None:
    """Find where a (possibly lightly paraphrased) claim comes from, in normalized chunk coordinates.

    Uses the claim's text, never its offsets -- the offsets are what's unreliable. Returns None when
    less than half of the claim can be aligned to the chunk (heavy paraphrase or not in the chunk).
    """
    claim = normalize_text(claim_text)
    if not claim:
        return None
    matcher = SequenceMatcher(None, claim, normalize_text(chunk_text), autojunk=False)
    blocks = [block for block in matcher.get_matching_blocks() if block.size >= LOCATE_MIN_BLOCK]
    if sum(block.size for block in blocks) < len(claim) / 2:
        return None
    return blocks[0].b, blocks[-1].b + blocks[-1].size


def span_overlap(a: tuple[int, int], b: tuple[int, int]) -> float:
    shorter = min(a[1] - a[0], b[1] - b[0])
    return max(0, min(a[1], b[1]) - max(a[0], b[0])) / shorter if shorter > 0 else 0.0


def gold_claim_metrics(
    predicted: list[tuple[str, str]], gold: list[tuple[str, str]], chunk_text: str
) -> dict[str, object]:
    """Compare a chunk's extracted claims with its gold claims, both given as (source text, claim_type).

    Matching is by source span, not wording: a gold claim is recalled if some predicted claim comes from
    an overlapping part of the chunk. Not one-to-one, since extractors split claims differently.
    """
    gold_spans = [(locate_quote(chunk_text, quote), claim_type) for quote, claim_type in gold]
    predicted_spans = [(locate_claim(text, chunk_text), claim_type) for text, claim_type in predicted]

    gold_matched = 0
    type_agree = 0
    for gold_span, gold_type in gold_spans:
        if gold_span is None:
            continue
        overlaps = [
            (span_overlap(gold_span, span), claim_type) for span, claim_type in predicted_spans if span is not None
        ]
        best = max(overlaps, default=(0.0, None))
        if best[0] >= SPAN_OVERLAP_THRESHOLD:
            gold_matched += 1
            type_agree += best[1] == gold_type

    predicted_matched = sum(
        1
        for span, _ in predicted_spans
        if span is not None
        and any(gold_span and span_overlap(span, gold_span) >= SPAN_OVERLAP_THRESHOLD for gold_span, _ in gold_spans)
    )
    return {
        "n_gold": len(gold),
        "n_predicted": len(predicted),
        "gold_matched": gold_matched,
        "predicted_matched": predicted_matched,
        "predicted_unlocatable": sum(span is None for span, _ in predicted_spans),
        "type_agree": type_agree,
    }


# --- Sections -------------------------------------------------------------------------------------


def section_metrics(
    detected: Sequence[DetectedSection],
    gold: Sequence[GoldHeading],
    chunks: Sequence[TextChunk],
) -> dict[str, object]:
    """Score detected headings against gold headings, and each chunk's section type against gold.

    A detected heading matches a gold one when it starts on the same line. Chunk-level accuracy is what
    reaches the LLM: the claim-extraction prompt includes the chunk's section type.
    """
    gold_positions = {(heading.page, heading.start_char): heading for heading in gold}
    matched = [
        (section, gold_positions[(section.page_start, section.start_char)])
        for section in detected
        if (section.page_start, section.start_char) in gold_positions
    ]

    ordered_gold = sorted(gold, key=lambda heading: (heading.page, heading.start_char))
    chunk_types: list[tuple[str, str]] = []
    for chunk in chunks:
        current = [h for h in ordered_gold if (h.page, h.start_char) <= (chunk.page_start, chunk.start_char)]
        gold_type = current[-1].normalized_type if current else "other"
        chunk_types.append((chunk.section_type or "other", gold_type))

    return {
        "n_detected": len(detected),
        "n_gold": len(gold),
        "headings_matched": len(matched),
        "matched_type_correct": sum(section.normalized_type == heading.normalized_type for section, heading in matched),
        "n_chunks": len(chunks),
        "chunks_type_correct": sum(detected_type == gold_type for detected_type, gold_type in chunk_types),
        "chunk_gold_types": dict(Counter(gold_type for _, gold_type in chunk_types).most_common()),
        "missed_headings": [
            f"p{heading.page} {heading.heading} ({heading.normalized_type})"
            for heading in gold
            if all(heading is not matched_heading for _, matched_heading in matched)
        ],
    }


# --- Retrieval ------------------------------------------------------------------------------------


def ranking_metrics(ranked_relevance: Sequence[bool], n_relevant: int) -> dict[str, float | None]:
    """Standard IR metrics for one query, given relevance flags of the ranked results (best first)."""
    if n_relevant == 0:
        return {"p@5": None, "p@10": None, "r@10": None, "r@20": None, "mrr": None, "ndcg@10": None}
    first_hit = next((rank for rank, relevant in enumerate(ranked_relevance, start=1) if relevant), None)
    dcg = sum(1 / math.log2(rank + 1) for rank, relevant in enumerate(ranked_relevance[:10], start=1) if relevant)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(n_relevant, 10) + 1))
    return {
        "p@5": round(sum(ranked_relevance[:5]) / 5, 3),
        "p@10": round(sum(ranked_relevance[:10]) / 10, 3),
        "r@10": round(sum(ranked_relevance[:10]) / n_relevant, 3),
        "r@20": round(sum(ranked_relevance[:20]) / n_relevant, 3),
        "mrr": round(1 / first_hit, 3) if first_hit else 0.0,
        "ndcg@10": round(dcg / ideal, 3),
    }
