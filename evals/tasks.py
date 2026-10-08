"""Run the pipeline stages (sections, claims, retrieval, syntheses, review) on snapshot inputs, without a database.

Each stage calls the same production `request_*` function the app uses (same prompt, instructions,
and default temperature), then reproduces what persistence would do to the output -- id overriding,
location validation, dropping unknown claim ids, citation cleanup -- on transient (never-flushed)
ORM objects, so downstream stages see exactly the inputs production would give them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, cast

import numpy as np
from openai import APIError
from sqlalchemy.orm import Session

from evals.gold import GoldClaim, GoldHeading, RetrievalQuery
from evals.llm_cache import RecordingMissError
from evals.metrics import (
    chunk_claims_metrics,
    gold_claim_metrics,
    ranking_metrics,
    review_metrics,
    section_metrics,
    synthesis_metrics,
)
from evals.snapshot import Snapshot, SnapshotChunk, SnapshotPaper, stable_id
from lit_review_assistant.db.models import Chunk, Claim, Paper, Section, Synthesis
from lit_review_assistant.llm.claims import locate_claims, persist_extracted_claims, request_claims
from lit_review_assistant.llm.client import StructuredLLM
from lit_review_assistant.llm.embeddings import claim_embedding_text, embed_texts
from lit_review_assistant.llm.review import (
    UUID_PATTERN,
    apply_academic_citations,
    build_citation_map,
    request_review_draft,
)
from lit_review_assistant.llm.synthesis import SynthesisType, request_syntheses
from lit_review_assistant.pipeline.chunking import chunk_pages_with_sections
from lit_review_assistant.pipeline.pdf import PageText
from lit_review_assistant.pipeline.sections import detect_sections
from lit_review_assistant.pipeline.support import ClaimSupport, calculate_support_counts


@dataclass(frozen=True)
class Scenario:
    id: str
    topic: str
    synthesis_types: list[SynthesisType]
    max_claims: int = 60


@dataclass
class ClaimsStageResult:
    cases: list[dict[str, Any]]
    # Claims as production would persist them: rejected chunks contribute none.
    accepted_claims: list[Claim] = field(default_factory=list)


class _NullSession:
    """Stands in for a SQLAlchemy session where production code only calls add() and flush()."""

    def add(self, instance: object) -> None:
        pass

    def flush(self) -> None:
        pass


NULL_SESSION = cast(Session, _NullSession())


def describe_error(exc: Exception) -> dict[str, Any]:
    # "infra" errors (missing recording, auth, network, rate limit) mean the harness couldn't evaluate the
    # case at all; anything else is the model producing unusable output, which is itself a quality signal.
    # (A third kind, "upstream", marks a case skipped because an earlier stage produced nothing to use.)
    infra = isinstance(exc, RecordingMissError | APIError)
    return {"error": f"{type(exc).__name__}: {exc}"[:500], "error_kind": "infra" if infra else "model"}


# --- ORM conversion -------------------------------------------------------------------------------


def to_orm_paper(paper: SnapshotPaper) -> Paper:
    return Paper(
        id=paper.id,
        paper_key=paper.key,
        title=paper.title,
        authors=paper.authors,
        year=paper.year,
        file_name=paper.file_name,
        file_sha256=paper.pdf_sha256,
    )


def to_orm_chunk(chunk: SnapshotChunk) -> Chunk:
    section = None
    if chunk.section_type:
        section = Section(
            id=chunk.section_id,
            paper_id=chunk.paper_id,
            title=chunk.section_title or "",
            normalized_type=chunk.section_type,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
        )
    return Chunk(
        id=chunk.id,
        paper_id=chunk.paper_id,
        section_id=chunk.section_id,
        section=section,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        start_char=chunk.start_char,
        end_char=chunk.end_char,
        text=chunk.text,
    )


# --- Stages ---------------------------------------------------------------------------------------


def run_sections_stage(snapshot: Snapshot, gold: Mapping[str, list[GoldHeading]]) -> list[dict[str, Any]]:
    """Re-run the *current* section detection and chunking on the frozen pages and score them against gold.

    No LLM involved. Uses the current code rather than the snapshot's stored chunks, so a change to
    section detection is measured without rebuilding the snapshot.
    """
    cases = []
    for paper in snapshot.papers:
        if paper.key not in gold:
            continue
        pages = [PageText(page_number=number, text=text) for number, text in sorted(paper.pages.items())]
        detected = detect_sections(pages)
        chunks = chunk_pages_with_sections(pages, detected)
        cases.append({"paper_key": paper.key, **section_metrics(detected, gold[paper.key], chunks)})
    return cases


def run_claims_stage(
    snapshot: Snapshot,
    chunks: Sequence[SnapshotChunk],
    llm: StructuredLLM,
    workers: int = 4,
    gold: Mapping[str, list[GoldClaim]] | None = None,
) -> ClaimsStageResult:
    papers = {paper.key: to_orm_paper(paper) for paper in snapshot.papers}

    def run_one(chunk: SnapshotChunk) -> tuple[dict[str, Any], list[Claim]]:
        case: dict[str, Any] = {
            "chunk_key": chunk.key,
            "paper_key": chunk.paper_key,
            "section_type": chunk.section_type,
        }
        orm_chunk = to_orm_chunk(chunk)
        try:
            result = request_claims(orm_chunk, llm=llm)
        except Exception as exc:
            return {**case, **describe_error(exc)}, []

        extracted, unlocated = locate_claims(result.parsed.claims, orm_chunk)
        metrics = chunk_claims_metrics(extracted, orm_chunk, snapshot.paper(chunk.paper_key).pages)
        metrics["n_unlocated"] = len(unlocated)
        metrics["unlocated_quotes"] = [candidate.source_quote for candidate in unlocated]

        if gold is not None and chunk.key in gold:
            gold_pairs = [(item.quote, item.claim_type) for item in gold[chunk.key]]
            # Each predicted claim is matched by the source passage it was located at.
            predicted = [(claim["span_text"], claim["claim_type"]) for claim in metrics["claims"]]
            metrics["gold"] = gold_claim_metrics(predicted, gold_pairs, chunk.text)

        accepted: list[Claim] = []
        if not metrics["rejected"]:
            accepted = persist_extracted_claims(NULL_SESSION, extracted, run_id="eval", chunk=orm_chunk)
            for index, claim in enumerate(accepted):
                claim.id = stable_id("claim", f"{chunk.key}:{index}")
                # Match what a Numeric(4, 3) column reads back as, since the value is echoed into prompts.
                claim.confidence = claim.confidence.quantize(Decimal("0.001"))
                claim.paper = papers[chunk.paper_key]
        return {**case, **metrics}, accepted

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        outcomes = list(pool.map(run_one, chunks))

    return ClaimsStageResult(
        cases=[case for case, _ in outcomes],
        accepted_claims=[claim for _, claims in outcomes for claim in claims],
    )


def run_retrieval_stage(
    snapshot: Snapshot,
    claims: Sequence[Claim],
    queries: Sequence[RetrievalQuery],
    embeddings_client: Any,
) -> list[dict[str, Any]]:
    """Rank the run's claims for each query by embedding similarity, as find_similar_claims does in pgvector.

    Relevance is labelled per chunk, so a claim counts as relevant when its source chunk is relevant to
    the query. Only chunks that were evaluated in this run can contribute claims, so queries whose
    relevant chunks produced no claims are reported as unscored rather than as misses.
    """
    if not claims:
        return [
            {"query_id": query.id, "error": "No accepted claims to retrieve from.", "error_kind": "upstream"}
            for query in queries
        ]
    chunk_key_by_id = {chunk.id: chunk.key for chunk in snapshot.all_chunks()}
    try:
        claim_vectors = np.asarray(
            embed_texts([claim_embedding_text(claim) for claim in claims], client=embeddings_client)
        )
        query_vectors = np.asarray(embed_texts([query.query for query in queries], client=embeddings_client))
    except Exception as exc:
        return [{"query_id": query.id, **describe_error(exc)} for query in queries]

    claim_vectors /= np.linalg.norm(claim_vectors, axis=1, keepdims=True)
    query_vectors /= np.linalg.norm(query_vectors, axis=1, keepdims=True)
    cases = []
    for query, query_vector in zip(queries, query_vectors, strict=True):
        order = np.argsort(-(claim_vectors @ query_vector), kind="stable")
        relevant = [chunk_key_by_id.get(claim.chunk_id) in query.relevant_chunks for claim in claims]
        ranked = [relevant[index] for index in order]
        cases.append(
            {
                "query_id": query.id,
                "query": query.query,
                "pool_size": len(claims),
                "n_relevant": sum(relevant),
                # Expected precision of a random ranking, for context.
                "random_precision": round(sum(relevant) / len(claims), 3),
                **ranking_metrics(ranked, sum(relevant)),
                "top5": [claim_embedding_text(claims[index])[:160] for index in order[:5]],
                "top5_relevant": ranked[:5],
            }
        )
    return cases


def select_claim_pool(claims: Sequence[Claim], limit: int) -> list[Claim]:
    """Round-robin claims across papers up to `limit`, so no single paper dominates the synthesis input.

    Production picks claims by embedding similarity to the topic (or recency); topic-aware retrieval is
    evaluated separately, so here every scenario draws from the same evenly-spread pool.
    """
    by_paper: dict[str, list[Claim]] = {}
    for claim in claims:
        by_paper.setdefault(claim.paper_id, []).append(claim)
    pool: list[Claim] = []
    index = 0
    while len(pool) < limit and any(index < len(bucket) for bucket in by_paper.values()):
        for bucket in by_paper.values():
            if index < len(bucket) and len(pool) < limit:
                pool.append(bucket[index])
        index += 1
    return pool


def run_synthesis_stage(
    scenario: Scenario, claim_pool: Sequence[Claim], llm: StructuredLLM
) -> tuple[list[dict[str, Any]], list[Synthesis]]:
    claims_by_id = {claim.id: claim for claim in claim_pool}
    paper_id_by_claim_id = {claim.id: claim.paper_id for claim in claim_pool}
    cases: list[dict[str, Any]] = []
    syntheses: list[Synthesis] = []

    if not claim_pool:
        # Every claim was lost upstream (errors or rejected chunks); that's already counted there.
        return [
            {"scenario": scenario.id, "error": "No accepted claims to synthesize from.", "error_kind": "upstream"}
        ], []

    for synthesis_type in scenario.synthesis_types:
        case: dict[str, Any] = {
            "scenario": scenario.id,
            "synthesis_type": synthesis_type,
            "n_input_claims": len(claim_pool),
        }
        try:
            result = request_syntheses(claim_pool, synthesis_type, llm=llm)
        except Exception as exc:
            cases.append({**case, **describe_error(exc)})
            continue

        generated = result.parsed.syntheses
        per_synthesis = []
        for index, item in enumerate(generated):
            per_synthesis.append({"title": item.title, **synthesis_metrics(item, synthesis_type, paper_id_by_claim_id)})
            supporting = [
                claims_by_id[claim_id]
                for claim_id in dict.fromkeys(item.supporting_claim_ids)
                if claim_id in claims_by_id
            ]
            counts = calculate_support_counts([ClaimSupport(claim.id, claim.paper_id) for claim in supporting])
            synthesis = Synthesis(
                id=stable_id("synthesis", f"{scenario.id}:{synthesis_type}:{index}"),
                synthesis_type=item.synthesis_type,
                title=item.title,
                body=item.body,
                supporting_papers=counts.supporting_papers,
                supporting_claims=counts.supporting_claims,
                confidence=Decimal(str(item.confidence)).quantize(Decimal("0.001")),
                run_id="eval",
            )
            synthesis.claims = supporting
            syntheses.append(synthesis)
        cases.append({**case, "n_syntheses": len(generated), "syntheses": per_synthesis})
    return cases, syntheses


def run_review_stage(
    scenario: Scenario, syntheses: Sequence[Synthesis], all_claims: Sequence[Claim], llm: StructuredLLM
) -> dict[str, Any]:
    case: dict[str, Any] = {"scenario": scenario.id, "topic": scenario.topic, "n_input_syntheses": len(syntheses)}
    if not syntheses:
        return {**case, "error": "No syntheses to draft from.", "error_kind": "upstream"}
    try:
        result = request_review_draft(scenario.topic, syntheses, llm=llm)
    except Exception as exc:
        return {**case, **describe_error(exc)}

    raw = result.parsed
    # Mirrors find_claims_for_citation_cleanup: any claim UUID the model wrote that exists in the "database".
    claims_by_id = {claim.id: claim for claim in all_claims}
    mentioned = UUID_PATTERN.findall("\n".join([raw.markdown, *(s.sentence_text for s in raw.sentences)]))
    extra_claims = [claims_by_id[claim_id] for claim_id in sorted(set(mentioned)) if claim_id in claims_by_id]
    final = apply_academic_citations(raw, syntheses, extra_claims=extra_claims)
    citation_map = build_citation_map(syntheses, extra_claims=extra_claims)

    return {
        **case,
        **review_metrics(raw, final.markdown, citation_map.values(), known_claim_ids=set(claims_by_id)),
        "title": final.title,
        "markdown": final.markdown,
    }
