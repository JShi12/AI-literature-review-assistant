from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import BaseModel

from evals import run as eval_run
from evals.gold import GoldHeading, RetrievalQuery, locate_line
from evals.llm_cache import RecordingEmbeddings, RecordingLLM, RecordingMissError
from evals.metrics import (
    cited_numbers,
    claim_metrics,
    gold_claim_metrics,
    ranking_metrics,
    review_metrics,
    section_metrics,
    synthesis_metrics,
)
from evals.pricing import estimate_cost
from evals.snapshot import load_snapshot, sample_chunks, write_snapshot
from evals.tasks import run_retrieval_stage, to_orm_chunk
from lit_review_assistant.db.models import Claim
from lit_review_assistant.llm.claims import ExtractedClaimsBatch
from lit_review_assistant.llm.client import LLMResult
from lit_review_assistant.llm.embeddings import embed_texts
from lit_review_assistant.llm.synthesis import GeneratedSynthesesBatch
from lit_review_assistant.pipeline.chunking import chunk_pages_with_sections
from lit_review_assistant.pipeline.pdf import PageText
from lit_review_assistant.pipeline.sections import detect_sections
from lit_review_assistant.schemas import (
    ExtractedClaim,
    GeneratedSynthesis,
    ReviewDraftPayload,
    ReviewSentencePayload,
)

PAGE_ONE = "1 Introduction\nWe train a CNN to steer a car from raw pixels. It drives without lane markings.\n"
PAGE_TWO = "2 Results\nThe model drives autonomously 98% of the time in simulation tests on held-out roads.\n"


def make_snapshot(tmp_path: Path) -> Path:
    papers = []
    for key, pages in {"paper_a": [PAGE_ONE, PAGE_TWO], "paper_b": [PAGE_TWO, PAGE_ONE]}.items():
        papers.append(
            {
                "key": key,
                "title": f"Title of {key}",
                "authors": ["Ada Lovelace"],
                "year": 2020,
                "file_name": f"{key}.pdf",
                "pdf_sha256": "0" * 64,
                "pages": {str(number): text for number, text in enumerate(pages, start=1)},
                "chunks": [
                    {
                        "key": f"{key}:p{number}:0-{len(text)}",
                        "section_title": "Introduction" if text is PAGE_ONE else "Results",
                        "section_type": "introduction" if text is PAGE_ONE else "results",
                        "page_start": number,
                        "page_end": number,
                        "start_char": 0,
                        "end_char": len(text),
                        "text": text,
                    }
                    for number, text in enumerate(pages, start=1)
                ],
            }
        )
    path = tmp_path / "snapshot.json"
    write_snapshot(papers, path)
    return path


class FakePipelineLLM:
    """Returns well-formed outputs for every stage by echoing ids back out of the prompt."""

    def __init__(self) -> None:
        self.calls = 0

    def parse(self, *, text_format, prompt_version, instructions, input_text, temperature=0.1):
        self.calls += 1
        if text_format is ExtractedClaimsBatch:
            fields = dict(re.findall(r"^(\w+): (.*)$", input_text, flags=re.MULTILINE))
            chunk_text = input_text.split("Chunk text:\n", 1)[1]
            sentence = chunk_text.splitlines()[1].split(". ")[0]
            start = chunk_text.index(sentence) + int(fields["chunk_start_char"])
            parsed: BaseModel = ExtractedClaimsBatch(
                claims=[
                    ExtractedClaim(
                        claim_text=sentence,
                        claim_type="finding",
                        paper_id=fields["paper_id"],
                        chunk_id=fields["chunk_id"],
                        section_id=fields["section_id"] or None,
                        page=int(fields["page_start"]),
                        start_char=start,
                        end_char=start + len(sentence),
                        confidence=0.9,
                    )
                ]
            )
        elif text_format is GeneratedSynthesesBatch:
            claim_ids = re.findall(r"claim_id=([0-9a-f-]{36})", input_text)
            synthesis_type = re.search(r"Create (\w+) syntheses", input_text).group(1)  # type: ignore[union-attr]
            parsed = GeneratedSynthesesBatch(
                syntheses=[
                    GeneratedSynthesis(
                        synthesis_type=synthesis_type,  # type: ignore[arg-type]
                        title="Pixels to steering",
                        body="Both papers learn steering directly from camera pixels.",
                        supporting_claim_ids=claim_ids[:2],
                        supporting_papers=2,
                        supporting_claims=2,
                        confidence=0.8,
                    )
                ]
            )
        else:
            claim_id = re.findall(r"([0-9a-f-]{36}) -> \[1\]", input_text)[0]
            sentence = "End-to-end networks learn to steer a car directly from raw camera pixels [1]."
            parsed = ReviewDraftPayload(
                title="Review",
                outline=["Overview"],
                markdown=f"# Review\n\n## Overview\n\n{sentence}\n\n## References\n\n[1] x",
                sentences=[
                    ReviewSentencePayload(
                        section_title="Overview",
                        sentence_index=0,
                        sentence_text=sentence,
                        supporting_claim_ids=[claim_id],
                        is_supported=True,
                    )
                ],
                confidence=0.7,
            )
        return LLMResult(
            parsed=parsed,
            model="fake",
            prompt_version=prompt_version,
            temperature=temperature,
            input_tokens=1_000,
            output_tokens=100,
        )


def test_estimate_cost_matches_longest_model_prefix() -> None:
    assert estimate_cost("gpt-4.1-mini-2025-04-14", 1_000_000, 1_000_000) == Decimal("2.00")
    assert estimate_cost("gpt-4.1", 1_000_000, 0) == Decimal("2.00")
    assert estimate_cost("some-unknown-model", 10, 10) is None


def test_recording_llm_records_then_replays_without_calling_the_api(tmp_path: Path) -> None:
    inner = FakePipelineLLM()
    snapshot = load_snapshot(make_snapshot(tmp_path))
    chunk = to_orm_chunk(snapshot.papers[0].chunks[0])
    from lit_review_assistant.llm.claims import request_claims

    recorder = RecordingLLM("gpt-4.1-mini", mode="record", recordings_dir=tmp_path, inner_factory=lambda _: inner)
    first = request_claims(chunk, llm=recorder)
    second = request_claims(chunk, llm=recorder)

    def explode(_: str) -> FakePipelineLLM:
        raise AssertionError("replay mode must not build an API client")

    replayer = RecordingLLM("gpt-4.1-mini", mode="replay", recordings_dir=tmp_path, inner_factory=explode)
    replayed = request_claims(chunk, llm=replayer)

    assert inner.calls == 1
    assert first.parsed == second.parsed == replayed.parsed
    assert [call.cache_hit for call in recorder.calls] == [False, True]
    assert replayed.cost == Decimal("0.00056")


def test_recording_llm_replay_miss_raises(tmp_path: Path) -> None:
    replayer = RecordingLLM("gpt-4.1-mini", mode="replay", recordings_dir=tmp_path)

    with pytest.raises(RecordingMissError):
        replayer.parse(text_format=ExtractedClaimsBatch, prompt_version="claims.v1", instructions="x", input_text="y")


def test_recording_llm_refresh_always_calls_the_api(tmp_path: Path) -> None:
    inner = FakePipelineLLM()
    snapshot = load_snapshot(make_snapshot(tmp_path))
    chunk = to_orm_chunk(snapshot.papers[0].chunks[0])
    from lit_review_assistant.llm.claims import request_claims

    refresher = RecordingLLM("gpt-4.1-mini", mode="refresh", recordings_dir=tmp_path, inner_factory=lambda _: inner)
    request_claims(chunk, llm=refresher)
    request_claims(chunk, llm=refresher)

    assert inner.calls == 2


def test_sample_chunks_is_reproducible_and_spread_across_papers(tmp_path: Path) -> None:
    snapshot = load_snapshot(make_snapshot(tmp_path))

    first = sample_chunks(snapshot, limit=2, seed=7)
    second = sample_chunks(snapshot, limit=2, seed=7)

    assert [chunk.key for chunk in first] == [chunk.key for chunk in second]
    assert {chunk.paper_key for chunk in first} == {"paper_a", "paper_b"}
    assert len(sample_chunks(snapshot, limit=0)) == 4


def _claim(chunk, start: int, end: int, text: str) -> ExtractedClaim:
    return ExtractedClaim(
        claim_text=text,
        claim_type="finding",
        paper_id=chunk.paper_id,
        chunk_id=chunk.id,
        section_id=chunk.section_id,
        page=chunk.page_start,
        start_char=start,
        end_char=end,
        confidence=0.9,
    )


def test_claim_metrics_detects_offsets_that_point_at_the_wrong_text(tmp_path: Path) -> None:
    snapshot = load_snapshot(make_snapshot(tmp_path))
    paper = snapshot.papers[0]
    chunk = to_orm_chunk(paper.chunks[0])
    text = "We train a CNN to steer a car from raw pixels"
    start = PAGE_ONE.index(text)

    correct = claim_metrics(_claim(chunk, start, start + len(text), text), chunk, paper.pages)
    shifted = claim_metrics(_claim(chunk, start + 20, start + 20 + len(text), text), chunk, paper.pages)
    out_of_chunk = claim_metrics(_claim(chunk, start, len(PAGE_ONE) + 50, text), chunk, paper.pages)

    assert correct["location_valid"] and correct["span_match"] and correct["verbatim"] and correct["ids_copied"]
    assert shifted["location_valid"] and not shifted["span_match"]
    assert not out_of_chunk["location_valid"]


def test_synthesis_metrics_flags_invented_ids_and_wrong_counts() -> None:
    synthesis = GeneratedSynthesis(
        synthesis_type="gap",
        title="t",
        body="b",
        supporting_claim_ids=["C1", "C2", "MADE-UP"],
        supporting_papers=3,
        supporting_claims=3,
        confidence=0.5,
    )

    metrics = synthesis_metrics(synthesis, "theme", {"C1": "P1", "C2": "P2"})

    assert metrics["n_invented"] == 1
    assert metrics["multi_paper"] is True
    assert metrics["type_matches"] is False
    assert metrics["count_mismatch"] is True


def test_review_metrics_flags_phantom_support_and_citation_problems() -> None:
    sentences = [
        ReviewSentencePayload(
            section_title="Intro",
            sentence_index=0,
            sentence_text="Networks steer cars from raw pixels in many settings.",
            supporting_claim_ids=["C1"],
            is_supported=True,
        ),
        ReviewSentencePayload(
            section_title="Intro",
            sentence_index=1,
            sentence_text="Simulation fully replaces real-world testing for every vehicle.",
            supporting_claim_ids=["INVENTED"],
            is_supported=True,
        ),
    ]
    raw = ReviewDraftPayload(title="R", outline=[], markdown="", sentences=sentences, confidence=0.5)
    final_markdown = (
        "# R\n\n## Intro\n\n"
        "Networks steer cars from raw pixels in many settings [1].\n"
        "Simulation fully replaces real-world testing for every vehicle [3].\n"
        "This sentence is long enough to count but carries no citation at all.\n\n"
        "## References\n\n[1] A.\n\n[2] B."
    )

    metrics = review_metrics(raw, final_markdown, citation_numbers=[1, 2], known_claim_ids={"C1"})

    assert metrics["sentence_supported_rate"] == 0.5
    assert metrics["sentence_phantom_support_rate"] == 0.5
    assert metrics["sentence_invented_id_rate"] == 0.5
    assert metrics["sentence_alignment_rate"] == 1.0
    assert metrics["body_citation_coverage"] == pytest.approx(0.667)
    assert metrics["invalid_citation_numbers"] == [3]
    assert metrics["uncited_references"] == [2]
    assert metrics["references_complete"] is True


def test_cited_numbers_expands_lists_and_ranges() -> None:
    assert cited_numbers("a [1] b [2, 4] c [5-7]") == {1, 2, 4, 5, 6, 7}


def test_eval_run_end_to_end_records_then_replays(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot_path = make_snapshot(tmp_path)
    scenarios_path = tmp_path / "scenarios.json"
    scenarios_path.write_text(json.dumps([{"id": "s1", "topic": "Driving", "synthesis_types": ["theme"]}]))
    fake = FakePipelineLLM()
    original_init = RecordingLLM.__init__

    def init_with_fake(self, model, mode="record", recordings_dir=Path(), inner_factory=None):
        original_init(self, model, mode=mode, recordings_dir=recordings_dir, inner_factory=lambda _: fake)

    monkeypatch.setattr(RecordingLLM, "__init__", init_with_fake)
    common = [
        "--model",
        "gpt-4.1-mini",
        "--snapshot",
        str(snapshot_path),
        "--scenarios",
        str(scenarios_path),
        "--recordings",
        str(tmp_path / "recordings"),
        "--out",
        str(tmp_path / "reports"),
        "--limit",
        "0",
    ]

    assert eval_run.main(common) == 0
    calls_after_record = fake.calls
    assert eval_run.main([*common, "--mode", "replay"]) == 0

    assert calls_after_record == 4 + 1 + 1  # one per chunk, one synthesis type, one review
    assert fake.calls == calls_after_record
    reports = sorted((tmp_path / "reports").glob("*/report.json"))
    report = json.loads(reports[-1].read_text())
    summary = report["runs"][0]["summary"]
    assert summary["claims"]["span_match_rate"] == 1.0
    assert summary["claims"]["rejected_chunk_rate"] == 0.0
    assert summary["synthesis"]["invented_id_rate"] == 0.0
    assert summary["review"]["sentence_supported_rate"] == 1.0
    assert summary["review"]["leaked_ids"] == 0
    assert summary["usage"]["cache_hits"] == summary["usage"]["calls"]


def test_gold_claim_metrics_match_by_source_span_not_wording() -> None:
    chunk_text = "We train a CNN to steer a car from raw pixels. It drives without lane markings."
    gold = [("We train a CNN to steer a car from raw pixels", "method"), ("It drives without lane markings", "finding")]
    predicted = [
        ExtractedClaim(
            claim_text="We train a CNN to steer a car from raw camera pixels.",
            claim_type="method",
            paper_id="P",
            chunk_id="C",
            page=1,
            start_char=0,
            end_char=10,
            confidence=0.9,
        ),
        ExtractedClaim(
            claim_text="Completely unrelated statement about weather radar.",
            claim_type="finding",
            paper_id="P",
            chunk_id="C",
            page=1,
            start_char=0,
            end_char=10,
            confidence=0.9,
        ),
    ]

    metrics = gold_claim_metrics(predicted, gold, chunk_text)

    assert metrics == {
        "n_gold": 2,
        "n_predicted": 2,
        "gold_matched": 1,
        "predicted_matched": 1,
        "predicted_unlocatable": 1,
        "type_agree": 1,
    }


def test_section_metrics_scores_headings_and_chunk_types() -> None:
    page = "1 Introduction\nText.\n2 Network Architecture\nMore text.\n"
    pages = [PageText(page_number=1, text=page)]
    detected = detect_sections(pages)
    gold = [
        GoldHeading("p", 1, "1 Introduction", "introduction", locate_line(page, "1 Introduction") or 0),
        GoldHeading("p", 1, "2 Network Architecture", "methods", locate_line(page, "2 Network Architecture") or 0),
    ]
    chunks = chunk_pages_with_sections(pages, detected, max_chars=200, overlap=10)

    metrics = section_metrics(detected, gold, chunks)

    assert metrics["headings_matched"] == 1
    assert metrics["matched_type_correct"] == 1
    assert metrics["missed_headings"] == ["p1 2 Network Architecture (methods)"]


def test_ranking_metrics() -> None:
    metrics = ranking_metrics([False, True, True, False, False, False, False, False, False, False], n_relevant=4)

    assert metrics["p@5"] == 0.4
    assert metrics["r@10"] == 0.5
    assert metrics["mrr"] == 0.5
    assert 0 < metrics["ndcg@10"] < 1
    assert ranking_metrics([True], n_relevant=0)["p@5"] is None


class FakeEmbeddingsClient:
    def __init__(self) -> None:
        self.inputs: list[str] = []
        self.embeddings = self

    def create(self, *, model: str, input: list[str]):
        self.inputs.extend(input)
        vectors = [[1.0, 0.0] if "steer" in text else [0.0, 1.0] for text in input]
        return FakeEmbeddingsResponse([FakeEmbeddingItem(index=i, embedding=v) for i, v in enumerate(vectors)])


@dataclass
class FakeEmbeddingItem:
    index: int
    embedding: list[float]


@dataclass
class FakeEmbeddingsResponse:
    data: list[FakeEmbeddingItem]


def test_recording_embeddings_records_then_replays(tmp_path: Path) -> None:
    fake = FakeEmbeddingsClient()
    recorder = RecordingEmbeddings(mode="record", recordings_dir=tmp_path, client_factory=lambda: fake)
    first = embed_texts(["steer left", "weather"], client=recorder)
    second = embed_texts(["weather", "steer left"], client=recorder)

    replayer = RecordingEmbeddings(mode="replay", recordings_dir=tmp_path, client_factory=lambda: 1 / 0)

    assert fake.inputs == ["steer left", "weather"]
    assert first == [[1.0, 0.0], [0.0, 1.0]]
    assert second == [[0.0, 1.0], [1.0, 0.0]]
    assert embed_texts(["steer left"], client=replayer) == [[1.0, 0.0]]
    with pytest.raises(RecordingMissError):
        embed_texts(["never seen"], client=replayer)


def test_retrieval_stage_ranks_claims_from_relevant_chunks_first(tmp_path: Path) -> None:
    snapshot = load_snapshot(make_snapshot(tmp_path))
    intro, results = snapshot.papers[0].chunks
    claims = [
        Claim(id="c1", claim_text="Rain is common", chunk_id=results.id, paper_id=results.paper_id),
        Claim(id="c2", claim_text="We steer from pixels", chunk_id=intro.id, paper_id=intro.paper_id),
    ]
    query = RetrievalQuery(id="q", query="how to steer", relevant_chunks=frozenset({intro.key}))

    [case] = run_retrieval_stage(snapshot, claims, [query], FakeEmbeddingsClient())

    assert case["n_relevant"] == 1
    assert case["mrr"] == 1.0
    assert case["top5_relevant"] == [True, False]
