from __future__ import annotations

from dataclasses import dataclass, field

from lit_review_assistant.llm.claims import build_claim_extraction_input
from lit_review_assistant.llm.client import LLMResult
from lit_review_assistant.llm.review import apply_academic_citations, build_review_input
from lit_review_assistant.pipeline.review_traceability import claim_support_map, normalize_sentence_support, unsupported_sentence_indexes
from lit_review_assistant.schemas import ReviewDraftPayload, ReviewSentencePayload


@dataclass
class FakeSection:
    normalized_type: str = "methods"


@dataclass
class FakeChunk:
    id: str = "CH001"
    paper_id: str = "P001"
    section_id: str = "SEC001"
    section: FakeSection = field(default_factory=FakeSection)
    page_start: int = 3
    page_end: int = 3
    start_char: int = 100
    end_char: int = 180
    text: str = "The proposed method improves accuracy on the benchmark."


@dataclass
class FakeClaim:
    id: str
    paper_id: str = "PAPER001"
    paper: FakePaper | None = None


@dataclass
class FakePaper:
    paper_key: str = "P001"
    title: str = "Retrieval-Augmented Literature Reviews"
    authors: list[str] = field(default_factory=lambda: ["Ada Lovelace", "Grace Hopper"])
    year: int = 2024
    file_name: str = "retrieval_review.pdf"


@dataclass
class FakeSynthesis:
    id: str = "SYN001"
    synthesis_type: str = "theme"
    title: str = "Retrieval improves grounding"
    body: str = "Multiple papers use retrieval to reduce unsupported summaries."
    claims: list[FakeClaim] = field(
        default_factory=lambda: [
            FakeClaim("CL001", paper=FakePaper()),
            FakeClaim("CL002", paper=FakePaper()),
        ]
    )


def test_claim_extraction_input_includes_provenance() -> None:
    prompt = build_claim_extraction_input(FakeChunk())  # type: ignore[arg-type]

    assert "paper_id: P001" in prompt
    assert "chunk_id: CH001" in prompt
    assert "chunk_start_char: 100" in prompt
    assert "The proposed method improves accuracy" in prompt


def test_review_input_includes_syntheses_and_claim_ids() -> None:
    prompt = build_review_input("AI literature review", [FakeSynthesis()])  # type: ignore[list-item]

    assert "Topic: AI literature review" in prompt
    assert "Paper references:" in prompt
    assert "[1] Ada Lovelace, Grace Hopper. (2024). Retrieval-Augmented Literature Reviews." in prompt
    assert "synthesis_id=SYN001" in prompt
    assert "CL001" in prompt
    assert "CL002" in prompt
    assert "CL001 -> [1]" in prompt


def test_review_markdown_claim_ids_are_replaced_with_numbered_references() -> None:
    payload = ReviewDraftPayload(
        title="Draft",
        outline=["Intro"],
        markdown='The approach improves grounding ["CL001","CL002"].',
        sentences=[
            ReviewSentencePayload(
                section_title="Intro",
                sentence_index=0,
                sentence_text="The approach improves grounding.",
                supporting_claim_ids=["CL001", "CL002"],
                is_supported=True,
            )
        ],
        confidence=0.8,
    )

    updated = apply_academic_citations(payload, [FakeSynthesis()])  # type: ignore[list-item]

    assert '["CL001","CL002"]' not in updated.markdown
    assert "The approach improves grounding [1]." in updated.markdown
    assert "## References" in updated.markdown
    assert "[1] Ada Lovelace, Grace Hopper. (2024). Retrieval-Augmented Literature Reviews." in updated.markdown


def test_review_markdown_unquoted_uuid_citations_are_replaced() -> None:
    claim_id = "fbacfa8f-d744-4a4f-8bcc-ea3417436d45"
    payload = ReviewDraftPayload(
        title="Coffee Ring Review",
        outline=["Intro"],
        markdown=(
            "Coffee Ring Review\n"
            "Coffee Ring Review\n"
            "Polymer additives suppress ring formation "
            f"[{claim_id}]."
        ),
        sentences=[
            ReviewSentencePayload(
                section_title="Intro",
                sentence_index=0,
                sentence_text="Polymer additives suppress ring formation.",
                supporting_claim_ids=[claim_id],
                is_supported=True,
            )
        ],
        confidence=0.8,
    )
    synthesis = FakeSynthesis(claims=[])
    claim = FakeClaim(claim_id, paper=FakePaper(title="Coffee Ring Suppression"))

    updated = apply_academic_citations(payload, [synthesis], extra_claims=[claim])  # type: ignore[list-item]

    assert claim_id not in updated.markdown
    assert "Polymer additives suppress ring formation [1]." in updated.markdown
    assert updated.markdown.count("Coffee Ring Review") == 1
    assert "## References" in updated.markdown
    assert "[1] Ada Lovelace, Grace Hopper. (2024). Coffee Ring Suppression." in updated.markdown


def test_review_traceability_normalizes_many_claim_support() -> None:
    sentence = ReviewSentencePayload(
        section_title="Methods",
        sentence_index=2,
        sentence_text="Retrieval improves grounding.",
        supporting_claim_ids=["CL002", "CL001", "CL001"],
        is_supported=False,
    )

    normalized = normalize_sentence_support(sentence)

    assert normalized.is_supported
    assert normalized.supporting_claim_ids == ["CL001", "CL002"]


def test_unsupported_sentence_indexes_and_support_map() -> None:
    sentences = [
        ReviewSentencePayload(
            section_title="Intro",
            sentence_index=0,
            sentence_text="Supported.",
            supporting_claim_ids=["CL001"],
            is_supported=True,
        ),
        ReviewSentencePayload(
            section_title="Intro",
            sentence_index=1,
            sentence_text="Unsupported.",
            supporting_claim_ids=[],
            is_supported=True,
        ),
    ]

    assert unsupported_sentence_indexes(sentences) == [1]
    assert claim_support_map(sentences) == {0: ["CL001"], 1: []}


def test_llm_result_carries_run_metadata() -> None:
    result = LLMResult(
        parsed=ReviewSentencePayload(
            section_title="Intro",
            sentence_index=0,
            sentence_text="Supported.",
            supporting_claim_ids=["CL001"],
            is_supported=True,
        ),
        model="test-model",
        prompt_version="test.v1",
        temperature=0.1,
        input_tokens=10,
        output_tokens=20,
    )

    assert result.model == "test-model"
    assert result.prompt_version == "test.v1"
    assert result.input_tokens == 10
