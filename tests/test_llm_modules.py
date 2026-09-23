from __future__ import annotations

from dataclasses import dataclass, field

import httpx2
from openai import APIStatusError

from lit_review_assistant.llm.claims import build_claim_extraction_input, persist_extracted_claims
from lit_review_assistant.llm.client import LLMResult, describe_openai_error
from lit_review_assistant.llm.embeddings import (
    embed_and_persist_claims,
    embed_texts,
    find_similar_claims,
)
from lit_review_assistant.llm.review import (
    apply_academic_citations,
    build_review_input,
    normalize_references_for_markdown,
)
from lit_review_assistant.pipeline.review_traceability import (
    claim_support_map,
    normalize_sentence_support,
    unsupported_sentence_indexes,
)
from lit_review_assistant.schemas import ExtractedClaim, ReviewDraftPayload, ReviewSentencePayload


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


def test_persist_extracted_claims_uses_authoritative_chunk_ids() -> None:
    class FakeSession:
        def __init__(self) -> None:
            self.added = []

        def add(self, model) -> None:
            self.added.append(model)

        def flush(self) -> None:
            return None

    extracted = ExtractedClaim(
        claim_text="The proposed method improves accuracy.",
        claim_type="finding",
        normalized_text="Method improves accuracy.",
        paper_id="WRONG_PAPER",
        chunk_id="WRONG_CHUNK",
        section_id="WRONG_SECTION",
        page=3,
        start_char=100,
        end_char=155,
        confidence=0.8,
    )
    session = FakeSession()

    claims = persist_extracted_claims(session, [extracted], "RUN001", chunk=FakeChunk())  # type: ignore[arg-type]

    assert len(claims) == 1
    assert claims[0].paper_id == "P001"
    assert claims[0].chunk_id == "CH001"
    assert claims[0].section_id == "SEC001"


@dataclass
class FakeEmbeddingItem:
    index: int
    embedding: list[float]


@dataclass
class FakeEmbeddingsResponse:
    data: list[FakeEmbeddingItem]


class FakeEmbeddingsAPI:
    def create(self, *, model: str, input: list[str]):
        # Return out of input order to prove callers re-sort by index rather than trust response order.
        return FakeEmbeddingsResponse(
            [FakeEmbeddingItem(index=i, embedding=[float(i)]) for i in reversed(range(len(input)))]
        )


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.embeddings = FakeEmbeddingsAPI()


class FailingEmbeddingsAPI:
    def create(self, *, model: str, input: list[str]):
        raise RuntimeError("embedding API unavailable")


class FailingOpenAIClient:
    def __init__(self) -> None:
        self.embeddings = FailingEmbeddingsAPI()


class ExplodingEmbeddingsAPI:
    def create(self, *, model: str, input: list[str]):
        raise AssertionError("should not call the embeddings API for a blank topic")


class ExplodingOpenAIClient:
    def __init__(self) -> None:
        self.embeddings = ExplodingEmbeddingsAPI()


class ExplodingSession:
    def scalars(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("should not query the database for a blank topic")


def test_embed_texts_reorders_response_to_match_input() -> None:
    vectors = embed_texts(["first", "second", "third"], client=FakeOpenAIClient())  # type: ignore[arg-type]

    assert vectors == [[0.0], [1.0], [2.0]]


def test_embed_and_persist_claims_stores_one_embedding_per_claim() -> None:
    class FakeSession:
        def __init__(self) -> None:
            self.added: list = []

        def add(self, model) -> None:
            self.added.append(model)

        def flush(self) -> None:
            return None

    @dataclass
    class FakeClaimForEmbedding:
        id: str
        claim_text: str
        normalized_text: str | None = None

    claims = [
        FakeClaimForEmbedding(id="CL001", claim_text="Text one", normalized_text="Normalized one"),
        FakeClaimForEmbedding(id="CL002", claim_text="Text two"),
    ]
    session = FakeSession()

    embed_and_persist_claims(session, claims, client=FakeOpenAIClient())  # type: ignore[arg-type]

    assert len(session.added) == 2
    assert {row.entity_id for row in session.added} == {"CL001", "CL002"}
    assert all(row.entity_type == "claim" for row in session.added)


def test_embed_and_persist_claims_swallows_client_errors() -> None:
    class ExplodingSessionForAdd:
        def add(self, model) -> None:
            raise AssertionError("should not persist an embedding when the API call fails")

        def flush(self) -> None:
            return None

    @dataclass
    class FakeClaimForEmbedding:
        id: str
        claim_text: str

    # Should not raise, even though the embeddings API call fails.
    embed_and_persist_claims(
        ExplodingSessionForAdd(),  # type: ignore[arg-type]
        [FakeClaimForEmbedding(id="CL001", claim_text="x")],
        client=FailingOpenAIClient(),  # type: ignore[arg-type]
    )


def test_find_similar_claims_short_circuits_on_blank_topic() -> None:
    result = find_similar_claims(
        ExplodingSession(),  # type: ignore[arg-type]
        "   ",
        limit=5,
        client=ExplodingOpenAIClient(),  # type: ignore[arg-type]
    )

    assert result == []


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
        markdown=(f"Coffee Ring Review\nCoffee Ring Review\nPolymer additives suppress ring formation [{claim_id}]."),
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


def test_review_references_are_separated_by_blank_lines() -> None:
    payload = ReviewDraftPayload(
        title="Draft",
        outline=["Intro"],
        markdown="Two papers support retrieval [CL001, CL003].",
        sentences=[
            ReviewSentencePayload(
                section_title="Intro",
                sentence_index=0,
                sentence_text="Two papers support retrieval.",
                supporting_claim_ids=["CL001", "CL003"],
                is_supported=True,
            )
        ],
        confidence=0.8,
    )
    synthesis = FakeSynthesis(
        claims=[
            FakeClaim("CL001", paper_id="PAPER001", paper=FakePaper(title="First Paper")),
            FakeClaim("CL003", paper_id="PAPER002", paper=FakePaper(title="Second Paper")),
        ]
    )

    updated = apply_academic_citations(payload, [synthesis])  # type: ignore[list-item]

    assert (
        "[1] Ada Lovelace, Grace Hopper. (2024). First Paper.\n[2] Ada Lovelace, Grace Hopper. (2024). Second Paper."
    ) in updated.markdown
    assert "- [" not in updated.markdown


def test_reference_fallback_infers_authors_from_filename() -> None:
    payload = ReviewDraftPayload(
        title="Draft",
        outline=["Intro"],
        markdown="Coffee ring studies compare surfactant mixtures [CL001].",
        sentences=[
            ReviewSentencePayload(
                section_title="Intro",
                sentence_index=0,
                sentence_text="Coffee ring studies compare surfactant mixtures.",
                supporting_claim_ids=["CL001"],
                is_supported=True,
            )
        ],
        confidence=0.8,
    )
    paper = FakePaper(
        title="Anyfantakis1 Baigl5-2015-Modulation of the Coffee-Ring Effect in Particle-Surfactant Mixtures",
        authors=[],
        year=None,  # type: ignore[arg-type]
        file_name="Anyfantakis1 Baigl5-2015-Modulation of the Coffee-Ring Effect in Particle-Surfactant Mixtures.pdf",
    )
    synthesis = FakeSynthesis(claims=[FakeClaim("CL001", paper_id="PAPER001", paper=paper)])

    updated = apply_academic_citations(payload, [synthesis])  # type: ignore[list-item]

    assert (
        "## References\n\n"
        "[1] Anyfantakis, Baigl. (2015). Modulation of the Coffee-Ring Effect in Particle-Surfactant Mixtures."
    ) in updated.markdown
    assert "- [" not in updated.markdown


def test_inline_references_are_normalized_for_markdown_display() -> None:
    markdown = (
        "Draft body.\n\n"
        "References\n"
        "[1] Unknown authors. (n.d.). First paper. [2] Unknown authors. (n.d.). Second paper."
    )

    normalized = normalize_references_for_markdown(markdown)

    assert "## References" in normalized
    assert "[1] Unknown authors. (n.d.). First paper." in normalized
    assert "[2] Unknown authors. (n.d.). Second paper." in normalized
    assert "First paper. [2]" not in normalized
    assert "- [" not in normalized


def test_inline_unknown_references_are_improved_from_titles() -> None:
    markdown = (
        "References\n"
        "[1] Unknown authors. (n.d.). Anyfantakis1 Baigl5-2015-Modulation of the Coffee-Ring Effect. "
        "[2] Unknown authors. (n.d.). Cui, B.Yang-2014-Suppression of the Coffee Ring Effect."
    )

    normalized = normalize_references_for_markdown(markdown)

    assert "[1] Anyfantakis, Baigl. (2015). Modulation of the Coffee-Ring Effect." in normalized
    assert "[2] Cui, B. Yang. (2014). Suppression of the Coffee Ring Effect." in normalized
    assert "Unknown authors" not in normalized
    assert "- [" not in normalized


def test_dash_prefixed_references_are_normalized_without_a_stray_bullet() -> None:
    # Regression: references already stored in the older "- [N] ..." bullet format (one per
    # line) previously ended up with an extra bare "-" line inserted before each real entry,
    # which Streamlit renders as a visible empty bullet point. The dash prefix itself should
    # also be dropped entirely -- references should read like a plain published-paper list,
    # not a bulleted list.
    markdown = (
        "Draft body.\n\n"
        "## References\n\n"
        "- [1] Ada Lovelace. (2020). First paper.\n"
        "- [2] Grace Hopper. (2021). Second paper."
    )

    normalized = normalize_references_for_markdown(markdown)

    assert "[1] Ada Lovelace. (2020). First paper." in normalized
    assert "[2] Grace Hopper. (2021). Second paper." in normalized
    assert "- [" not in normalized
    assert "\n-\n" not in normalized
    assert "Effect. [2]" not in normalized


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


def test_describe_openai_error_expands_permission_denied() -> None:
    response = httpx2.Response(403, request=httpx2.Request("POST", "https://api.openai.com/v1/responses"))
    exc = APIStatusError(
        "Error code: 403",
        response=response,
        body={"error": {"message": "Project does not have access to this model."}},
    )

    message = describe_openai_error(exc)

    assert "status 403" in message
    assert "Project does not have access to this model." in message
    assert "OPENAI_API_KEY" in message
    assert "OPENAI_CHAT_MODEL" in message
