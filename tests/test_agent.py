from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.models.test import TestModel

from lit_review_assistant.llm import agent as agent_module
from lit_review_assistant.llm.agent import AgentDeps, build_agent, summarize_tool_calls


@dataclass
class FakeClaim:
    id: str = "CL001"
    paper_id: str = "PAPER001"
    claim_text: str = "Retrieval improves grounding."
    normalized_text: str | None = None


@dataclass
class FakeSynthesis:
    id: str = "SYN001"
    synthesis_type: str = "theme"
    title: str = "Retrieval improves grounding"
    body: str = "Multiple papers use retrieval to reduce unsupported summaries."


@dataclass
class FakeReviewDraft:
    id: str = "DRAFT001"
    title: str = "A Review of Retrieval-Augmented Grounding"
    markdown: str = "# A Review\n\nRetrieval helps [1].\n\n## References\n\n[1] Someone. (2024). A Paper."


def test_agent_tools_call_underlying_pipeline_functions(monkeypatch) -> None:
    session = object()  # tools only pass this through untouched, so an opaque sentinel is enough

    def fake_find_similar_claims(sess: object, topic: str, limit: int) -> list[FakeClaim]:
        assert sess is session
        return [FakeClaim()]

    def fake_find_similar_syntheses(sess: object, topic: str, limit: int) -> list[FakeSynthesis]:
        assert sess is session
        return [FakeSynthesis()]

    def fake_generate_syntheses(sess: object, claim_ids: list[str], synthesis_type: str) -> list[FakeSynthesis]:
        assert sess is session
        return [FakeSynthesis(id="SYN002")]

    def fake_generate_review_draft(sess: object, topic: str, synthesis_ids: list[str]) -> FakeReviewDraft:
        assert sess is session
        return FakeReviewDraft()

    monkeypatch.setattr(agent_module, "find_similar_claims", fake_find_similar_claims)
    monkeypatch.setattr(agent_module, "find_similar_syntheses", fake_find_similar_syntheses)
    monkeypatch.setattr(agent_module, "generate_syntheses", fake_generate_syntheses)
    monkeypatch.setattr(agent_module, "generate_review_draft", fake_generate_review_draft)

    agent = build_agent(model=TestModel())
    result = agent.run_sync("find claims and draft a review", deps=AgentDeps(session=session))  # type: ignore[arg-type]

    tool_names = {call.tool_name for call in summarize_tool_calls(result)}
    assert tool_names == {"find_claims", "find_syntheses", "generate_new_syntheses", "generate_draft"}
    assert isinstance(result.output, str)


def test_generate_draft_tool_reports_when_no_draft_was_created(monkeypatch) -> None:
    monkeypatch.setattr(agent_module, "generate_review_draft", lambda sess, topic, synthesis_ids: None)

    agent = build_agent(model=TestModel(call_tools=["generate_draft"]))
    result = agent.run_sync("draft a review", deps=AgentDeps(session=object()))  # type: ignore[arg-type]

    calls = summarize_tool_calls(result)
    assert len(calls) == 1
    assert calls[0].tool_name == "generate_draft"


def test_summarize_tool_calls_returns_empty_list_when_no_tools_were_called() -> None:
    agent = build_agent(model=TestModel(call_tools=[]))
    result = agent.run_sync("just say hello", deps=AgentDeps(session=object()))  # type: ignore[arg-type]

    assert summarize_tool_calls(result) == []
