"""PydanticAI agent that wraps the existing retrieval, synthesis, and review-drafting pipeline as
tools, so a single natural-language request can chain them together instead of using each tab by hand.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from pydantic_ai import Agent, AgentRunResult, RunContext
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models import Model
from sqlalchemy.orm import Session

from lit_review_assistant.llm.embeddings import find_similar_claims, find_similar_syntheses
from lit_review_assistant.llm.review import generate_review_draft
from lit_review_assistant.llm.synthesis import SynthesisType, generate_syntheses

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

AGENT_INSTRUCTIONS = """You are a research assistant working on top of a literature review database
that already contains claims extracted from papers, and possibly syntheses and review drafts built
from those claims.

Use the tools to find claims/syntheses relevant to what the user asks about, generate new syntheses
from claims when useful, and generate a review draft when the user wants a written review produced.
Prefer calling find_claims or find_syntheses first to see what already exists before generating
anything new -- don't regenerate syntheses or a draft that already covers the same ground unless the
user asks for something new. In your final answer, report the concrete results you produced or found
(titles, counts, ids), not just a general description of what you did.
"""


@dataclass
class AgentDeps:
    session: Session


@dataclass
class ToolCallSummary:
    tool_name: str
    args: dict[str, object] = field(default_factory=dict)


def build_agent(model: str | Model | None = None) -> Agent[AgentDeps, str]:
    """Construct the assistant agent, with tools bound to a SQLAlchemy session via deps at run time.

    `model` defaults to `openai:$OPENAI_CHAT_MODEL`, but accepts any PydanticAI model string or
    `Model` instance (e.g. `TestModel()`) so tests don't need a real OpenAI API key.
    """
    resolved_model = model if model is not None else f"openai:{os.getenv('OPENAI_CHAT_MODEL', 'gpt-4.1-mini')}"
    agent: Agent[AgentDeps, str] = Agent(
        model=resolved_model,
        deps_type=AgentDeps,
        output_type=str,
        instructions=AGENT_INSTRUCTIONS,
    )

    @agent.tool
    def find_claims(ctx: RunContext[AgentDeps], topic: str, limit: int = 10) -> list[dict[str, str]]:
        """Find existing claims whose embeddings are most similar to `topic`."""
        claims = find_similar_claims(ctx.deps.session, topic, limit=limit)
        return [
            {"claim_id": claim.id, "paper_id": claim.paper_id, "text": claim.normalized_text or claim.claim_text}
            for claim in claims
        ]

    @agent.tool
    def find_syntheses(ctx: RunContext[AgentDeps], topic: str, limit: int = 10) -> list[dict[str, str]]:
        """Find existing syntheses whose embeddings are most similar to `topic`."""
        syntheses = find_similar_syntheses(ctx.deps.session, topic, limit=limit)
        return [
            {"synthesis_id": s.id, "synthesis_type": s.synthesis_type, "title": s.title, "body": s.body}
            for s in syntheses
        ]

    @agent.tool
    def generate_new_syntheses(
        ctx: RunContext[AgentDeps], claim_ids: list[str], synthesis_type: SynthesisType
    ) -> list[dict[str, str]]:
        """Generate and persist new syntheses of one type from the given claim ids."""
        syntheses = generate_syntheses(ctx.deps.session, claim_ids=claim_ids, synthesis_type=synthesis_type)
        return [{"synthesis_id": s.id, "title": s.title, "body": s.body} for s in syntheses]

    @agent.tool
    def generate_draft(ctx: RunContext[AgentDeps], topic: str, synthesis_ids: list[str]) -> dict[str, str]:
        """Generate and persist a review draft from the given synthesis ids."""
        draft = generate_review_draft(ctx.deps.session, topic=topic, synthesis_ids=synthesis_ids)
        if draft is None:
            return {"error": "No review draft was created -- none of the given synthesis ids matched."}
        return {"review_draft_id": draft.id, "title": draft.title, "markdown": draft.markdown}

    return agent


def summarize_tool_calls(result: AgentRunResult[str]) -> list[ToolCallSummary]:
    """Extract (tool_name, args) for every tool call made during an agent run, for display."""
    summaries: list[ToolCallSummary] = []
    for message in result.all_messages():
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                summaries.append(ToolCallSummary(tool_name=part.tool_name, args=part.args_as_dict()))
    return summaries
