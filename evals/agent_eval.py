"""Trajectory evals for the "Ask the Assistant" agent: does it call the right tools, in a sensible order,
with ids it actually got back from earlier tools, and does it report concrete results?

The agent under test is production's `build_agent` -- same instructions, same tools. Only what sits
*under* the tools is swapped: instead of Postgres/pgvector, an in-memory store built from the eval's own
claims and syntheses answers the four tool functions, using recorded embeddings and the recorded
synthesis/review LLM. The tools' own quality is measured by the other stages; this stage measures the
agent's decisions. The agent's model calls are recorded and replayed too (RecordingModel), so the stage
replays offline like everything else.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    ToolReturnPart,
)
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from evals.llm_cache import DEFAULT_RECORDINGS_DIR, Mode, RecordingMissError
from evals.snapshot import stable_id
from evals.tasks import build_syntheses, describe_error
from lit_review_assistant.db.models import Claim, ReviewDraft, Synthesis
from lit_review_assistant.llm import agent as agent_module
from lit_review_assistant.llm.agent import AgentDeps, build_agent
from lit_review_assistant.llm.client import StructuredLLM
from lit_review_assistant.llm.embeddings import claim_embedding_text, embed_texts
from lit_review_assistant.llm.review import apply_academic_citations, request_review_draft
from lit_review_assistant.llm.synthesis import SynthesisType, request_syntheses

DEFAULT_AGENT_CASES_PATH = Path("evals/datasets/agent_cases.json")
UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
# Message fields that change between otherwise identical runs; left out of the recording key.
VOLATILE_KEYS = {
    "timestamp",
    "run_id",
    "conversation_id",
    "usage",
    "provider_response_id",
    "provider_details",
    "provider_url",
    "metadata",
    "id",
}


# --- Recording the agent's own model calls --------------------------------------------------------


def _strip_volatile(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_volatile(item) for key, item in value.items() if key not in VOLATILE_KEYS}
    if isinstance(value, list):
        return [_strip_volatile(item) for item in value]
    return value


class RecordingModel(WrapperModel):
    """Wraps the production OpenAI model, recording each response keyed by the conversation so far."""

    def __init__(
        self,
        model_name: str,
        mode: Mode = "record",
        recordings_dir: Path = DEFAULT_RECORDINGS_DIR,
        wrapped: Model | None = None,
    ) -> None:
        # By default the same model class production's "openai:<name>" resolves to, so tool schemas -- and
        # therefore the recording keys -- match. Replay needs no key: it never reaches the API.
        if wrapped is None:
            api_key = os.getenv("OPENAI_API_KEY") or "unused-in-replay"
            wrapped = OpenAIResponsesModel(model_name, provider=OpenAIProvider(api_key=api_key))
        super().__init__(wrapped)
        self.mode = mode
        self.directory = recordings_dir / re.sub(r"[^A-Za-z0-9._-]", "_", model_name)
        self._recorded_name = model_name
        self.used_paths: set[Path] = set()
        self.hits = 0
        self.misses = 0

    def _key(self, messages: list[ModelMessage], parameters: ModelRequestParameters) -> str:
        material = {
            "model": self._recorded_name,
            "messages": _strip_volatile(ModelMessagesTypeAdapter.dump_python(messages, mode="json")),
            "tools": [[tool.name, tool.description, tool.parameters_json_schema] for tool in parameters.function_tools],
        }
        return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        path = self.directory / f"agent-{self._key(messages, model_request_parameters)[:24]}.json"
        self.used_paths.add(path)
        if self.mode != "refresh" and path.exists():
            self.hits += 1
            stored = json.loads(path.read_text())["response"]
            response = ModelMessagesTypeAdapter.validate_python([stored])[0]
            assert isinstance(response, ModelResponse)
            return response
        if self.mode == "replay":
            raise RecordingMissError(f"No recording for agent request {path.stem} (model {self.model_name}).")
        self.misses += 1
        response = await super().request(messages, model_settings, model_request_parameters)
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = ModelMessagesTypeAdapter.dump_python([response], mode="json")[0]
        path.write_text(json.dumps({"model": self._recorded_name, "response": serialized}, indent=1) + "\n")
        return response


# --- In-memory stand-ins for the agent's tool functions -------------------------------------------


@dataclass
class AgentStore:
    """Answers find_similar_*/generate_* from in-memory claims and syntheses instead of Postgres."""

    case_id: str
    claims: list[Claim]
    syntheses: list[Synthesis]
    llm: StructuredLLM
    embeddings_client: Any
    drafts: list[ReviewDraft] = field(default_factory=list)
    _generated: int = 0

    def _rank(self, topic: str, texts: Sequence[str], limit: int) -> list[int]:
        if not texts or not topic.strip():
            return []
        vectors = np.asarray(embed_texts(list(texts), client=self.embeddings_client))
        query = np.asarray(embed_texts([topic], client=self.embeddings_client)[0])
        scores = (vectors @ query) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query))
        return [int(index) for index in np.argsort(-scores, kind="stable")[:limit]]

    def find_similar_claims(self, session: object, topic: str, limit: int, client: object = None) -> list[Claim]:
        order = self._rank(topic, [claim_embedding_text(claim) for claim in self.claims], limit)
        return [self.claims[index] for index in order]

    def find_similar_syntheses(self, session: object, topic: str, limit: int, client: object = None) -> list[Synthesis]:
        texts = [f"{synthesis.title}\n{synthesis.body}" for synthesis in self.syntheses]
        return [self.syntheses[index] for index in self._rank(topic, texts, limit)]

    def generate_syntheses(
        self, session: object, claim_ids: list[str], synthesis_type: SynthesisType, **_: object
    ) -> list[Synthesis]:
        by_id = {claim.id: claim for claim in self.claims}
        claims = [by_id[claim_id] for claim_id in claim_ids if claim_id in by_id]
        if not claims:
            return []
        self._generated += 1
        result = request_syntheses(claims, synthesis_type, llm=self.llm)
        created = build_syntheses(
            result.parsed.syntheses, by_id, id_prefix=f"agent:{self.case_id}:{self._generated}:{synthesis_type}"
        )
        self.syntheses.extend(created)
        return created

    def generate_review_draft(
        self, session: object, topic: str, synthesis_ids: list[str], **_: object
    ) -> ReviewDraft | None:
        by_id = {synthesis.id: synthesis for synthesis in self.syntheses}
        syntheses = [by_id[synthesis_id] for synthesis_id in synthesis_ids if synthesis_id in by_id]
        if not syntheses:
            return None
        parsed = apply_academic_citations(request_review_draft(topic, syntheses, llm=self.llm).parsed, syntheses)
        draft = ReviewDraft(
            id=stable_id("review", f"agent:{self.case_id}:{len(self.drafts)}"),
            title=parsed.title,
            outline=parsed.outline,
            markdown=parsed.markdown,
        )
        self.drafts.append(draft)
        return draft


# --- Running and scoring cases --------------------------------------------------------------------


@dataclass(frozen=True)
class AgentCase:
    id: str
    prompt: str
    must_call: list[str] = field(default_factory=list)
    must_not_call: list[str] = field(default_factory=list)
    # Pairs [a, b]: if b is called, a must have been called before b's first call.
    order: list[list[str]] = field(default_factory=list)
    synthesis_type: str | None = None
    max_tool_calls: int = 8


def load_agent_cases(path: Path = DEFAULT_AGENT_CASES_PATH) -> list[AgentCase]:
    return [AgentCase(**case) for case in json.loads(path.read_text())]


@dataclass
class ToolResults:
    """What one tool call returned, as the agent saw it."""

    position: int
    ids: set[str]
    titles: set[str]
    texts: list[str]


def tool_results(messages: Sequence[ModelMessage]) -> list[ToolResults]:
    results = []
    for position, message in enumerate(messages):
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if isinstance(part, ToolReturnPart):
                content = part.content if isinstance(part.content, list) else [part.content]
                items = [item for item in content if isinstance(item, dict)]
                results.append(
                    ToolResults(
                        position=position,
                        ids=set(UUID_RE.findall(json.dumps(part.content, default=str))),
                        titles={str(item["title"]) for item in items if item.get("title")},
                        texts=[
                            str(item.get("text") or item.get("body"))
                            for item in items
                            if item.get("text") or item.get("body")
                        ],
                    )
                )
    return results


def _content_words(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", text.casefold()) if len(word) > 3}


def reports_results(output: str, results: Sequence[ToolResults]) -> bool:
    """Whether the final answer conveys concrete results: a returned id or title, or the substance of a
    returned claim/synthesis (most of its content words). Vacuously true if no tool returned anything."""
    if not any(result.ids for result in results):
        return True
    answer = output.casefold()
    answer_words = _content_words(output)
    for result in results:
        if result.ids & set(UUID_RE.findall(output)) or any(title.casefold() in answer for title in result.titles):
            return True
        for text in result.texts:
            words = _content_words(text)
            if words and len(words & answer_words) >= 0.6 * len(words):
                return True
    return False


def check_trajectory(case: AgentCase, messages: Sequence[ModelMessage], output: str) -> dict[str, Any]:
    """Score one agent run against its case's expectations plus checks that apply to every case."""
    calls: list[tuple[int, str, dict[str, Any]]] = []
    for position, message in enumerate(messages):
        if isinstance(message, ModelResponse):
            for call in message.tool_calls:
                calls.append((position, call.tool_name, call.args_as_dict()))
    names = [name for _, name, _ in calls]
    results = tool_results(messages)

    # Ids the agent passed to a tool that no *earlier* tool returned -- i.e. made up.
    invented: list[str] = []
    for position, _, args in calls:
        seen = set().union(*(result.ids for result in results if result.position < position))
        invented.extend(i for i in args.get("claim_ids", []) + args.get("synthesis_ids", []) if i not in seen)

    def first(name: str) -> int | None:
        return names.index(name) if name in names else None

    checks: dict[str, bool] = {}
    for name in case.must_call:
        checks[f"calls {name}"] = name in names
    for name in case.must_not_call:
        checks[f"does not call {name}"] = name not in names
    for before, after in case.order:
        b_index, a_index = first(after), first(before)
        checks[f"{before} before {after}"] = b_index is None or (a_index is not None and a_index < b_index)
    if case.synthesis_type is not None:
        requested = [args.get("synthesis_type") for _, name, args in calls if name == "generate_new_syntheses"]
        # Presence is checked by must_call; this checks the type of any synthesis the agent did request.
        checks[f"synthesis_type {case.synthesis_type}"] = all(value == case.synthesis_type for value in requested)
    checks[f"at most {case.max_tool_calls} tool calls"] = len(calls) <= case.max_tool_calls
    checks["no invented ids"] = not invented

    return {
        "tool_calls": [{"tool": name, "args": args} for _, name, args in calls],
        "checks": checks,
        "passed": all(checks.values()),
        "n_tool_calls": len(calls),
        "invented_ids": invented,
        # The instructions ask the agent to report concrete results (ids, titles), not just a description.
        "answer_reports_results": reports_results(output, results),
        "output": output,
    }


def run_agent_stage(
    cases: Sequence[AgentCase],
    claims: Sequence[Claim],
    syntheses: Sequence[Synthesis],
    model: RecordingModel,
    llm: StructuredLLM,
    embeddings_client: Any,
) -> list[dict[str, Any]]:
    agent = build_agent(model=model)
    results = []
    for case in cases:
        # Every case starts from the same state, so cases can't leak into each other.
        store = AgentStore(case.id, list(claims), list(syntheses), llm, embeddings_client)
        record: dict[str, Any] = {"case_id": case.id, "prompt": case.prompt}
        with mock.patch.multiple(
            agent_module,
            find_similar_claims=store.find_similar_claims,
            find_similar_syntheses=store.find_similar_syntheses,
            generate_syntheses=store.generate_syntheses,
            generate_review_draft=store.generate_review_draft,
        ):
            try:
                result = agent.run_sync(case.prompt, deps=AgentDeps(session=None))  # type: ignore[arg-type]
            except Exception as exc:
                results.append({**record, **describe_error(exc)})
                continue
        results.append({**record, **check_trajectory(case, result.all_messages(), result.output)})
    return results
