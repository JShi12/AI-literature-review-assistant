from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal
from typing import Generic, Protocol, TypeVar

from openai import OpenAI
from pydantic import BaseModel
from sqlalchemy.orm import Session

from lit_review_assistant.db.models import LLMRun


ParsedModel = TypeVar("ParsedModel", bound=BaseModel)


class StructuredLLM(Protocol):
    def parse(
        self,
        *,
        text_format: type[ParsedModel],
        prompt_version: str,
        instructions: str,
        input_text: str,
        temperature: float = 0.1,
    ) -> LLMResult[ParsedModel]:
        pass


@dataclass(frozen=True)
class LLMResult(Generic[ParsedModel]):
    parsed: ParsedModel
    model: str
    prompt_version: str
    temperature: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost: Decimal = Decimal("0")


class OpenAIStructuredLLM:
    def __init__(self, model: str | None = None, client: OpenAI | None = None) -> None:
        self.model = model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
        self.client = client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def parse(
        self,
        *,
        text_format: type[ParsedModel],
        prompt_version: str,
        instructions: str,
        input_text: str,
        temperature: float = 0.1,
    ) -> LLMResult[ParsedModel]:
        response = self.client.responses.parse(
            model=self.model,
            text_format=text_format,
            instructions=instructions,
            input=input_text,
            temperature=temperature,
        )
        parsed = _get_parsed_response(response)
        usage = getattr(response, "usage", None)
        return LLMResult(
            parsed=parsed,
            model=self.model,
            prompt_version=prompt_version,
            temperature=temperature,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        )


def create_llm_run(session: Session, result: LLMResult[BaseModel]) -> LLMRun:
    run = LLMRun(
        model=result.model,
        prompt_version=result.prompt_version,
        temperature=Decimal(str(result.temperature)),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost=result.cost,
    )
    session.add(run)
    session.flush()
    return run


def _get_parsed_response(response: object) -> ParsedModel:
    direct = getattr(response, "output_parsed", None)
    if direct is not None:
        return direct

    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            parsed = getattr(content, "parsed", None)
            if parsed is not None:
                return parsed
    raise ValueError("OpenAI response did not include parsed structured output.")
