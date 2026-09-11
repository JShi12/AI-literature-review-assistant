"""OpenAI structured-output client wrapper, LLM run usage/cost logging, and API error message formatting."""

from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal
from typing import Generic, Protocol, TypeVar, cast

from openai import APIStatusError, OpenAI
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
    """Calls the OpenAI Responses API and parses the result into a Pydantic model."""

    def __init__(self, model: str | None = None, client: OpenAI | None = None) -> None:
        self.model: str = model if model is not None else os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
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
        """Request a structured completion and return the parsed result with usage metadata."""
        response = self.client.responses.parse(
            model=self.model,
            text_format=text_format,
            instructions=instructions,
            input=input_text,
            temperature=temperature,
        )
        parsed = cast(ParsedModel, _get_parsed_response(response))
        usage = getattr(response, "usage", None)
        return LLMResult(
            parsed=parsed,
            model=self.model,
            prompt_version=prompt_version,
            temperature=temperature,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        )


def create_llm_run(session: Session, result: LLMResult[ParsedModel]) -> LLMRun:
    """Persist an LLM call's model, prompt version, token usage, and cost for observability."""
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


def _get_parsed_response(response: object) -> BaseModel:
    direct = getattr(response, "output_parsed", None)
    if direct is not None:
        return direct

    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            parsed = getattr(content, "parsed", None)
            if parsed is not None:
                return parsed
    raise ValueError("OpenAI response did not include parsed structured output.")


def describe_openai_error(exc: Exception) -> str:
    """Translate an OpenAI API error into an actionable message, special-casing 401/403/429."""
    if not isinstance(exc, APIStatusError):
        return str(exc)

    status_code = getattr(exc, "status_code", None)
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
    message = _extract_openai_error_message(getattr(exc, "body", None)) or str(exc)
    parts = [f"OpenAI API request failed with status {status_code}: {message}"]

    if status_code == 403:
        parts.append(
            "This usually means the API key or project is not allowed to make this request. "
            f"Check that OPENAI_API_KEY belongs to the right project, that the project can use "
            f"OPENAI_CHAT_MODEL='{model}', and that billing, organization policy, and region "
            "restrictions are not blocking the request."
        )
    elif status_code == 401:
        parts.append("Check that OPENAI_API_KEY is set correctly in the terminal running Streamlit.")
    elif status_code == 429:
        parts.append("The project is rate-limited or out of quota. Reduce request size or check billing/usage limits.")

    return " ".join(parts)


def _extract_openai_error_message(body: object) -> str | None:
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str):
                return message
        message = body.get("message")
        if isinstance(message, str):
            return message
    return None
