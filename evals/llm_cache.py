"""Record/replay wrapper around the production StructuredLLM, so evals are reproducible and cheap.

Each response is stored as one JSON file keyed by everything that determines the request: model,
prompt version, temperature, response schema, instructions, and input text. Editing a prompt therefore
misses the recording and forces a fresh call -- stale responses are never replayed against a new prompt.
Only a hash of the input is stored, not the paper text itself.

Modes:
    record   use a recording if one exists, otherwise call the API and save the response (default)
    replay   recordings only; a miss raises RecordingMissError (no API key needed -- for CI)
    refresh  always call the API and overwrite the recording (e.g. to measure run-to-run variance)
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal

from evals.pricing import estimate_cost
from lit_review_assistant.llm.client import LLMResult, OpenAIStructuredLLM, ParsedModel, StructuredLLM

Mode = Literal["record", "replay", "refresh"]
DEFAULT_RECORDINGS_DIR = Path("evals/recordings")


class RecordingMissError(RuntimeError):
    """Raised in replay mode when no recording exists for a request."""


@dataclass(frozen=True)
class CallRecord:
    model: str
    prompt_version: str
    input_tokens: int
    output_tokens: int
    cost: Decimal | None
    cache_hit: bool
    latency_s: float | None


def request_key(
    *,
    model: str,
    text_format: type[object],
    prompt_version: str,
    instructions: str,
    input_text: str,
    temperature: float,
) -> str:
    material = json.dumps(
        {
            "model": model,
            "text_format": text_format.__name__,
            "prompt_version": prompt_version,
            "temperature": temperature,
            "instructions": instructions,
            "input_text": input_text,
        },
        sort_keys=True,
    )
    return hashlib.sha256(material.encode()).hexdigest()


class RecordingLLM:
    """A StructuredLLM that serves responses from disk when it can and records them when it can't."""

    def __init__(
        self,
        model: str,
        mode: Mode = "record",
        recordings_dir: Path = DEFAULT_RECORDINGS_DIR,
        inner_factory: Callable[[str], StructuredLLM] | None = None,
    ) -> None:
        self.model = model
        self.mode = mode
        self.directory = recordings_dir / re.sub(r"[^A-Za-z0-9._-]", "_", model)
        self._inner_factory = inner_factory or (lambda name: OpenAIStructuredLLM(model=name))
        self._inner: StructuredLLM | None = None
        self._lock = threading.Lock()
        self.calls: list[CallRecord] = []

    def parse(
        self,
        *,
        text_format: type[ParsedModel],
        prompt_version: str,
        instructions: str,
        input_text: str,
        temperature: float = 0.1,
    ) -> LLMResult[ParsedModel]:
        key = request_key(
            model=self.model,
            text_format=text_format,
            prompt_version=prompt_version,
            instructions=instructions,
            input_text=input_text,
            temperature=temperature,
        )
        path = self.directory / f"{prompt_version}-{key[:24]}.json"

        if self.mode != "refresh" and path.exists():
            stored = json.loads(path.read_text())
            result = LLMResult(
                parsed=text_format.model_validate(stored["parsed"]),
                model=self.model,
                prompt_version=prompt_version,
                temperature=temperature,
                input_tokens=stored["input_tokens"],
                output_tokens=stored["output_tokens"],
            )
            self._log(result, cache_hit=True, latency_s=None)
            return self._with_cost(result)

        if self.mode == "replay":
            raise RecordingMissError(f"No recording for {prompt_version} request {key[:24]} (model {self.model}).")

        started = time.perf_counter()
        result = self._get_inner().parse(
            text_format=text_format,
            prompt_version=prompt_version,
            instructions=instructions,
            input_text=input_text,
            temperature=temperature,
        )
        latency_s = time.perf_counter() - started
        self._save(path, key, result, latency_s)
        self._log(result, cache_hit=False, latency_s=latency_s)
        return self._with_cost(result)

    def _get_inner(self) -> StructuredLLM:
        # Built lazily: constructing the OpenAI client needs an API key, which replay mode must not require.
        with self._lock:
            if self._inner is None:
                self._inner = self._inner_factory(self.model)
            return self._inner

    def _save(self, path: Path, key: str, result: LLMResult[ParsedModel], latency_s: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "key": key,
            "model": result.model,
            "prompt_version": result.prompt_version,
            "temperature": result.temperature,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "latency_s": round(latency_s, 3),
            "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "parsed": result.parsed.model_dump(mode="json"),
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
        tmp.replace(path)

    def _log(self, result: LLMResult[ParsedModel], *, cache_hit: bool, latency_s: float | None) -> None:
        record = CallRecord(
            model=self.model,
            prompt_version=result.prompt_version,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost=estimate_cost(self.model, result.input_tokens, result.output_tokens),
            cache_hit=cache_hit,
            latency_s=latency_s,
        )
        with self._lock:
            self.calls.append(record)

    def _with_cost(self, result: LLMResult[ParsedModel]) -> LLMResult[ParsedModel]:
        cost = estimate_cost(self.model, result.input_tokens, result.output_tokens)
        return result if cost is None else replace(result, cost=cost)
