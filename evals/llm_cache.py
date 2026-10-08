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

import base64
import hashlib
import json
import os
import re
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import numpy as np
from openai import OpenAI

from evals.pricing import estimate_cost
from lit_review_assistant.llm.client import LLMResult, OpenAIStructuredLLM, ParsedModel, StructuredLLM

Mode = Literal["record", "replay", "refresh"]
DEFAULT_RECORDINGS_DIR = Path("evals/recordings")


def prune_recordings(used_paths: set[Path]) -> int:
    """Delete recordings in the directories this run used that the run itself didn't use; return the count."""
    removed = 0
    for directory in {path.parent for path in used_paths}:
        for path in directory.glob("*.json"):
            if path not in used_paths:
                path.unlink()
                removed += 1
    return removed


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
        self.used_paths: set[Path] = set()

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
        with self._lock:
            self.used_paths.add(path)

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
        # Unique per writer: identical requests (e.g. two chunks with the same text) can be recorded concurrently.
        tmp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
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


# --- Embeddings -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class _EmbeddingItem:
    index: int
    embedding: list[float]


@dataclass(frozen=True)
class _EmbeddingsResponse:
    data: list[_EmbeddingItem]


class RecordingEmbeddings:
    """Duck-types `OpenAI().embeddings` for production's embed_texts, recording one file per input text.

    Vectors are stored as base64 float16 (~4 KB each instead of ~30 KB of JSON floats). That rounding is
    applied to live results too, so a live run and its replay rank identically.
    """

    def __init__(
        self,
        mode: Mode = "record",
        recordings_dir: Path = DEFAULT_RECORDINGS_DIR,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.mode = mode
        self.recordings_dir = recordings_dir
        self._client_factory = client_factory or (lambda: OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
        self._client: Any = None
        self.calls: list[CallRecord] = []
        self.used_paths: set[Path] = set()

    @property
    def embeddings(self) -> RecordingEmbeddings:
        return self

    def create(self, *, model: str, input: list[str]) -> _EmbeddingsResponse:
        directory = self.recordings_dir / re.sub(r"[^A-Za-z0-9._-]", "_", model)
        paths = [directory / f"embedding-{_text_key(model, text)}.json" for text in input]
        self.used_paths.update(paths)
        vectors: dict[int, list[float]] = {}
        if self.mode != "refresh":
            for index, path in enumerate(paths):
                if path.exists():
                    vectors[index] = _decode_vector(json.loads(path.read_text())["embedding_f16_b64"])

        missing = [index for index in range(len(input)) if index not in vectors]
        if missing and self.mode == "replay":
            raise RecordingMissError(f"No recording for {len(missing)} embedding input(s) (model {model}).")
        if missing:
            if self._client is None:
                self._client = self._client_factory()
            started = time.perf_counter()
            response = self._client.embeddings.create(model=model, input=[input[index] for index in missing])
            latency_s = time.perf_counter() - started
            directory.mkdir(parents=True, exist_ok=True)
            for item in response.data:
                index = missing[item.index]
                encoded = _encode_vector(item.embedding)
                paths[index].write_text(json.dumps({"model": model, "embedding_f16_b64": encoded}) + "\n")
                vectors[index] = _decode_vector(encoded)
            tokens = int(getattr(getattr(response, "usage", None), "total_tokens", 0) or 0)
            self.calls.append(
                CallRecord(model, "embeddings", tokens, 0, estimate_cost(model, tokens, 0), False, latency_s)
            )
        if len(missing) < len(input):
            self.calls.append(CallRecord(model, "embeddings", 0, 0, estimate_cost(model, 0, 0), True, None))
        return _EmbeddingsResponse(data=[_EmbeddingItem(index, vectors[index]) for index in range(len(input))])


def _text_key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\n{text}".encode()).hexdigest()[:24]


def _encode_vector(vector: list[float]) -> str:
    return base64.b64encode(np.asarray(vector, dtype="<f2").tobytes()).decode()


def _decode_vector(encoded: str) -> list[float]:
    return np.frombuffer(base64.b64decode(encoded), dtype="<f2").astype(float).tolist()
