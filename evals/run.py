"""Run the eval harness over a frozen snapshot, for one or more models.

Stages: sections (no LLM) -> claims (+ gold claims) -> retrieval -> syntheses -> review drafts.

    python -m evals.run                               # default model, record mode, 30 chunks
    python -m evals.run --mode replay                 # recordings only, no API key needed
    python -m evals.run --model gpt-4.1-mini --model gpt-4.1 --limit 0   # compare models on every chunk
    python -m evals.run --stop-after claims           # just the claims stage

Writes evals/reports/<run id>/report.json (every case, every metric) and summary.md (headline table).
Exits 1 if any case couldn't be evaluated at all (missing recording, API/auth/network error).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from dotenv import load_dotenv

from evals.gold import (
    GOLD_CLAIMS_PATH,
    GOLD_SECTIONS_PATH,
    RETRIEVAL_QUERIES_PATH,
    load_gold_claims,
    load_gold_sections,
    load_retrieval_queries,
)
from evals.llm_cache import DEFAULT_RECORDINGS_DIR, RecordingEmbeddings, RecordingLLM
from evals.report import build_model_run, infra_errors, render_markdown, summarize_sections, write_report
from evals.snapshot import DEFAULT_SNAPSHOT_PATH, Snapshot, load_snapshot, sample_chunks
from evals.tasks import (
    Scenario,
    run_claims_stage,
    run_retrieval_stage,
    run_review_stage,
    run_sections_stage,
    run_synthesis_stage,
    select_claim_pool,
)
from lit_review_assistant.llm import claims, review, synthesis

DEFAULT_SCENARIOS_PATH = Path("evals/datasets/review_scenarios.json")
DEFAULT_REPORTS_DIR = Path("evals/reports")
STAGES = ["claims", "retrieval", "synthesis", "review"]
T = TypeVar("T")


def load_scenarios(path: Path) -> list[Scenario]:
    return [Scenario(**scenario) for scenario in json.loads(path.read_text())]


def load_gold(loader: Callable[[Snapshot, Path], T], snapshot: Snapshot, path: Path) -> T | None:
    """Load a gold file, or warn and return None if it's missing or was labelled on a different snapshot."""
    try:
        return loader(snapshot, path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Skipping gold labels in {path}: {exc}", file=sys.stderr)
        return None


def git_state() -> tuple[str, bool]:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown", False
    return sha.stdout.strip(), bool(status.stdout.strip())


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", help="Chat model(s) to evaluate (repeatable).")
    parser.add_argument("--mode", choices=["record", "replay", "refresh"], default="record")
    parser.add_argument("--stop-after", choices=STAGES, default="review")

    parser.add_argument("--limit", type=int, default=30, help="Chunks to evaluate, spread across papers (0 = all).")
    parser.add_argument("--seed", type=int, default=0, help="Chunk sampling seed.")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent claim-extraction calls.")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT_PATH)
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS_PATH)
    parser.add_argument("--recordings", type=Path, default=DEFAULT_RECORDINGS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORTS_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)
    models = args.model or [os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")]
    snapshot = load_snapshot(args.snapshot)
    chunks = sample_chunks(snapshot, args.limit, args.seed)
    scenarios = load_scenarios(args.scenarios)
    gold_sections = load_gold(load_gold_sections, snapshot, GOLD_SECTIONS_PATH)
    gold_claims = load_gold(load_gold_claims, snapshot, GOLD_CLAIMS_PATH)
    queries = load_gold(load_retrieval_queries, snapshot, RETRIEVAL_QUERIES_PATH)
    run_through = STAGES[: STAGES.index(args.stop_after) + 1]

    sections = None
    if gold_sections is not None:
        section_cases = run_sections_stage(snapshot, gold_sections)
        sections = {"summary": summarize_sections(section_cases), "cases": section_cases}

    runs = []
    for model in models:
        llm = RecordingLLM(model, mode=args.mode, recordings_dir=args.recordings)
        embeddings = RecordingEmbeddings(mode=args.mode, recordings_dir=args.recordings)
        stages: dict[str, list[dict[str, Any]]] = {}

        print(f"[{model}] claims: {len(chunks)} chunk(s)...", file=sys.stderr)
        claims_result = run_claims_stage(snapshot, chunks, llm, workers=args.workers, gold=gold_claims)
        stages["claims"] = claims_result.cases

        if "retrieval" in run_through and queries:
            print(
                f"[{model}] retrieval: {len(queries)} queries over {len(claims_result.accepted_claims)} claim(s)...",
                file=sys.stderr,
            )
            stages["retrieval"] = run_retrieval_stage(snapshot, claims_result.accepted_claims, queries, embeddings)

        if "synthesis" in run_through:
            stages["synthesis"] = []
            if args.stop_after == "review":
                stages["review"] = []
            for scenario in scenarios:
                pool = select_claim_pool(claims_result.accepted_claims, scenario.max_claims)
                print(f"[{model}] synthesis: {scenario.id} ({len(pool)} claim(s))...", file=sys.stderr)
                cases, syntheses = run_synthesis_stage(scenario, pool, llm)
                stages["synthesis"].extend(cases)
                if args.stop_after == "review":
                    print(f"[{model}] review: {scenario.id} ({len(syntheses)} synthesis/es)...", file=sys.stderr)
                    stages["review"].append(run_review_stage(scenario, syntheses, claims_result.accepted_claims, llm))

        runs.append(build_model_run(model, stages, [*llm.calls, *embeddings.calls]))

    git_sha, git_dirty = git_state()
    report: dict[str, Any] = {
        "meta": {
            "run_id": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
            "git_sha": git_sha,
            "git_dirty": git_dirty,
            "mode": args.mode,
            "models": models,
            "snapshot_path": str(snapshot.path),
            "snapshot_fingerprint": snapshot.fingerprint,
            "snapshot_pymupdf_version": snapshot.pymupdf_version,
            # Production's embed_texts reads this from the environment; set OPENAI_EMBEDDING_MODEL to change it.
            "embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            "n_chunks": len(chunks),
            "limit": args.limit,
            "seed": args.seed,
            "scenarios": [scenario.id for scenario in scenarios],
            "prompt_versions": [claims.PROMPT_VERSION, synthesis.PROMPT_VERSION, review.PROMPT_VERSION],
            "temperatures": {
                claims.PROMPT_VERSION: claims.DEFAULT_TEMPERATURE,
                synthesis.PROMPT_VERSION: synthesis.DEFAULT_TEMPERATURE,
                review.PROMPT_VERSION: review.DEFAULT_TEMPERATURE,
            },
        },
        "sections": sections,
        "runs": runs,
    }
    run_dir = write_report(report, args.out)
    print(render_markdown(report))
    print(f"Report written to {run_dir}", file=sys.stderr)

    errors = infra_errors(runs)
    if errors:
        print(f"\n{len(errors)} case(s) could not be evaluated, e.g.: {errors[0]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
