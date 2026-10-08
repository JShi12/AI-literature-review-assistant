"""Measure how far the LLM judges can be trusted, against a frozen, labelled calibration set.

    python -m evals.calibrate                        # default judge model, record mode
    python -m evals.calibrate --judge-model gpt-4.1-mini --mode replay

The set (evals/datasets/judge_calibration.json) mixes real pipeline outputs, labelled independently,
with constructed negatives whose labels are known by construction: perturbed claims (wrong number,
negation, overgeneralization), synthesis bodies paired with another synthesis's citations, and
review sentences paired with another sentence's citations. A judge that agrees on the real items but
misses the constructed negatives is too lenient to be useful.

Judges are called the way the pipeline calls them: claims batched per chunk, review sentences in
batches, syntheses one at a time.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from evals.judges import (
    CLAIM_GROUNDING_VERSION,
    DEFAULT_JUDGE_MODEL,
    REVIEW_CITATIONS_VERSION,
    SYNTHESIS_VERSION,
    CitedClaim,
    judge_claims,
    judge_review_citations,
    judge_synthesis,
)
from evals.llm_cache import DEFAULT_RECORDINGS_DIR, RecordingLLM
from evals.snapshot import DEFAULT_SNAPSHOT_PATH, load_snapshot
from lit_review_assistant.pipeline.quotes import strip_control_chars

CALIBRATION_PATH = Path("evals/datasets/judge_calibration.json")
REVIEW_BATCH_SIZE = 12
SUPPORT_LABELS = ["supported", "partially_supported", "unsupported"]
FAITHFULNESS_LABELS = ["faithful", "partially_faithful", "unfaithful"]


def cohen_kappa(pairs: Sequence[tuple[str, str]], labels: Sequence[str]) -> float | None:
    """Chance-corrected agreement between reference and judge labels."""
    if not pairs:
        return None
    observed = sum(reference == predicted for reference, predicted in pairs) / len(pairs)
    reference_counts = Counter(reference for reference, _ in pairs)
    predicted_counts = Counter(predicted for _, predicted in pairs)
    expected = sum(reference_counts[label] * predicted_counts[label] for label in labels) / len(pairs) ** 2
    return round((observed - expected) / (1 - expected), 3) if expected < 1 else None


def agreement(items: Sequence[dict[str, Any]], labels: Sequence[str], positive: str) -> dict[str, Any]:
    """Agreement stats for items carrying a reference "label", a judge "predicted", and an "origin"."""
    scored = [item for item in items if item["predicted"] is not None]
    pairs = [(item["label"], item["predicted"]) for item in scored]
    negatives = [item for item in scored if item["origin"] != "real" and item["label"] != positive]

    def share(hits: int, total: int) -> float | None:
        return round(hits / total, 3) if total else None

    return {
        "n": len(items),
        "judge_skipped": len(items) - len(scored),
        "accuracy": share(sum(reference == predicted for reference, predicted in pairs), len(pairs)),
        # Collapses partial into "not fully supported", the distinction that matters most in practice.
        "binary_accuracy": share(
            sum((reference == positive) == (predicted == positive) for reference, predicted in pairs), len(pairs)
        ),
        "kappa": cohen_kappa(pairs, labels),
        "real_accuracy": share(
            sum(item["label"] == item["predicted"] for item in scored if item["origin"] == "real"),
            sum(item["origin"] == "real" for item in scored),
        ),
        # Of the deliberately broken items, how many the judge flagged as not fully supported.
        "constructed_negatives_caught": share(sum(item["predicted"] != positive for item in negatives), len(negatives)),
        "confusion": {
            reference: {predicted: sum(pair == (reference, predicted) for pair in pairs) for predicted in labels}
            for reference in labels
        },
    }


def calibrate(dataset: dict[str, Any], judge: RecordingLLM) -> dict[str, Any]:
    snapshot = load_snapshot(DEFAULT_SNAPSHOT_PATH)
    if dataset["snapshot_fingerprint"] != snapshot.fingerprint:
        raise ValueError("Calibration set was built against a different snapshot; rebuild the snapshot first.")
    chunk_text = {chunk.key: chunk.text for chunk in snapshot.all_chunks()}

    claim_items = [dict(item) for item in dataset["claim_grounding"]]
    by_chunk: dict[str, list[dict[str, Any]]] = {}
    for item in claim_items:
        by_chunk.setdefault(item["chunk_key"], []).append(item)
    for key, items in by_chunk.items():
        verdicts = judge_claims(strip_control_chars(chunk_text[key]), [item["claim_text"] for item in items], judge)
        for item, verdict in zip(items, verdicts, strict=True):
            item["predicted"] = verdict

    synthesis_items = [dict(item) for item in dataset["synthesis"]]
    for item in synthesis_items:
        cited = [CitedClaim(**claim) for claim in item["cited_claims"]]
        judgment = judge_synthesis(item["synthesis_type"], item["title"], item["body"], cited, judge)
        item["label"], item["predicted"] = item["faithfulness"], judgment.faithfulness
        item["predicted_type_appropriate"] = judgment.type_appropriate

    review_items = [dict(item) for item in dataset["review_citations"]]
    for start in range(0, len(review_items), REVIEW_BATCH_SIZE):
        batch = review_items[start : start + REVIEW_BATCH_SIZE]
        sentences = [(item["sentence"], [CitedClaim(**claim) for claim in item["cited_claims"]]) for item in batch]
        for item, verdict in zip(batch, judge_review_citations(sentences, judge), strict=True):
            item["predicted"] = verdict

    typed = [item for item in synthesis_items if item.get("type_appropriate") is not None]
    return {
        "claim_grounding": agreement(claim_items, SUPPORT_LABELS, "supported"),
        "synthesis": {
            **agreement(synthesis_items, FAITHFULNESS_LABELS, "faithful"),
            "type_appropriate_accuracy": (
                round(sum(i["type_appropriate"] == i["predicted_type_appropriate"] for i in typed) / len(typed), 3)
                if typed
                else None
            ),
        },
        "review_citations": agreement(review_items, SUPPORT_LABELS, "supported"),
        "items": {"claim_grounding": claim_items, "synthesis": synthesis_items, "review_citations": review_items},
    }


def render(result: dict[str, Any], judge_model: str) -> str:
    rows = [
        ("Items", "n", "int"),
        ("Exact label accuracy", "accuracy", "pct"),
        ("Supported vs not (binary) accuracy", "binary_accuracy", "pct"),
        ("Cohen's kappa", "kappa", "float"),
        ("Accuracy on real items", "real_accuracy", "pct"),
        ("Constructed negatives caught", "constructed_negatives_caught", "pct"),
        ("Synthesis type judgement accuracy", "type_appropriate_accuracy", "pct"),
    ]
    judges = ["claim_grounding", "synthesis", "review_citations"]
    lines = [
        f"# Judge calibration — `{judge_model}`",
        "",
        "| Metric | Claim grounding | Synthesis faithfulness | Review citations |",
        "|---|---|---|---|",
    ]
    for label, key, kind in rows:
        cells = []
        for name in judges:
            value = result[name].get(key)
            if value is None:
                cells.append("–")
            elif kind == "pct":
                cells.append(f"{value:.1%}")
            elif kind == "float":
                cells.append(f"{value:.2f}")
            else:
                cells.append(str(value))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    for name in judges:
        confusion = result[name]["confusion"]
        lines.extend(["", f"**{name}** confusion (rows = reference, columns = judge):", ""])
        labels = list(confusion)
        lines.append("| | " + " | ".join(labels) + " |")
        lines.append("|---|" + "---|" * len(labels))
        lines.extend(f"| {ref} | " + " | ".join(str(confusion[ref][p]) for p in labels) + " |" for ref in labels)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Score the LLM judges against the calibration set.")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--mode", choices=["record", "replay", "refresh"], default="record")
    # Separate from the pipeline's recordings, so `evals.run --prune` never deletes calibration recordings.
    parser.add_argument("--recordings", type=Path, default=DEFAULT_RECORDINGS_DIR / "calibration")
    parser.add_argument("--out", type=Path, default=Path("evals/reports"))
    args = parser.parse_args(argv)

    judge = RecordingLLM(args.judge_model, mode=args.mode, recordings_dir=args.recordings)
    result = calibrate(json.loads(CALIBRATION_PATH.read_text()), judge)
    result["meta"] = {
        "judge_model": args.judge_model,
        "judge_versions": [CLAIM_GROUNDING_VERSION, SYNTHESIS_VERSION, REVIEW_CITATIONS_VERSION],
    }
    run_dir = args.out / f"calibration-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "calibration.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    summary = render(result, args.judge_model)
    (run_dir / "summary.md").write_text(summary)
    print(summary)
    print(f"Written to {run_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
