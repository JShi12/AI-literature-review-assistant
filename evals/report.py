"""Aggregate per-case eval results into per-model summaries, and render/write the run report."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from decimal import Decimal
from pathlib import Path
from statistics import mean
from typing import Any

from evals.llm_cache import CallRecord

Case = dict[str, Any]

# (stage, label, key, format, better) -- the rows of the Markdown summary table. `better` is a hint
# for the reader ("up"/"down"); it isn't enforced anywhere yet.
SECTION_METRICS: list[tuple[str, str, str]] = [
    ("Gold headings", "n_gold", "int"),
    ("Detected headings", "n_detected", "int"),
    ("Heading precision", "heading_precision", "pct"),
    ("Heading recall", "heading_recall", "pct"),
    ("Matched headings with correct type", "matched_type_accuracy", "pct"),
    ("Chunks with correct section type", "chunk_type_accuracy", "pct"),
]

HEADLINE_METRICS: list[tuple[str, str, str, str, str]] = [
    ("claims", "Chunks evaluated", "chunks", "int", ""),
    ("claims", "Chunks errored", "errors", "int", "down"),
    ("claims", "Claims per chunk", "claims_per_chunk", "float", ""),
    ("claims", "Empty chunks", "empty_chunk_rate", "pct", ""),
    ("claims", "Chunks rejected by location check", "rejected_chunk_rate", "pct", "down"),
    ("claims", "Claim offsets inside chunk", "location_valid_rate", "pct", "up"),
    ("claims", "Claim offsets point at the claim (span match)", "span_match_rate", "pct", "up"),
    ("claims", "Mean span similarity", "span_similarity_mean", "float", "up"),
    ("claims", "Verbatim claims", "verbatim_rate", "pct", ""),
    ("claims", "Mean lexical coverage of chunk", "lexical_coverage_mean", "float", "up"),
    ("claims", "Claims dropped: quote not found in chunk", "unlocated_rate", "pct", "down"),
    ("claims", "Gold chunks scored", "gold_chunks", "int", ""),
    ("claims", "Gold claims recalled", "gold_recall", "pct", "up"),
    ("claims", "Extracted claims matching a gold claim", "gold_precision", "pct", "up"),
    ("claims", "Gold F1", "gold_f1", "float", "up"),
    ("claims", "Claim type agrees with gold", "gold_type_agreement", "pct", "up"),
    ("retrieval", "Queries scored", "queries_scored", "int", ""),
    ("retrieval", "Precision@5", "p@5", "pct", "up"),
    ("retrieval", "Precision@10", "p@10", "pct", "up"),
    ("retrieval", "Recall@20", "r@20", "pct", "up"),
    ("retrieval", "MRR", "mrr", "float", "up"),
    ("retrieval", "nDCG@10", "ndcg@10", "float", "up"),
    ("retrieval", "Random-ranking precision (baseline)", "random_precision", "pct", ""),
    ("synthesis", "Syntheses generated", "syntheses", "int", ""),
    ("synthesis", "Invented claim ids (of cited)", "invented_id_rate", "pct", "down"),
    ("synthesis", "Syntheses with no valid support", "unsupported_rate", "pct", "down"),
    ("synthesis", "Type matches request", "type_match_rate", "pct", "up"),
    ("synthesis", "Draws on 2+ papers", "multi_paper_rate", "pct", "up"),
    ("synthesis", "Self-reported counts wrong", "count_mismatch_rate", "pct", "down"),
    ("review", "Drafts generated", "drafts", "int", ""),
    ("review", "Structured sentences with linked claims", "sentence_supported_rate", "pct", "up"),
    ("review", "Phantom-supported sentences", "sentence_phantom_support_rate", "pct", "down"),
    ("review", "Invented claim ids (of cited)", "sentence_invented_id_rate", "pct", "down"),
    ("review", "Structured sentences found in Markdown", "sentence_alignment_rate", "pct", "up"),
    ("review", "Substantive Markdown sentences cited", "body_citation_coverage", "pct", "up"),
    ("review", "Invalid citation numbers", "invalid_citation_numbers", "int", "down"),
    ("review", "Leaked ids in Markdown", "leaked_ids", "int", "down"),
    ("review", "References section complete", "references_complete_rate", "pct", "up"),
    ("usage", "LLM calls (recorded / live)", "calls", "calls", ""),
    ("usage", "Input / output tokens", "tokens", "tokens", ""),
    ("usage", "Cost to run live (USD)", "cost_usd", "usd", "down"),
    ("usage", "Mean live latency (s)", "latency_mean_s", "float", "down"),
]


def _rate(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 3) if denominator else None


def _mean(values: Iterable[float]) -> float | None:
    collected = list(values)
    return round(mean(collected), 3) if collected else None


def _ok(cases: Sequence[Case]) -> list[Case]:
    return [case for case in cases if "error" not in case]


def summarize_claims(cases: Sequence[Case]) -> dict[str, Any]:
    scored = _ok(cases)
    claims = [claim for case in scored for claim in case["claims"]]
    return {
        "chunks": len(cases),
        "errors": len(cases) - len(scored),
        "claims": len(claims),
        "claims_per_chunk": _rate(len(claims), len(scored)),
        "empty_chunk_rate": _rate(sum(bool(case["empty"]) for case in scored), len(scored)),
        "rejected_chunk_rate": _rate(sum(bool(case["rejected"]) for case in scored), len(scored)),
        "location_valid_rate": _rate(sum(claim["location_valid"] for claim in claims), len(claims)),
        "span_match_rate": _rate(sum(claim["span_match"] for claim in claims), len(claims)),
        "span_similarity_mean": _mean(claim["span_similarity"] for claim in claims),
        "verbatim_rate": _rate(sum(claim["verbatim"] for claim in claims), len(claims)),
        "lexical_coverage_mean": _mean(claim["lexical_coverage"] for claim in claims),
        "unlocated_rate": _rate(
            sum(case["n_unlocated"] for case in scored), len(claims) + sum(case["n_unlocated"] for case in scored)
        ),
        "confidence_mean": _mean(claim["confidence"] for claim in claims),
        "claim_types": dict(Counter(claim["claim_type"] for claim in claims).most_common()),
        **summarize_gold_claims([case["gold"] for case in scored if "gold" in case]),
    }


def summarize_gold_claims(golds: Sequence[Case]) -> dict[str, Any]:
    n_gold = sum(gold["n_gold"] for gold in golds)
    n_predicted = sum(gold["n_predicted"] for gold in golds)
    gold_matched = sum(gold["gold_matched"] for gold in golds)
    recall = _rate(gold_matched, n_gold)
    precision = _rate(sum(gold["predicted_matched"] for gold in golds), n_predicted)
    return {
        "gold_chunks": len(golds),
        "gold_recall": recall,
        "gold_precision": precision,
        "gold_f1": round(2 * recall * precision / (recall + precision), 3) if recall and precision else None,
        "gold_type_agreement": _rate(sum(gold["type_agree"] for gold in golds), gold_matched),
        "gold_predicted_unlocatable": sum(gold["predicted_unlocatable"] for gold in golds),
    }


def summarize_sections(cases: Sequence[Case]) -> dict[str, Any]:
    matched = sum(case["headings_matched"] for case in cases)
    return {
        "papers": len(cases),
        "n_gold": sum(case["n_gold"] for case in cases),
        "n_detected": sum(case["n_detected"] for case in cases),
        "heading_precision": _rate(matched, sum(case["n_detected"] for case in cases)),
        "heading_recall": _rate(matched, sum(case["n_gold"] for case in cases)),
        "matched_type_accuracy": _rate(sum(case["matched_type_correct"] for case in cases), matched),
        "chunk_type_accuracy": _rate(
            sum(case["chunks_type_correct"] for case in cases), sum(case["n_chunks"] for case in cases)
        ),
    }


def summarize_retrieval(cases: Sequence[Case]) -> dict[str, Any]:
    scored = [case for case in _ok(cases) if case["n_relevant"] > 0]
    summary: dict[str, Any] = {
        "queries": len(cases),
        "errors": len(cases) - len(_ok(cases)),
        "queries_scored": len(scored),
    }
    for key in ("p@5", "p@10", "r@10", "r@20", "mrr", "ndcg@10", "random_precision"):
        summary[key] = _mean(case[key] for case in scored)
    return summary


def summarize_syntheses(cases: Sequence[Case]) -> dict[str, Any]:
    scored = _ok(cases)
    items = [item for case in scored for item in case["syntheses"]]
    cited = sum(item["n_cited"] for item in items)
    return {
        "calls": len(cases),
        "errors": len(cases) - len(scored),
        "syntheses": len(items),
        "empty_call_rate": _rate(sum(case["n_syntheses"] == 0 for case in scored), len(scored)),
        "claims_cited_mean": _mean(item["n_cited"] for item in items),
        "invented_id_rate": _rate(sum(item["n_invented"] for item in items), cited),
        "unsupported_rate": _rate(sum(item["unsupported"] for item in items), len(items)),
        "type_match_rate": _rate(sum(item["type_matches"] for item in items), len(items)),
        "multi_paper_rate": _rate(sum(item["multi_paper"] for item in items), len(items)),
        "count_mismatch_rate": _rate(sum(item["count_mismatch"] for item in items), len(items)),
    }


def summarize_reviews(cases: Sequence[Case]) -> dict[str, Any]:
    scored = _ok(cases)
    rate_keys = [
        "sentence_supported_rate",
        "sentence_phantom_support_rate",
        "sentence_invented_id_rate",
        "sentence_flag_mismatch_rate",
        "sentence_alignment_rate",
        "body_citation_coverage",
    ]
    summary: dict[str, Any] = {"drafts": len(scored), "errors": len(cases) - len(scored)}
    for key in rate_keys:
        summary[key] = _mean(case[key] for case in scored if case[key] is not None)
    summary["invalid_citation_numbers"] = sum(len(case["invalid_citation_numbers"]) for case in scored)
    summary["leaked_ids"] = sum(case["leaked_ids"] for case in scored)
    summary["references_complete_rate"] = _rate(sum(bool(case["references_complete"]) for case in scored), len(scored))
    return summary


def summarize_usage(calls: Sequence[CallRecord]) -> dict[str, Any]:
    costs = [call.cost for call in calls if call.cost is not None]
    live_latencies = [call.latency_s for call in calls if call.latency_s is not None]
    return {
        "calls": len(calls),
        "cache_hits": sum(call.cache_hit for call in calls),
        "input_tokens": sum(call.input_tokens for call in calls),
        "output_tokens": sum(call.output_tokens for call in calls),
        # What the run's calls cost (or would cost live, for replayed ones). None if the model isn't priced.
        "cost_usd": float(sum(costs, Decimal(0))) if len(costs) == len(calls) else None,
        "latency_mean_s": _mean(live_latencies),
    }


def build_model_run(model: str, stages: dict[str, list[Case]], calls: Sequence[CallRecord]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    if "claims" in stages:
        summaries["claims"] = summarize_claims(stages["claims"])
    if "retrieval" in stages:
        summaries["retrieval"] = summarize_retrieval(stages["retrieval"])
    if "synthesis" in stages:
        summaries["synthesis"] = summarize_syntheses(stages["synthesis"])
    if "review" in stages:
        summaries["review"] = summarize_reviews(stages["review"])
    summaries["usage"] = summarize_usage(calls)
    return {"model": model, "summary": summaries, "cases": stages}


def infra_errors(runs: Sequence[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for run in runs:
        for stage_cases in run["cases"].values():
            errors.extend(str(case["error"]) for case in stage_cases if case.get("error_kind") == "infra")
    return errors


# --- Rendering ------------------------------------------------------------------------------------


def _format(value: Any, kind: str, summary: dict[str, Any]) -> str:
    if kind == "calls":
        live = summary["calls"] - summary["cache_hits"]
        return f"{summary['calls']} ({summary['cache_hits']} recorded / {live} live)"
    if kind == "tokens":
        return f"{summary['input_tokens']:,} / {summary['output_tokens']:,}"
    if value is None:
        return "–"
    if kind == "pct":
        return f"{float(value):.1%}"
    if kind == "float":
        return f"{float(value):.2f}"
    if kind == "usd":
        return f"${float(value):.4f}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    meta: dict[str, Any] = report["meta"]
    runs: list[dict[str, Any]] = report["runs"]
    models = [str(run["model"]) for run in runs]

    lines = [
        f"# Eval run {meta['run_id']}",
        "",
        f"- git: `{meta['git_sha']}`{' (dirty)' if meta['git_dirty'] else ''} · mode: `{meta['mode']}`",
        f"- snapshot: `{meta['snapshot_path']}` (fingerprint `{meta['snapshot_fingerprint']}`), "
        f"{meta['n_chunks']} chunk(s), seed {meta['seed']}",
        f"- prompts: {', '.join(f'`{version}`' for version in meta['prompt_versions'])}",
        "",
        "| Stage | Metric | " + " | ".join(f"`{model}`" for model in models) + " |",
        "|---|---|" + "---|" * len(models),
    ]
    for stage, label, key, kind, better in HEADLINE_METRICS:
        summaries = [run["summary"].get(stage) for run in runs]
        if all(summary is None for summary in summaries):
            continue
        arrow = {"up": " ↑", "down": " ↓"}.get(better, "")
        cells = [_format(summary.get(key), kind, summary) if summary else "–" for summary in summaries]
        lines.append(f"| {stage} | {label}{arrow} | " + " | ".join(cells) + " |")

    sections = report.get("sections")
    if sections:
        lines.extend(["", "## Section detection (model-independent)", "", "| Metric | Value |", "|---|---|"])
        lines.extend(
            f"| {label} | {_format(sections['summary'].get(key), kind, sections['summary'])} |"
            for label, key, kind in SECTION_METRICS
        )
        missed = [f"`{case['paper_key']}`: {', '.join(case['missed_headings'])}" for case in sections["cases"]]
        lines.extend(["", "Missed headings:", "", *(f"- {line}" for line in missed)])

    for run in runs:
        worst = _worst_claim_locations(run["cases"].get("claims", []))
        if worst:
            lines.extend(["", f"## Lowest claim span matches — `{run['model']}`", ""])
            lines.extend(worst)
        errors = [
            f"- `{_case_label(case)}` ({case['error_kind']}): {case['error']}"
            for cases in run["cases"].values()
            for case in cases
            if "error" in case
        ]
        if errors:
            lines.extend(["", f"## Errors — `{run['model']}`", "", *errors[:20]])
    return "\n".join(lines) + "\n"


def _case_label(case: Case) -> str:
    if "chunk_key" in case:
        return str(case["chunk_key"])
    if "query_id" in case:
        return str(case["query_id"])
    return ":".join(str(case[key]) for key in ("scenario", "synthesis_type") if key in case)


def _worst_claim_locations(cases: Sequence[Case], limit: int = 5) -> list[str]:
    claims = [(case["chunk_key"], claim) for case in _ok(cases) for claim in case["claims"] if not claim["span_match"]]
    claims.sort(key=lambda item: item[1]["span_similarity"])
    return [
        f"- `{chunk_key}` p{claim['page']} [{claim['start_char']}:{claim['end_char']}] "
        f"similarity {claim['span_similarity']:.2f}\n"
        f"  - claim: {_clip(str(claim['claim_text']))}\n"
        f"  - span:  {_clip(str(claim['span_text'])) or '(empty / out of range)'}"
        for chunk_key, claim in claims[:limit]
    ]


def _clip(text: str, width: int = 140) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def write_report(report: dict[str, Any], out_dir: Path) -> Path:
    run_dir = out_dir / str(report["meta"]["run_id"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=_json_default))
    (run_dir / "summary.md").write_text(render_markdown(report))
    return run_dir


def _json_default(value: object) -> object:
    if isinstance(value, Decimal):
        return float(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")
