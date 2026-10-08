"""Regression gate: compare an eval report with the committed baseline, failing if any metric got worse.

    python -m evals.gate                    # check the latest report against evals/baselines/baseline.json
    python -m evals.gate --update           # accept the latest report as the new baseline

CI replays the eval from committed recordings, then runs this gate, so replayed metrics are
deterministic: any change comes from code (post-processing, metrics, section detection) or from
new recordings. Changing a prompt therefore means re-recording live, reviewing the summary, and
running --update, so the PR shows the metric changes as a diff of the baseline file.

Only metrics with a direction (better up or down) in report.HEADLINE_METRICS are gated, plus
section detection. Informational metrics (counts, claims per chunk) are recorded but not gated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from evals.report import HEADLINE_METRICS

BASELINE_PATH = Path("evals/baselines/baseline.json")
REPORTS_DIR = Path("evals/reports")
# Run settings that must match for a comparison to mean anything.
CONFIG_KEYS = [
    "snapshot_fingerprint",
    "models",
    "judge_model",
    "n_chunks",
    "seed",
    "scenarios",
    "agent_cases",
    "prompt_versions",
]
# Latency only exists for live calls and is noise, not quality.
UNGATED = {"latency_mean_s"}
SECTION_DIRECTIONS = {
    "heading_precision": "up",
    "heading_recall": "up",
    "matched_type_accuracy": "up",
    "chunk_type_accuracy": "up",
}


def latest_report(reports_dir: Path = REPORTS_DIR) -> Path:
    runs = sorted(path for path in reports_dir.glob("*/report.json") if not path.parent.name.startswith("calibration"))
    if not runs:
        raise FileNotFoundError(f"No eval reports in {reports_dir}; run python -m evals.run first.")
    return runs[-1]


def gated_metrics(report: dict[str, Any]) -> dict[str, tuple[float | None, str]]:
    """Flatten a report into {"model/stage/metric": (value, direction)} for every gated metric."""
    metrics: dict[str, tuple[float | None, str]] = {}
    for run in report["runs"]:
        for stage, _, key, _, better in HEADLINE_METRICS:
            summary = run["summary"].get(stage)
            if better and summary is not None and key in summary and key not in UNGATED:
                metrics[f"{run['model']}/{stage}/{key}"] = (summary[key], better)
    sections = report.get("sections")
    if sections:
        for key, better in SECTION_DIRECTIONS.items():
            metrics[f"sections/{key}"] = (sections["summary"].get(key), better)
    return metrics


def _as_number(value: Any) -> float | None:
    if isinstance(value, list):  # e.g. invalid_citation_numbers: a list whose length is what matters
        return float(len(value))
    return None if value is None else float(value)


def _status(actual: float, expected: float, direction: str, tolerance: float) -> str:
    # Flip the sign for lower-is-better metrics so "gain" is always positive.
    gain = actual - expected if direction == "up" else expected - actual
    if gain < -tolerance:
        return "regressed"
    return "improved" if gain > 0 else "ok"


def compare(
    current: dict[str, tuple[float | None, str]], baseline: dict[str, Any], tolerance: float
) -> list[dict[str, Any]]:
    rows = []
    for name, expected in baseline["metrics"].items():
        direction = baseline["directions"][name]
        actual = _as_number(current[name][0]) if name in current else None
        expected_value = _as_number(expected)
        if expected_value is None:
            status = "ok"
        elif actual is None:
            status = "missing"
        else:
            status = _status(actual, expected_value, direction, tolerance)
        rows.append({"metric": name, "baseline": expected_value, "current": actual, "status": status})
    rows.extend(
        {"metric": name, "baseline": None, "current": _as_number(value), "status": "new"}
        for name, (value, _) in current.items()
        if name not in baseline["metrics"]
    )
    return rows


def config_mismatches(report: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    return [
        f"{key}: baseline {baseline['config'].get(key)!r}, report {report['meta'].get(key)!r}"
        for key in CONFIG_KEYS
        if baseline["config"].get(key) != report["meta"].get(key)
    ]


def write_baseline(report: dict[str, Any], path: Path) -> None:
    metrics = gated_metrics(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {key: report["meta"].get(key) for key in CONFIG_KEYS},
        "metrics": {name: value for name, (value, _) in sorted(metrics.items())},
        "directions": {name: direction for name, (_, direction) in sorted(metrics.items())},
    }
    path.write_text(json.dumps(payload, indent=1) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail if eval metrics regressed against the baseline.")
    parser.add_argument("--report", type=Path, help="report.json to check (default: the latest eval report)")
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    parser.add_argument("--tolerance", type=float, default=0.0, help="Allowed slack before a drop counts.")
    parser.add_argument("--update", action="store_true", help="Write the report's metrics as the new baseline.")
    args = parser.parse_args(argv)

    report_path = args.report or latest_report()
    report = json.loads(report_path.read_text())
    if args.update:
        write_baseline(report, args.baseline)
        print(f"Baseline updated from {report_path} -> {args.baseline}")
        return 0

    baseline = json.loads(args.baseline.read_text())
    mismatches = config_mismatches(report, baseline)
    if mismatches:
        print("Report and baseline were produced with different settings, so they can't be compared:")
        print("\n".join(f"  - {line}" for line in mismatches))
        print("Re-run with the baseline's settings, or accept new settings with: python -m evals.gate --update")
        return 1

    rows = compare(gated_metrics(report), baseline, args.tolerance)
    failed = [row for row in rows if row["status"] in ("regressed", "missing")]
    for row in rows:
        if row["status"] != "ok":
            print(f"{row['status']:>9}  {row['metric']}: {row['baseline']} -> {row['current']}")
    print(f"{len(rows)} gated metric(s): {len(failed)} regressed or missing (report {report_path}).")
    if failed:
        print(
            "If this change is intended (e.g. a prompt change you re-recorded and reviewed), accept it with: "
            "python -m evals.gate --update"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
