"""Gold labels: section headings, reference claims, and retrieval relevance, all keyed to a snapshot.

Labels refer to snapshot pages and chunk keys, so each file records the snapshot fingerprint it was
labelled against; loading them against a different snapshot is refused rather than silently scoring
against shifted chunks.

    python -m evals.gold validate                     # check the committed gold files
    python -m evals.gold validate --partial labels.json   # check one labeller's partial file
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from evals.metrics import locate_quote
from evals.snapshot import DEFAULT_SNAPSHOT_PATH, Snapshot, load_snapshot

GOLD_SECTIONS_PATH = Path("evals/datasets/gold_sections.json")
GOLD_CLAIMS_PATH = Path("evals/datasets/gold_claims.json")
RETRIEVAL_QUERIES_PATH = Path("evals/datasets/retrieval_queries.json")

SECTION_TYPES = {
    "abstract",
    "introduction",
    "related_work",
    "methods",
    "results",
    "discussion",
    "limitations",
    "conclusion",
    "other",
}
CLAIM_TYPES = {"finding", "method", "dataset", "metric", "limitation", "future_work", "background", "other"}


@dataclass(frozen=True)
class GoldHeading:
    paper_key: str
    page: int
    heading: str
    normalized_type: str
    start_char: int


@dataclass(frozen=True)
class GoldClaim:
    chunk_key: str
    quote: str
    claim_type: str


@dataclass(frozen=True)
class RetrievalQuery:
    id: str
    query: str
    relevant_chunks: frozenset[str]


def locate_line(page_text: str, heading: str) -> int | None:
    """Return the page offset of the first line whose stripped text is `heading`, like detect_sections."""
    cursor = 0
    for line in page_text.splitlines(keepends=True):
        if line.strip() == heading.strip():
            return cursor
        cursor += len(line)
    return None


# --- Loading --------------------------------------------------------------------------------------


def _check_fingerprint(raw: dict[str, object], path: Path, snapshot: Snapshot) -> None:
    if raw.get("snapshot_fingerprint") != snapshot.fingerprint:
        raise ValueError(
            f"{path} was labelled against snapshot {raw.get('snapshot_fingerprint')!r}, but the loaded snapshot "
            f"is {snapshot.fingerprint!r}. Rebuild the snapshot from the same PDFs, or relabel."
        )


def load_gold_sections(snapshot: Snapshot, path: Path = GOLD_SECTIONS_PATH) -> dict[str, list[GoldHeading]]:
    raw = json.loads(path.read_text())
    _check_fingerprint(raw, path, snapshot)
    gold: dict[str, list[GoldHeading]] = {}
    for paper_key, headings in raw["papers"].items():
        pages = snapshot.paper(paper_key).pages
        gold[paper_key] = [
            GoldHeading(
                paper_key=paper_key,
                page=item["page"],
                heading=item["heading"],
                normalized_type=item["normalized_type"],
                start_char=locate_line(pages[item["page"]], item["heading"]) or 0,
            )
            for item in headings
        ]
    return gold


def load_gold_claims(snapshot: Snapshot, path: Path = GOLD_CLAIMS_PATH) -> dict[str, list[GoldClaim]]:
    raw = json.loads(path.read_text())
    _check_fingerprint(raw, path, snapshot)
    return {
        chunk_key: [GoldClaim(chunk_key, item["quote"], item["claim_type"]) for item in claims]
        for chunk_key, claims in raw["chunks"].items()
    }


def load_retrieval_queries(snapshot: Snapshot, path: Path = RETRIEVAL_QUERIES_PATH) -> list[RetrievalQuery]:
    raw = json.loads(path.read_text())
    _check_fingerprint(raw, path, snapshot)
    return [
        RetrievalQuery(id=item["id"], query=item["query"], relevant_chunks=frozenset(item["relevant_chunks"]))
        for item in raw["queries"]
    ]


# --- Validation -----------------------------------------------------------------------------------


def validate_sections(snapshot: Snapshot, papers: dict[str, list[dict[str, object]]]) -> list[str]:
    problems = []
    for paper_key, headings in papers.items():
        paper = next((paper for paper in snapshot.papers if paper.key == paper_key), None)
        if paper is None:
            problems.append(f"sections: unknown paper {paper_key!r}")
            continue
        for item in headings:
            page, heading, kind = item.get("page"), str(item.get("heading")), item.get("normalized_type")
            if kind not in SECTION_TYPES:
                problems.append(f"sections[{paper_key}]: {heading!r} has invalid normalized_type {kind!r}")
            if not isinstance(page, int) or page not in paper.pages:
                problems.append(f"sections[{paper_key}]: {heading!r} has invalid page {page!r}")
            elif locate_line(paper.pages[page], heading) is None:
                problems.append(f"sections[{paper_key}]: no line equal to {heading!r} on page {page}")
    return problems


def validate_claims(snapshot: Snapshot, chunks: dict[str, list[dict[str, object]]]) -> list[str]:
    problems = []
    chunk_text = {chunk.key: chunk.text for chunk in snapshot.all_chunks()}
    for chunk_key, claims in chunks.items():
        if chunk_key not in chunk_text:
            problems.append(f"claims: unknown chunk {chunk_key!r}")
            continue
        for item in claims:
            quote, kind = str(item.get("quote")), item.get("claim_type")
            if kind not in CLAIM_TYPES:
                problems.append(f"claims[{chunk_key}]: invalid claim_type {kind!r} for {quote[:60]!r}")
            if locate_quote(chunk_text[chunk_key], quote) is None:
                problems.append(f"claims[{chunk_key}]: quote not found verbatim in chunk: {quote[:80]!r}")
    return problems


def validate_relevance(snapshot: Snapshot, relevance: dict[str, list[str]], query_ids: set[str]) -> list[str]:
    problems = []
    keys = {chunk.key for chunk in snapshot.all_chunks()}
    for query_id, chunk_keys in relevance.items():
        if query_id not in query_ids:
            problems.append(f"relevance: unknown query id {query_id!r}")
        problems.extend(f"relevance[{query_id}]: unknown chunk {key!r}" for key in chunk_keys if key not in keys)
    return problems


def validate_all(snapshot: Snapshot) -> list[str]:
    problems: list[str] = []
    for path in (GOLD_SECTIONS_PATH, GOLD_CLAIMS_PATH, RETRIEVAL_QUERIES_PATH):
        raw = json.loads(path.read_text())
        if raw.get("snapshot_fingerprint") != snapshot.fingerprint:
            problems.append(f"{path}: snapshot fingerprint mismatch")
    problems += validate_sections(snapshot, json.loads(GOLD_SECTIONS_PATH.read_text())["papers"])
    problems += validate_claims(snapshot, json.loads(GOLD_CLAIMS_PATH.read_text())["chunks"])
    queries = json.loads(RETRIEVAL_QUERIES_PATH.read_text())["queries"]
    problems += validate_relevance(
        snapshot, {q["id"]: q["relevant_chunks"] for q in queries}, {q["id"] for q in queries}
    )
    return problems


def validate_partial(snapshot: Snapshot, path: Path) -> list[str]:
    """Validate one labeller's file: {"paper_key", "sections": [...], "claims": {...}, "relevance": {...}}."""
    raw = json.loads(path.read_text())
    query_ids = {query["id"] for query in json.loads(RETRIEVAL_QUERIES_PATH.read_text())["queries"]}
    problems = validate_sections(snapshot, {raw["paper_key"]: raw.get("sections", [])})
    problems += validate_claims(snapshot, raw.get("claims", {}))
    problems += validate_relevance(snapshot, raw.get("relevance", {}), query_ids)
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate gold label files against the snapshot.")
    parser.add_argument("command", choices=["validate"])
    parser.add_argument("--partial", type=Path, help="Validate one labeller's partial file instead.")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT_PATH)
    args = parser.parse_args()

    snapshot = load_snapshot(args.snapshot)
    problems = validate_partial(snapshot, args.partial) if args.partial else validate_all(snapshot)
    for problem in problems:
        print(problem)
    print(f"{len(problems)} problem(s)", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
