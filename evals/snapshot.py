"""Frozen, DB-free snapshot of ingested papers (page text + section-aware chunks) used as eval input.

Freezing the ingestion output means a change to chunking or section detection can't silently change
what the LLM stages are evaluated on; rebuild the snapshot deliberately when you want that.

Build it (downloads the same three arXiv papers the live demo is seeded with):

    python -m evals.snapshot

or from local PDFs:

    python -m evals.snapshot --pdf data/uploads/a.pdf --pdf data/uploads/b.pdf --out evals/snapshots/mine.json

The snapshot is gitignored: it contains the papers' full extracted text, which is the papers' authors'
to redistribute, not this repo's. It is cheap to rebuild, and its fingerprint is stamped on every eval
report so runs over different inputs are never compared by accident.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from lit_review_assistant.pipeline.chunking import chunk_pages_with_sections
from lit_review_assistant.pipeline.pdf import extract_pages, extract_pdf_metadata, infer_paper_metadata_from_name
from lit_review_assistant.pipeline.sections import detect_sections
from lit_review_assistant.services import sha256_file

SNAPSHOT_VERSION = 1
DEFAULT_SNAPSHOT_PATH = Path("evals/snapshots/demo_papers.json")

# Fixed namespace so every id derived from a snapshot key is identical across runs and machines. Ids are
# real UUIDs (not short keys) because production code treats UUID-shaped text specially, e.g.
# review.UUID_PATTERN for citation cleanup.
ID_NAMESPACE = uuid.UUID("6f1c2a52-5d4e-4f0e-9a51-6c0b1e7d2f10")


def stable_id(kind: str, key: str) -> str:
    return str(uuid.uuid5(ID_NAMESPACE, f"{kind}:{key}"))


@dataclass(frozen=True)
class SnapshotChunk:
    key: str
    paper_key: str
    section_title: str | None
    section_type: str | None
    page_start: int
    page_end: int
    start_char: int
    end_char: int
    text: str

    @property
    def id(self) -> str:
        return stable_id("chunk", self.key)

    @property
    def paper_id(self) -> str:
        return stable_id("paper", self.paper_key)

    @property
    def section_id(self) -> str | None:
        if not self.section_type:
            return None
        return stable_id("section", f"{self.paper_key}:{self.section_title}:{self.section_type}")


@dataclass(frozen=True)
class SnapshotPaper:
    key: str
    title: str | None
    authors: list[str]
    year: int | None
    file_name: str
    pdf_sha256: str
    pages: dict[int, str]
    chunks: list[SnapshotChunk] = field(default_factory=list)

    @property
    def id(self) -> str:
        return stable_id("paper", self.key)


@dataclass(frozen=True)
class Snapshot:
    path: Path
    fingerprint: str
    created_at: str
    pymupdf_version: str
    papers: list[SnapshotPaper]

    def paper(self, key: str) -> SnapshotPaper:
        return next(paper for paper in self.papers if paper.key == key)

    def all_chunks(self) -> list[SnapshotChunk]:
        return [chunk for paper in self.papers for chunk in paper.chunks]


def load_snapshot(path: str | Path = DEFAULT_SNAPSHOT_PATH) -> Snapshot:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Snapshot not found at {path}. Build it first: python -m evals.snapshot")
    raw = json.loads(path.read_text())
    if raw.get("version") != SNAPSHOT_VERSION:
        raise ValueError(f"Unsupported snapshot version {raw.get('version')!r}; rebuild with python -m evals.snapshot")

    papers = []
    for paper in raw["papers"]:
        chunks = [SnapshotChunk(paper_key=paper["key"], **chunk) for chunk in paper["chunks"]]
        papers.append(
            SnapshotPaper(
                key=paper["key"],
                title=paper["title"],
                authors=paper["authors"],
                year=paper["year"],
                file_name=paper["file_name"],
                pdf_sha256=paper["pdf_sha256"],
                pages={int(number): text for number, text in paper["pages"].items()},
                chunks=chunks,
            )
        )
    return Snapshot(
        path=path,
        fingerprint=fingerprint(raw["papers"]),
        created_at=raw["created_at"],
        pymupdf_version=raw["pymupdf_version"],
        papers=papers,
    )


def fingerprint(papers_json: list[dict[str, Any]]) -> str:
    """Hash what labels and recordings refer to: page text plus chunk keys and text.

    Derived metadata (section types, titles, authors) is left out, so rebuilding the snapshot after
    improving section detection doesn't invalidate gold labels that only point at pages and chunks.
    """
    material = [
        {
            "key": paper["key"],
            "pages": paper["pages"],
            "chunks": [[chunk["key"], chunk["text"]] for chunk in paper["chunks"]],
        }
        for paper in papers_json
    ]
    canonical = json.dumps(material, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(canonical).hexdigest()[:16]


def sample_chunks(snapshot: Snapshot, limit: int, seed: int = 0) -> list[SnapshotChunk]:
    """Pick up to `limit` chunks (0 = all), spread evenly across papers, reproducibly for a given seed."""
    shuffled: list[list[SnapshotChunk]] = []
    for paper in snapshot.papers:
        chunks = list(paper.chunks)
        random.Random(f"{seed}:{paper.key}").shuffle(chunks)
        shuffled.append(chunks)

    total = sum(len(chunks) for chunks in shuffled)
    target = total if limit <= 0 else min(limit, total)
    selected: list[SnapshotChunk] = []
    index = 0
    while len(selected) < target:
        for chunks in shuffled:
            if index < len(chunks) and len(selected) < target:
                selected.append(chunks[index])
        index += 1

    order = {paper.key: position for position, paper in enumerate(snapshot.papers)}
    return sorted(selected, key=lambda chunk: (order[chunk.paper_key], chunk.page_start, chunk.start_char))


def build_paper_record(
    pdf_path: Path, file_name: str, known_metadata: dict[str, dict[str, object]]
) -> dict[str, object]:
    """Run the production ingestion steps (minus the database) on one PDF."""
    paper_key = Path(file_name).stem
    pages = extract_pages(pdf_path)
    sections = detect_sections(pages)
    # Same defaults ingest_pdf uses (max_chars=1_500, overlap=150).
    chunks = chunk_pages_with_sections(pages, sections)

    pdf_metadata = extract_pdf_metadata(pdf_path)
    name_metadata = infer_paper_metadata_from_name(file_name)
    known = known_metadata.get(file_name, {})
    return {
        "key": paper_key,
        "title": known.get("title") or pdf_metadata.title or name_metadata.title or paper_key,
        "authors": known.get("authors") or pdf_metadata.authors or name_metadata.authors or [],
        "year": known.get("year") or pdf_metadata.year or name_metadata.year,
        "file_name": file_name,
        "pdf_sha256": sha256_file(pdf_path),
        "pages": {str(page.page_number): page.text for page in pages},
        "chunks": [
            {
                "key": f"{paper_key}:p{chunk.page_start}:{chunk.start_char}-{chunk.end_char}",
                "section_title": chunk.section_title,
                "section_type": chunk.section_type,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "text": chunk.text,
            }
            for chunk in chunks
        ],
    }


def write_snapshot(papers: list[dict[str, object]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": SNAPSHOT_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "pymupdf_version": version("pymupdf"),
        "papers": papers,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a frozen eval snapshot of ingested papers.")
    parser.add_argument("--pdf", action="append", type=Path, help="Local PDF(s) to use instead of the demo papers.")
    parser.add_argument("--out", type=Path, default=DEFAULT_SNAPSHOT_PATH)
    parser.add_argument(
        "--pdf-cache", type=Path, help="Keep downloaded demo PDFs here and reuse them (CI caches this directory)."
    )
    args = parser.parse_args()

    # Reuse the demo seed script's paper list, download helper, and curated metadata so the eval set is
    # exactly the data the live demo shows.
    from scripts.seed_demo_data import DEMO_PAPERS, KNOWN_METADATA, download

    papers: list[dict[str, object]] = []
    if args.pdf:
        for pdf in args.pdf:
            papers.append(build_paper_record(pdf, pdf.name, KNOWN_METADATA))
    else:
        with tempfile.TemporaryDirectory() as tmp:
            directory = args.pdf_cache or Path(tmp)
            directory.mkdir(parents=True, exist_ok=True)
            for url, file_name in DEMO_PAPERS:
                path = directory / file_name
                if not path.exists():
                    print(f"Downloading {file_name} from {url} ...")
                    download(url, path)
                papers.append(build_paper_record(path, file_name, KNOWN_METADATA))

    write_snapshot(papers, args.out)
    snapshot = load_snapshot(args.out)
    for paper in snapshot.papers:
        print(f"  {paper.key}: {len(paper.pages)} page(s), {len(paper.chunks)} chunk(s)")
    print(f"Wrote {args.out} (fingerprint {snapshot.fingerprint})")


if __name__ == "__main__":
    main()
