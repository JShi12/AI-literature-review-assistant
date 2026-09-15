"""One-time script to seed a database with example autonomous-driving papers and a generated
review draft, for a read-only public demo (see READ_ONLY_DEMO in .env.example / README).

Downloads three real, open-access papers from arXiv, ingests them, extracts claims from every
chunk, generates syntheses across every synthesis type, then generates one review draft.

Usage (run against whichever database the deployed app actually uses -- e.g. point DATABASE_URL
at your Neon/Supabase connection string, not local Postgres, if you want the *deployed* demo to
show this data):

    DATABASE_URL=<target database> OPENAI_API_KEY=<key> python scripts/seed_demo_data.py

Requires the project installed (`pip install -e .`), same as running the app itself. Not designed
to be re-run repeatedly against the same database -- paper ingestion is deduplicated by file hash,
but claim/synthesis/review-draft generation is not, so running it twice would create duplicates.
"""

from __future__ import annotations

import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from lit_review_assistant import services
from lit_review_assistant.db.models import Chunk
from lit_review_assistant.db.session import session_scope
from lit_review_assistant.llm.claims import extract_claims_for_chunk
from lit_review_assistant.llm.review import generate_review_draft
from lit_review_assistant.llm.synthesis import generate_syntheses

# arXiv (and many hosts) reject urllib's default "Python-urllib/x.y" User-Agent as a bot.
DOWNLOAD_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; lit-review-assistant-demo-seed/1.0)"}
DEMO_PAPERS = [
    ("https://arxiv.org/pdf/1604.07316.pdf", "nvidia_end_to_end_self_driving.pdf"),
    ("https://arxiv.org/pdf/1812.03079.pdf", "chauffeurnet.pdf"),
    ("https://arxiv.org/pdf/1711.03938.pdf", "carla_simulator.pdf"),
]
SYNTHESIS_TYPES = ["theme", "contradiction", "gap", "method_comparison", "insight"]
REVIEW_TOPIC = "Autonomous driving: end-to-end learning, imitation learning, and simulation"


def download(url: str, path: Path, attempts: int = 3) -> None:
    request = urllib.request.Request(url, headers=DOWNLOAD_HEADERS)
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                path.write_bytes(response.read())
            return
        except urllib.error.HTTPError as exc:
            if attempt == attempts:
                raise
            print(f"    Download attempt {attempt} failed ({exc}); retrying...")
            time.sleep(2 * attempt)


def ingest_papers() -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        paper_ids = []
        with session_scope() as session:
            for url, filename in DEMO_PAPERS:
                path = tmp_dir / filename
                print(f"Downloading {filename} from {url} ...")
                download(url, path)
                paper = services.ingest_pdf(session, path, file_name=filename)
                paper_ids.append(paper.id)
                print(f"  Ingested as {paper.paper_key}: {paper.title!r} ({len(paper.chunks)} chunk(s))")
        return paper_ids


def extract_all_claims(paper_ids: list[str]) -> list[str]:
    """Extract claims chunk by chunk, each in its own transaction.

    An occasional chunk fails validation (the LLM's returned claim offsets don't quite line up
    with the source chunk's offsets) -- that's a pre-existing, correct safety check elsewhere in
    the pipeline, not something to weaken here. Isolating each chunk in its own session_scope
    means one bad chunk only loses that chunk's claims, not every claim already committed for
    every other chunk in the same run.
    """
    all_claim_ids: list[str] = []
    for paper_id in paper_ids:
        with session_scope() as session:
            chunk_ids = [c.id for c in services.select_chunks_for_claim_extraction(session, paper_id, limit=1_000)]

        skipped = 0
        for chunk_id in chunk_ids:
            try:
                with session_scope() as session:
                    chunk = session.get(Chunk, chunk_id)
                    claims = extract_claims_for_chunk(session, chunk)
                    all_claim_ids.extend(claim.id for claim in claims)
            except Exception as exc:
                skipped += 1
                print(f"    Skipped a chunk due to an extraction error: {exc}")
        print(f"  Paper {paper_id}: processed {len(chunk_ids) - skipped}/{len(chunk_ids)} chunk(s).")
    return all_claim_ids


def generate_all_syntheses(claim_ids: list[str]) -> list[str]:
    synthesis_ids: list[str] = []
    for synthesis_type in SYNTHESIS_TYPES:
        try:
            with session_scope() as session:
                syntheses = generate_syntheses(
                    session,
                    claim_ids=claim_ids,
                    synthesis_type=synthesis_type,  # type: ignore[arg-type]
                )
                synthesis_ids.extend(synthesis.id for synthesis in syntheses)
            print(f"  Generated {len(syntheses)} {synthesis_type} synthesis/syntheses.")
        except Exception as exc:
            print(f"  Skipped {synthesis_type} syntheses due to an error: {exc}")
    return synthesis_ids


def main() -> None:
    print("Ingesting papers...")
    paper_ids = ingest_papers()

    print("\nExtracting claims...")
    all_claim_ids = extract_all_claims(paper_ids)
    print(f"Extracted {len(all_claim_ids)} claim(s) total.")

    print("\nGenerating syntheses...")
    synthesis_ids = generate_all_syntheses(all_claim_ids)
    print(f"Generated {len(synthesis_ids)} synthesis/syntheses total.")

    print("\nGenerating review draft...")
    with session_scope() as session:
        draft = generate_review_draft(session, topic=REVIEW_TOPIC, synthesis_ids=synthesis_ids)
        if draft is None:
            print("No review draft was created.")
        else:
            print(f"Generated review draft: {draft.title!r}")


if __name__ == "__main__":
    main()
