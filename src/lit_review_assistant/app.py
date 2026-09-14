"""Streamlit UI for uploading papers and running the ingestion, claims, synthesis, and review pipeline."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Sequence
from pathlib import Path

import streamlit as st
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from lit_review_assistant import services
from lit_review_assistant.db.models import Chunk, Claim, Paper, ReviewDraft, Synthesis
from lit_review_assistant.db.session import session_scope
from lit_review_assistant.llm import client as llm_client
from lit_review_assistant.llm import review as review_llm
from lit_review_assistant.llm.claims import extract_claims_for_chunk
from lit_review_assistant.llm.embeddings import find_similar_claims, find_similar_syntheses
from lit_review_assistant.llm.synthesis import generate_syntheses
from lit_review_assistant.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("data/uploads")
SYNTHESIS_TYPES = ["theme", "contradiction", "gap", "method_comparison", "insight"]
CHUNK_SIZE_OPTIONS = [1_500, 3_000, 4_000]
CHUNK_OVERLAP_OPTIONS = [150, 250, 300]


def main() -> None:
    st.set_page_config(page_title="AI Literature Review Assistant", layout="wide")
    st.title("AI Literature Review Assistant")

    tabs = st.tabs(
        [
            "Upload Papers",
            "Paper Structure",
            "Claims",
            "Syntheses",
            "Review Drafts",
            "Database",
        ]
    )

    with tabs[0]:
        upload_papers_tab()
    with tabs[1]:
        paper_structure_tab()
    with tabs[2]:
        claims_tab()
    with tabs[3]:
        syntheses_tab()
    with tabs[4]:
        review_drafts_tab()
    with tabs[5]:
        database_tab()


def upload_papers_tab() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    chunk_max_chars = st.selectbox(
        "Chunk size", CHUNK_SIZE_OPTIONS, index=0, format_func=lambda value: f"{value} characters"
    )
    chunk_overlap = st.selectbox(
        "Chunk overlap",
        [option for option in CHUNK_OVERLAP_OPTIONS if option < chunk_max_chars],
        index=0,
        format_func=lambda value: f"{value} characters",
    )
    st.caption("Larger chunks usually reduce API calls, but can make claim extraction less precise.")
    uploads = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if not uploads:
        return

    if st.button("Ingest PDFs", type="primary"):
        try:
            ingested: list[str] = []
            for upload in uploads:
                destination = UPLOAD_DIR / upload.name
                destination.write_bytes(upload.getbuffer())
                with session_scope() as session:
                    paper = services.ingest_pdf(
                        session,
                        destination,
                        file_name=upload.name,
                        chunk_max_chars=chunk_max_chars,
                        chunk_overlap=chunk_overlap,
                    )
                    ingested.append(f"{paper.paper_key}: {paper.file_name}")
            st.success(f"Ingested {len(ingested)} PDF(s).")
            st.write(ingested)
        except Exception as exc:
            logger.exception("PDF ingestion failed")
            st.error(f"PDF ingestion failed: {describe_error(exc)}")


def paper_structure_tab() -> None:
    with session_scope() as session:
        papers = session.scalars(select(Paper).order_by(Paper.created_at.desc()).limit(20)).all()
        rows = []
        for paper in papers:
            rows.append(
                {
                    "paper_key": paper.paper_key,
                    "title": paper.title,
                    "file_name": paper.file_name,
                    "pages": len(paper.pages),
                    "sections": len(paper.sections),
                    "chunks": len(paper.chunks),
                }
            )
    st.dataframe(rows, use_container_width=True)


def claims_tab() -> None:
    with session_scope() as session:
        claim_count = session.scalar(select(func.count(Claim.id))) or 0
        chunk_count = session.scalar(select(func.count(Chunk.id))) or 0
        papers = session.scalars(select(Paper).order_by(Paper.created_at.desc())).all()

    st.metric("Claims", claim_count)
    st.metric("Available chunks", chunk_count)

    if chunk_count == 0:
        st.info("Upload and ingest at least one PDF before extracting claims.")
        return

    paper_options: dict[str, str | None] = {"All papers": None}
    paper_options.update({f"{paper.paper_key}: {paper.file_name}": paper.id for paper in papers})
    selected_label = st.selectbox("Paper", list(paper_options.keys()))
    selected_paper_id = paper_options[selected_label]
    selected_chunk_count = count_chunks_for_selection(selected_paper_id)
    st.caption(f"Selected scope contains {selected_chunk_count} chunk(s).")
    max_chunks = st.number_input(
        "Max chunks to process",
        min_value=1,
        max_value=max(selected_chunk_count, 1),
        value=min(3, max(selected_chunk_count, 1)),
        step=1,
    )
    st.caption(
        "Chunks without claims yet are processed round-robin across papers (so a small limit still "
        "covers every paper), then in page and character-offset order within each paper."
    )

    if st.button("Extract Claims", type="primary"):
        if not ensure_openai_key():
            return
        with st.spinner("Extracting claims from chunks..."):
            try:
                with session_scope() as session:
                    chunks = services.select_chunks_for_claim_extraction(session, selected_paper_id, max_chunks)
                    created = 0
                    for chunk in chunks:
                        created += len(extract_claims_for_chunk(session, chunk))
                if not chunks:
                    st.info("No unprocessed chunks in this scope -- every chunk already has claims.")
                else:
                    st.success(f"Extracted {created} claim(s) from {len(chunks)} chunk(s).")
            except Exception as exc:
                logger.exception("Claim extraction failed")
                st.error(f"Claim extraction failed: {describe_error(exc)}")


def count_chunks_for_selection(paper_id: str | None) -> int:
    with session_scope() as session:
        query = select(func.count(Chunk.id))
        if paper_id is not None:
            query = query.where(Chunk.paper_id == paper_id)
        return session.scalar(query) or 0


def syntheses_tab() -> None:
    with session_scope() as session:
        synthesis_count = session.scalar(select(func.count(Synthesis.id))) or 0
        claim_count = session.scalar(select(func.count(Claim.id))) or 0

    st.metric("Syntheses", synthesis_count)
    st.metric("Available claims", claim_count)

    if claim_count == 0:
        st.info("Extract claims before generating syntheses.")
        return

    synthesis_types = st.multiselect("Synthesis types", SYNTHESIS_TYPES, default=SYNTHESIS_TYPES)
    topic = st.text_input(
        "Topic (optional)",
        value="",
        help="When set, the claims most relevant to this topic are used instead of the most recent ones.",
    )
    max_claims = st.number_input(
        "Claims to use",
        min_value=1,
        max_value=claim_count,
        value=min(10, claim_count),
        step=1,
    )
    st.caption(f"Up to {claim_count} available claim(s) can be used.")

    if st.button("Generate Syntheses", type="primary"):
        if not ensure_openai_key():
            return
        if not synthesis_types:
            st.warning("Choose at least one synthesis type.")
            return
        with st.spinner("Generating syntheses from claims..."):
            try:
                with session_scope() as session:
                    claims: Sequence[Claim] = []
                    if topic.strip():
                        claims = find_similar_claims(session, topic, limit=max_claims)
                    if not claims:
                        claims = session.scalars(select(Claim).order_by(Claim.created_at.asc()).limit(max_claims)).all()
                    total_created = 0
                    for synthesis_type in synthesis_types:
                        syntheses = generate_syntheses(
                            session,
                            claim_ids=[claim.id for claim in claims],
                            synthesis_type=synthesis_type,  # type: ignore[arg-type]
                        )
                        total_created += len(syntheses)
                st.success(f"Generated {total_created} synthesis item(s) across {len(synthesis_types)} type(s).")
            except Exception as exc:
                logger.exception("Synthesis generation failed")
                st.error(f"Synthesis generation failed: {describe_error(exc)}")


def review_drafts_tab() -> None:
    with session_scope() as session:
        draft_count = session.scalar(select(func.count(ReviewDraft.id))) or 0
        synthesis_count = session.scalar(select(func.count(Synthesis.id))) or 0
        drafts = session.scalars(select(ReviewDraft).order_by(ReviewDraft.created_at.desc())).all()

    st.metric("Review Drafts", draft_count)
    st.metric("Available syntheses", synthesis_count)

    if synthesis_count == 0:
        st.info("Generate syntheses before drafting a review.")
        return

    topic = st.text_input("Review topic", value="AI literature review")
    max_syntheses = st.number_input(
        "Max syntheses to use",
        min_value=1,
        max_value=synthesis_count,
        value=min(20, synthesis_count),
        step=1,
    )
    st.caption(
        f"Up to {synthesis_count} available synthesis item(s) can be used, "
        "selected by relevance to the topic above when embeddings are available."
    )

    if st.button("Generate Review Draft", type="primary"):
        if not ensure_openai_key():
            return
        with st.spinner("Generating review draft..."):
            try:
                with session_scope() as session:
                    syntheses: Sequence[Synthesis] = []
                    if topic.strip():
                        syntheses = find_similar_syntheses(session, topic, limit=max_syntheses)
                    if not syntheses:
                        syntheses = session.scalars(
                            select(Synthesis).order_by(Synthesis.created_at.desc()).limit(max_syntheses)
                        ).all()
                    draft = review_llm.generate_review_draft(
                        session,
                        topic=topic,
                        synthesis_ids=[synthesis.id for synthesis in syntheses],
                    )
                if draft is None:
                    st.warning("No review draft was created.")
                else:
                    st.success(f"Generated review draft: {draft.title}")
                    st.rerun()
            except Exception as exc:
                logger.exception("Review draft generation failed")
                st.error(f"Review draft generation failed: {describe_error(exc)}")

    selected_draft = select_review_draft(drafts)
    if selected_draft is not None:
        render_review_markdown(selected_draft.markdown)


def select_review_draft(drafts: Sequence[ReviewDraft]) -> ReviewDraft | None:
    if not drafts:
        st.info("No review drafts have been generated yet.")
        return None

    draft_by_id = {draft.id: draft for draft in drafts}
    selected_id = st.selectbox(
        "Review draft",
        [draft.id for draft in drafts],
        format_func=lambda draft_id: format_review_draft_label(draft_by_id[draft_id]),
    )
    return draft_by_id[selected_id]


def format_review_draft_label(draft: ReviewDraft) -> str:
    created = draft.created_at.strftime("%Y-%m-%d %H:%M") if draft.created_at is not None else "unknown time"
    return f"{created} - {draft.title}"


def ensure_openai_key() -> bool:
    if os.getenv("OPENAI_API_KEY"):
        return True
    st.warning("Set OPENAI_API_KEY in the terminal before running Streamlit, then restart the app.")
    return False


def describe_error(exc: Exception) -> str:
    return llm_client.describe_openai_error(exc)


def format_review_markdown(markdown: str) -> str:
    return review_llm.normalize_references_for_markdown(markdown)


def render_review_markdown(markdown: str) -> None:
    formatted = format_review_markdown(markdown)
    parts = re.split(r"(?im)^\s{0,3}#{0,6}\s*references\s*$", formatted, maxsplit=1)
    if len(parts) != 2:
        st.markdown(formatted)
        return

    body, references = parts
    if body.strip():
        st.markdown(body.strip())
    st.markdown("## References")
    for reference in references.splitlines():
        reference = reference.strip()
        if reference:
            st.markdown(reference)


def database_tab() -> None:
    with session_scope() as session:
        counts = {
            "papers": session.scalar(select(func.count(Paper.id))) or 0,
            "chunks": session.scalar(select(func.count(Chunk.id))) or 0,
            "claims": session.scalar(select(func.count(Claim.id))) or 0,
            "syntheses": session.scalar(select(func.count(Synthesis.id))) or 0,
            "review_drafts": session.scalar(select(func.count(ReviewDraft.id))) or 0,
        }
        papers = session.scalars(select(Paper).order_by(Paper.paper_key.asc())).all()
        paper_rows = [
            {
                "paper_key": paper.paper_key,
                "title": paper.title,
                "authors": ", ".join(paper.authors or []),
                "year": paper.year,
                "file_name": paper.file_name,
            }
            for paper in papers
        ]
    st.json(counts)

    if st.button("Backfill Paper Metadata"):
        try:
            with session_scope() as session:
                updated = backfill_metadata(session, UPLOAD_DIR)
            st.success(f"Updated metadata for {updated} paper(s). Refresh the page to see updated rows.")
        except Exception as exc:
            logger.exception("Paper metadata backfill failed")
            st.error(f"Paper metadata backfill failed: {describe_error(exc)}")

    if paper_rows:
        st.dataframe(paper_rows, use_container_width=True)


def backfill_metadata(session: Session, upload_dir: Path) -> int:
    """Backfill missing paper title/author/year metadata for already-ingested papers."""
    return services.backfill_paper_metadata(session, upload_dir)


if __name__ == "__main__":
    main()
