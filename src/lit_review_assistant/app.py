from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from sqlalchemy import func, select

from lit_review_assistant.db.models import Chunk, Claim, Paper, ReviewDraft, Synthesis
from lit_review_assistant.db.session import session_scope
from lit_review_assistant.llm.claims import extract_claims_for_chunk
from lit_review_assistant.llm.review import generate_review_draft
from lit_review_assistant.llm.synthesis import generate_syntheses
from lit_review_assistant.services import ingest_pdf


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
    chunk_max_chars = st.selectbox("Chunk size", CHUNK_SIZE_OPTIONS, index=0, format_func=lambda value: f"{value} characters")
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
        ingested: list[str] = []
        for upload in uploads:
            destination = UPLOAD_DIR / upload.name
            destination.write_bytes(upload.getbuffer())
            with session_scope() as session:
                paper = ingest_pdf(
                    session,
                    destination,
                    file_name=upload.name,
                    chunk_max_chars=chunk_max_chars,
                    chunk_overlap=chunk_overlap,
                )
                ingested.append(f"{paper.paper_key}: {paper.file_name}")
        st.success(f"Ingested {len(ingested)} PDF(s).")
        st.write(ingested)


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

    paper_options = {"All papers": None}
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
    st.caption("Chunks are processed in paper order, then page order, then character-offset order.")

    if st.button("Extract Claims", type="primary"):
        if not ensure_openai_key():
            return
        with st.spinner("Extracting claims from chunks..."):
            try:
                with session_scope() as session:
                    query = (
                        select(Chunk)
                        .join(Paper, Paper.id == Chunk.paper_id)
                        .order_by(Paper.paper_key.asc(), Chunk.page_start.asc(), Chunk.start_char.asc())
                        .limit(max_chunks)
                    )
                    if selected_paper_id is not None:
                        query = (
                            select(Chunk)
                            .join(Paper, Paper.id == Chunk.paper_id)
                            .where(Chunk.paper_id == selected_paper_id)
                            .order_by(Paper.paper_key.asc(), Chunk.page_start.asc(), Chunk.start_char.asc())
                            .limit(max_chunks)
                        )
                    chunks = session.scalars(query).all()
                    created = 0
                    for chunk in chunks:
                        created += len(extract_claims_for_chunk(session, chunk))
                st.success(f"Extracted {created} claim(s) from {len(chunks)} chunk(s).")
            except Exception as exc:
                st.error(f"Claim extraction failed: {exc}")


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
    max_claims = st.number_input("Max recent claims to use", min_value=1, max_value=50, value=10, step=1)

    if st.button("Generate Syntheses", type="primary"):
        if not ensure_openai_key():
            return
        if not synthesis_types:
            st.warning("Choose at least one synthesis type.")
            return
        with st.spinner("Generating syntheses from claims..."):
            try:
                with session_scope() as session:
                    claims = session.scalars(
                        select(Claim).order_by(Claim.created_at.desc()).limit(max_claims)
                    ).all()
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
                st.error(f"Synthesis generation failed: {exc}")


def review_drafts_tab() -> None:
    with session_scope() as session:
        draft_count = session.scalar(select(func.count(ReviewDraft.id))) or 0
        synthesis_count = session.scalar(select(func.count(Synthesis.id))) or 0
        latest_draft = session.scalars(select(ReviewDraft).order_by(ReviewDraft.created_at.desc()).limit(1)).first()
        latest_title = latest_draft.title if latest_draft is not None else None
        latest_markdown = latest_draft.markdown if latest_draft is not None else None

    st.metric("Review Drafts", draft_count)
    st.metric("Available syntheses", synthesis_count)

    if synthesis_count == 0:
        st.info("Generate syntheses before drafting a review.")
        return

    topic = st.text_input("Review topic", value="AI literature review")
    max_syntheses = st.number_input("Max recent syntheses to use", min_value=1, max_value=50, value=20, step=1)

    if st.button("Generate Review Draft", type="primary"):
        if not ensure_openai_key():
            return
        with st.spinner("Generating review draft..."):
            try:
                with session_scope() as session:
                    syntheses = session.scalars(
                        select(Synthesis).order_by(Synthesis.created_at.desc()).limit(max_syntheses)
                    ).all()
                    draft = generate_review_draft(
                        session,
                        topic=topic,
                        synthesis_ids=[synthesis.id for synthesis in syntheses],
                    )
                if draft is None:
                    st.warning("No review draft was created.")
                else:
                    latest_title = draft.title
                    latest_markdown = draft.markdown
                    st.success(f"Generated review draft: {draft.title}")
            except Exception as exc:
                st.error(f"Review draft generation failed: {exc}")

    if latest_markdown:
        st.subheader(latest_title or "Latest Review Draft")
        st.markdown(latest_markdown)


def ensure_openai_key() -> bool:
    if os.getenv("OPENAI_API_KEY"):
        return True
    st.warning("Set OPENAI_API_KEY in the terminal before running Streamlit, then restart the app.")
    return False


def table_count_tab(label: str, model: type) -> None:
    with session_scope() as session:
        count = session.scalar(select(func.count(model.id))) or 0
    st.metric(label, count)


def database_tab() -> None:
    with session_scope() as session:
        counts = {
            "papers": session.scalar(select(func.count(Paper.id))) or 0,
            "chunks": session.scalar(select(func.count(Chunk.id))) or 0,
            "claims": session.scalar(select(func.count(Claim.id))) or 0,
            "syntheses": session.scalar(select(func.count(Synthesis.id))) or 0,
            "review_drafts": session.scalar(select(func.count(ReviewDraft.id))) or 0,
        }
    st.json(counts)


if __name__ == "__main__":
    main()
