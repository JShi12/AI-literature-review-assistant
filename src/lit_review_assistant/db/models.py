from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def uuid_str() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class LLMRun(Base):
    __tablename__ = "llm_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    model: Mapped[str] = mapped_column(String(120), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(80), nullable=False)
    temperature: Mapped[Decimal] = mapped_column(Numeric(4, 3), nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cost: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    paper_key: Mapped[str] = mapped_column(String(32), nullable=False, unique=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    authors: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    file_name: Mapped[str] = mapped_column(Text, nullable=False)
    file_sha256: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    pages: Mapped[list[Page]] = relationship(back_populates="paper", cascade="all, delete-orphan")
    sections: Mapped[list[Section]] = relationship(back_populates="paper", cascade="all, delete-orphan")
    chunks: Mapped[list[Chunk]] = relationship(back_populates="paper", cascade="all, delete-orphan")
    claims: Mapped[list[Claim]] = relationship(back_populates="paper", cascade="all, delete-orphan")


class Page(Base):
    __tablename__ = "pages"
    __table_args__ = (UniqueConstraint("paper_id", "page_number", name="uq_pages_paper_page_number"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    paper: Mapped[Paper] = relationship(back_populates="pages")


class Section(Base):
    __tablename__ = "sections"
    __table_args__ = (
        CheckConstraint(
            "normalized_type in ('abstract','introduction','related_work','methods','results','discussion','limitations','conclusion','other')",
            name="ck_sections_normalized_type",
        ),
        CheckConstraint("confidence is null or (confidence >= 0 and confidence <= 1)", name="ck_sections_confidence_range"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_type: Mapped[str] = mapped_column(String(40), nullable=False)
    page_start: Mapped[int] = mapped_column(Integer, nullable=False)
    page_end: Mapped[int] = mapped_column(Integer, nullable=False)
    start_char: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_char: Mapped[int | None] = mapped_column(Integer, nullable=True)
    confidence: Mapped[Decimal | None] = mapped_column(Numeric(4, 3), nullable=True)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("llm_runs.id", ondelete="SET NULL"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    paper: Mapped[Paper] = relationship(back_populates="sections")
    chunks: Mapped[list[Chunk]] = relationship(back_populates="section")


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        CheckConstraint("start_char >= 0", name="ck_chunks_start_char_nonnegative"),
        CheckConstraint("end_char > start_char", name="ck_chunks_valid_offsets"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    section_id: Mapped[str | None] = mapped_column(ForeignKey("sections.id", ondelete="SET NULL"), nullable=True)
    page_start: Mapped[int] = mapped_column(Integer, nullable=False)
    page_end: Mapped[int] = mapped_column(Integer, nullable=False)
    start_char: Mapped[int] = mapped_column(Integer, nullable=False)
    end_char: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    paper: Mapped[Paper] = relationship(back_populates="chunks")
    section: Mapped[Section | None] = relationship(back_populates="chunks")
    claims: Mapped[list[Claim]] = relationship(back_populates="chunk", cascade="all, delete-orphan")


class Claim(Base):
    __tablename__ = "claims"
    __table_args__ = (
        CheckConstraint(
            "claim_type in ('finding','method','dataset','metric','limitation','future_work','background','other')",
            name="ck_claims_claim_type",
        ),
        CheckConstraint("confidence >= 0 and confidence <= 1", name="ck_claims_confidence_range"),
        CheckConstraint("end_char > start_char", name="ck_claims_valid_offsets"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    claim_text: Mapped[str] = mapped_column(Text, nullable=False)
    claim_type: Mapped[str] = mapped_column(String(40), nullable=False)
    normalized_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    chunk_id: Mapped[str] = mapped_column(ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False)
    section_id: Mapped[str | None] = mapped_column(ForeignKey("sections.id", ondelete="SET NULL"), nullable=True)
    page: Mapped[int] = mapped_column(Integer, nullable=False)
    start_char: Mapped[int] = mapped_column(Integer, nullable=False)
    end_char: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[Decimal] = mapped_column(Numeric(4, 3), nullable=False)
    run_id: Mapped[str] = mapped_column(ForeignKey("llm_runs.id", ondelete="RESTRICT"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    paper: Mapped[Paper] = relationship(back_populates="claims")
    chunk: Mapped[Chunk] = relationship(back_populates="claims")
    syntheses: Mapped[list[Synthesis]] = relationship(
        secondary="synthesis_claims",
        back_populates="claims",
    )
    review_sentences: Mapped[list[ReviewSentence]] = relationship(
        secondary="review_sentence_claims",
        back_populates="claims",
    )


class Synthesis(Base):
    __tablename__ = "syntheses"
    __table_args__ = (
        CheckConstraint(
            "synthesis_type in ('theme','contradiction','gap','method_comparison','insight')",
            name="ck_syntheses_synthesis_type",
        ),
        CheckConstraint("confidence >= 0 and confidence <= 1", name="ck_syntheses_confidence_range"),
        CheckConstraint("supporting_papers >= 0", name="ck_syntheses_supporting_papers_nonnegative"),
        CheckConstraint("supporting_claims >= 0", name="ck_syntheses_supporting_claims_nonnegative"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    synthesis_type: Mapped[str] = mapped_column(String(40), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    supporting_papers: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    supporting_claims: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    confidence: Mapped[Decimal] = mapped_column(Numeric(4, 3), nullable=False)
    run_id: Mapped[str] = mapped_column(ForeignKey("llm_runs.id", ondelete="RESTRICT"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    claims: Mapped[list[Claim]] = relationship(
        secondary="synthesis_claims",
        back_populates="syntheses",
    )


class SynthesisClaim(Base):
    __tablename__ = "synthesis_claims"

    synthesis_id: Mapped[str] = mapped_column(ForeignKey("syntheses.id", ondelete="CASCADE"), primary_key=True)
    claim_id: Mapped[str] = mapped_column(ForeignKey("claims.id", ondelete="CASCADE"), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class ReviewDraft(Base):
    __tablename__ = "review_drafts"
    __table_args__ = (CheckConstraint("confidence >= 0 and confidence <= 1", name="ck_review_drafts_confidence_range"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    outline: Mapped[object] = mapped_column(JSON, nullable=False)
    markdown: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[Decimal] = mapped_column(Numeric(4, 3), nullable=False)
    run_id: Mapped[str] = mapped_column(ForeignKey("llm_runs.id", ondelete="RESTRICT"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    sentences: Mapped[list[ReviewSentence]] = relationship(back_populates="review_draft", cascade="all, delete-orphan")


class ReviewSentence(Base):
    __tablename__ = "review_sentences"
    __table_args__ = (
        UniqueConstraint("review_draft_id", "section_title", "sentence_index", name="uq_review_sentences_draft_section_index"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    review_draft_id: Mapped[str] = mapped_column(ForeignKey("review_drafts.id", ondelete="CASCADE"), nullable=False)
    section_title: Mapped[str] = mapped_column(Text, nullable=False)
    sentence_index: Mapped[int] = mapped_column(Integer, nullable=False)
    sentence_text: Mapped[str] = mapped_column(Text, nullable=False)
    is_supported: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    review_draft: Mapped[ReviewDraft] = relationship(back_populates="sentences")
    claims: Mapped[list[Claim]] = relationship(
        secondary="review_sentence_claims",
        back_populates="review_sentences",
    )


class ReviewSentenceClaim(Base):
    __tablename__ = "review_sentence_claims"

    review_sentence_id: Mapped[str] = mapped_column(ForeignKey("review_sentences.id", ondelete="CASCADE"), primary_key=True)
    claim_id: Mapped[str] = mapped_column(ForeignKey("claims.id", ondelete="CASCADE"), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Embedding(Base):
    __tablename__ = "embeddings"
    __table_args__ = (
        CheckConstraint("entity_type in ('chunk','claim','synthesis')", name="ck_embeddings_entity_type"),
        UniqueConstraint("entity_type", "entity_id", "model", name="uq_embeddings_entity_model"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    entity_type: Mapped[str] = mapped_column(String(40), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(36), nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(1536), nullable=False)
    model: Mapped[str] = mapped_column(String(120), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
