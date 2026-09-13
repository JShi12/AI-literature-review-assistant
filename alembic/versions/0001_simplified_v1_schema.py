from __future__ import annotations

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

revision = "0001_simplified_v1_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "llm_runs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("model", sa.String(length=120), nullable=False),
        sa.Column("prompt_version", sa.String(length=80), nullable=False),
        sa.Column("temperature", sa.Numeric(4, 3), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cost", sa.Numeric(12, 6), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "papers",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("paper_key", sa.String(length=32), nullable=False, unique=True),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("authors", sa.JSON(), nullable=False, server_default=sa.text("'[]'::json")),
        sa.Column("year", sa.Integer(), nullable=True),
        sa.Column("file_name", sa.Text(), nullable=False),
        sa.Column("file_sha256", sa.String(length=64), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "pages",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("paper_id", sa.String(length=36), sa.ForeignKey("papers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("paper_id", "page_number", name="uq_pages_paper_page_number"),
    )

    op.create_table(
        "sections",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("paper_id", sa.String(length=36), sa.ForeignKey("papers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("normalized_type", sa.String(length=40), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=False),
        sa.Column("page_end", sa.Integer(), nullable=False),
        sa.Column("start_char", sa.Integer(), nullable=True),
        sa.Column("end_char", sa.Integer(), nullable=True),
        sa.Column("confidence", sa.Numeric(4, 3), nullable=True),
        sa.Column("run_id", sa.String(length=36), sa.ForeignKey("llm_runs.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("normalized_type in ('abstract','introduction','related_work','methods','results','discussion','limitations','conclusion','other')", name="ck_sections_normalized_type"),
        sa.CheckConstraint("confidence is null or (confidence >= 0 and confidence <= 1)", name="ck_sections_confidence_range"),
    )

    op.create_table(
        "chunks",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("paper_id", sa.String(length=36), sa.ForeignKey("papers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("section_id", sa.String(length=36), sa.ForeignKey("sections.id", ondelete="SET NULL"), nullable=True),
        sa.Column("page_start", sa.Integer(), nullable=False),
        sa.Column("page_end", sa.Integer(), nullable=False),
        sa.Column("start_char", sa.Integer(), nullable=False),
        sa.Column("end_char", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("start_char >= 0", name="ck_chunks_start_char_nonnegative"),
        sa.CheckConstraint("end_char > start_char", name="ck_chunks_valid_offsets"),
    )

    op.create_table(
        "claims",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("claim_text", sa.Text(), nullable=False),
        sa.Column("claim_type", sa.String(length=40), nullable=False),
        sa.Column("normalized_text", sa.Text(), nullable=True),
        sa.Column("paper_id", sa.String(length=36), sa.ForeignKey("papers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_id", sa.String(length=36), sa.ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("section_id", sa.String(length=36), sa.ForeignKey("sections.id", ondelete="SET NULL"), nullable=True),
        sa.Column("page", sa.Integer(), nullable=False),
        sa.Column("start_char", sa.Integer(), nullable=False),
        sa.Column("end_char", sa.Integer(), nullable=False),
        sa.Column("confidence", sa.Numeric(4, 3), nullable=False),
        sa.Column("run_id", sa.String(length=36), sa.ForeignKey("llm_runs.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("claim_type in ('finding','method','dataset','metric','limitation','future_work','background','other')", name="ck_claims_claim_type"),
        sa.CheckConstraint("confidence >= 0 and confidence <= 1", name="ck_claims_confidence_range"),
        sa.CheckConstraint("end_char > start_char", name="ck_claims_valid_offsets"),
    )

    op.create_table(
        "syntheses",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("synthesis_type", sa.String(length=40), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("supporting_papers", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("supporting_claims", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("confidence", sa.Numeric(4, 3), nullable=False),
        sa.Column("run_id", sa.String(length=36), sa.ForeignKey("llm_runs.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("synthesis_type in ('theme','contradiction','gap','method_comparison','insight')", name="ck_syntheses_synthesis_type"),
        sa.CheckConstraint("confidence >= 0 and confidence <= 1", name="ck_syntheses_confidence_range"),
        sa.CheckConstraint("supporting_papers >= 0", name="ck_syntheses_supporting_papers_nonnegative"),
        sa.CheckConstraint("supporting_claims >= 0", name="ck_syntheses_supporting_claims_nonnegative"),
    )

    op.create_table(
        "synthesis_claims",
        sa.Column("synthesis_id", sa.String(length=36), sa.ForeignKey("syntheses.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("claim_id", sa.String(length=36), sa.ForeignKey("claims.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "review_drafts",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("outline", sa.JSON(), nullable=False),
        sa.Column("markdown", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Numeric(4, 3), nullable=False),
        sa.Column("run_id", sa.String(length=36), sa.ForeignKey("llm_runs.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("confidence >= 0 and confidence <= 1", name="ck_review_drafts_confidence_range"),
    )

    op.create_table(
        "review_sentences",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("review_draft_id", sa.String(length=36), sa.ForeignKey("review_drafts.id", ondelete="CASCADE"), nullable=False),
        sa.Column("section_title", sa.Text(), nullable=False),
        sa.Column("sentence_index", sa.Integer(), nullable=False),
        sa.Column("sentence_text", sa.Text(), nullable=False),
        sa.Column("is_supported", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("review_draft_id", "section_title", "sentence_index", name="uq_review_sentences_draft_section_index"),
    )

    op.create_table(
        "review_sentence_claims",
        sa.Column("review_sentence_id", sa.String(length=36), sa.ForeignKey("review_sentences.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("claim_id", sa.String(length=36), sa.ForeignKey("claims.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "embeddings",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("entity_type", sa.String(length=40), nullable=False),
        sa.Column("entity_id", sa.String(length=36), nullable=False),
        sa.Column("embedding", Vector(1536), nullable=False),
        sa.Column("model", sa.String(length=120), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("entity_type in ('chunk','claim','synthesis')", name="ck_embeddings_entity_type"),
        sa.UniqueConstraint("entity_type", "entity_id", "model", name="uq_embeddings_entity_model"),
    )

    op.create_index("ix_pages_paper_id", "pages", ["paper_id"])
    op.create_index("ix_sections_paper_id", "sections", ["paper_id"])
    op.create_index("ix_chunks_paper_id", "chunks", ["paper_id"])
    op.create_index("ix_chunks_section_id", "chunks", ["section_id"])
    op.create_index("ix_claims_paper_id", "claims", ["paper_id"])
    op.create_index("ix_claims_chunk_id", "claims", ["chunk_id"])
    op.create_index("ix_syntheses_synthesis_type", "syntheses", ["synthesis_type"])
    op.create_index("ix_embeddings_entity", "embeddings", ["entity_type", "entity_id"])


def downgrade() -> None:
    op.drop_index("ix_embeddings_entity", table_name="embeddings")
    op.drop_index("ix_syntheses_synthesis_type", table_name="syntheses")
    op.drop_index("ix_claims_chunk_id", table_name="claims")
    op.drop_index("ix_claims_paper_id", table_name="claims")
    op.drop_index("ix_chunks_section_id", table_name="chunks")
    op.drop_index("ix_chunks_paper_id", table_name="chunks")
    op.drop_index("ix_sections_paper_id", table_name="sections")
    op.drop_index("ix_pages_paper_id", table_name="pages")
    op.drop_table("embeddings")
    op.drop_table("review_sentence_claims")
    op.drop_table("review_sentences")
    op.drop_table("review_drafts")
    op.drop_table("synthesis_claims")
    op.drop_table("syntheses")
    op.drop_table("claims")
    op.drop_table("chunks")
    op.drop_table("sections")
    op.drop_table("pages")
    op.drop_table("papers")
    op.drop_table("llm_runs")
