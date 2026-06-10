from __future__ import annotations

from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from lit_review_assistant.db.models import Base, Claim, Embedding, ReviewSentence, ReviewSentenceClaim, Synthesis
from lit_review_assistant.services import ingest_pdf, recalculate_synthesis_support_counts


def test_simplified_v1_tables_are_present() -> None:
    assert set(Base.metadata.tables) == {
        "papers",
        "pages",
        "sections",
        "chunks",
        "claims",
        "syntheses",
        "synthesis_claims",
        "review_drafts",
        "review_sentences",
        "review_sentence_claims",
        "embeddings",
        "llm_runs",
    }


def test_review_sentences_use_join_table_for_many_claims() -> None:
    review_columns = set(ReviewSentence.__table__.columns)
    assert "claim_id" not in review_columns
    assert "synthesis_id" not in review_columns

    assert {"review_sentence_id", "claim_id"} == set(ReviewSentenceClaim.__table__.primary_key.columns.keys())


def test_llm_generated_core_objects_require_confidence_and_run_id() -> None:
    for table in [Claim.__table__, Synthesis.__table__]:
        assert not table.c.confidence.nullable
        assert not table.c.run_id.nullable


def test_embeddings_compile_to_pgvector_column() -> None:
    ddl = str(CreateTable(Embedding.__table__).compile(dialect=postgresql.dialect()))
    assert "embedding VECTOR(1536)" in ddl
    assert "entity_type" in ddl


def test_service_entrypoints_are_importable() -> None:
    assert callable(ingest_pdf)
    assert callable(recalculate_synthesis_support_counts)
