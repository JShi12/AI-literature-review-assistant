"""Generate and query embeddings so claims and syntheses can be selected by topic relevance."""

from __future__ import annotations

import logging
import os

from openai import OpenAI
from sqlalchemy import select
from sqlalchemy.orm import Session

from lit_review_assistant.db.models import Claim, Embedding, Synthesis

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def _embedding_model() -> str:
    return os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def embed_texts(texts: list[str], client: OpenAI | None = None) -> list[list[float]]:
    """Embed a batch of texts with the configured OpenAI embedding model, preserving input order."""
    if not texts:
        return []
    client = client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(model=_embedding_model(), input=texts)
    ordered = sorted(response.data, key=lambda item: item.index)
    return [item.embedding for item in ordered]


def embed_and_persist_claims(session: Session, claims: list[Claim], client: OpenAI | None = None) -> None:
    """Embed and store vectors for newly created claims. Logs and continues on failure."""
    if not claims:
        return
    try:
        texts = [claim.normalized_text or claim.claim_text for claim in claims]
        vectors = embed_texts(texts, client=client)
        model = _embedding_model()
        for claim, vector in zip(claims, vectors, strict=True):
            session.add(Embedding(entity_type="claim", entity_id=claim.id, embedding=vector, model=model))
        session.flush()
    except Exception:
        logger.exception("Failed to embed claims; they will not be selectable by topic similarity")


def embed_and_persist_syntheses(session: Session, syntheses: list[Synthesis], client: OpenAI | None = None) -> None:
    """Embed and store vectors for newly created syntheses. Logs and continues on failure."""
    if not syntheses:
        return
    try:
        texts = [f"{synthesis.title}\n{synthesis.body}" for synthesis in syntheses]
        vectors = embed_texts(texts, client=client)
        model = _embedding_model()
        for synthesis, vector in zip(syntheses, vectors, strict=True):
            session.add(Embedding(entity_type="synthesis", entity_id=synthesis.id, embedding=vector, model=model))
        session.flush()
    except Exception:
        logger.exception("Failed to embed syntheses; they will not be selectable by topic similarity")


def find_similar_claims(session: Session, topic: str, limit: int, client: OpenAI | None = None) -> list[Claim]:
    """Return claims whose embeddings are closest to `topic`, or [] if none are available."""
    topic_vector = _embed_topic(topic, client=client)
    if topic_vector is None:
        return []
    return list(
        session.scalars(
            select(Claim)
            .join(Embedding, (Embedding.entity_type == "claim") & (Embedding.entity_id == Claim.id))
            .order_by(Embedding.embedding.cosine_distance(topic_vector))
            .limit(limit)
        ).all()
    )


def find_similar_syntheses(session: Session, topic: str, limit: int, client: OpenAI | None = None) -> list[Synthesis]:
    """Return syntheses whose embeddings are closest to `topic`, or [] if none are available."""
    topic_vector = _embed_topic(topic, client=client)
    if topic_vector is None:
        return []
    return list(
        session.scalars(
            select(Synthesis)
            .join(Embedding, (Embedding.entity_type == "synthesis") & (Embedding.entity_id == Synthesis.id))
            .order_by(Embedding.embedding.cosine_distance(topic_vector))
            .limit(limit)
        ).all()
    )


def _embed_topic(topic: str, client: OpenAI | None = None) -> list[float] | None:
    if not topic.strip():
        return None
    try:
        vectors = embed_texts([topic], client=client)
    except Exception:
        logger.exception("Failed to embed topic %r for similarity search", topic)
        return None
    return vectors[0] if vectors else None
