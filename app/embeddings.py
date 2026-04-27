from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db.models import Chunk
from .providers import get_embedding_provider


def persist_missing_chunk_embeddings(
    session: Session,
    *,
    product_id: UUID,
    batch_size: int = 32,
) -> dict[str, int]:
    stmt = (
        select(Chunk)
        .where(Chunk.product_id == product_id, Chunk.embedding.is_(None))
        .order_by(Chunk.created_at.asc())
    )
    chunks = session.scalars(stmt).all()
    if not chunks:
        return {"embedded_chunks": 0}

    provider = get_embedding_provider()
    embedded = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = provider.embed_texts([chunk.text for chunk in batch])
        if len(vectors) != len(batch):
            raise RuntimeError("Embedding provider returned a mismatched number of vectors.")
        for chunk, vector in zip(batch, vectors, strict=True):
            chunk.embedding = vector
            embedded += 1
    session.flush()
    return {"embedded_chunks": embedded}
