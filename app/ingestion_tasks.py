from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from .db.repositories import IngestionJobRepository, ProductRepository
from .db.session import session_scope
from .settings import get_settings


def _resolved_raw_dir(raw_dir: Path | None = None) -> Path:
    if raw_dir is not None:
        return raw_dir
    settings = get_settings()
    root = settings.data_dir if settings.data_dir.is_absolute() else Path.cwd() / settings.data_dir
    return root / "raw"


def store_normalized_product(
    session: Session,
    *,
    job_id: UUID,
    product_blob: dict[str, Any],
    reviews: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    source_url: str | None = None,
) -> dict[str, Any]:
    job_repo = IngestionJobRepository(session)
    product_repo = ProductRepository(session)
    job = job_repo.get(job_id)
    if job is None:
        raise ValueError(f"Ingestion job not found: {job_id}")

    product = product_repo.replace_product_corpus(
        product_blob=product_blob,
        reviews=reviews,
        chunks=chunks,
        source_type=job.source_type,
        source_url=source_url,
    )
    session.flush()

    stats = {
        "sections": len(product_blob.get("sections") or []),
        "specs": len(product_blob.get("specs") or []),
        "reviews": len(reviews),
        "chunks": len(chunks),
    }
    if get_settings().enable_embedding_writes:
        from .embeddings import persist_missing_chunk_embeddings

        stats.update(persist_missing_chunk_embeddings(session, product_id=product.id))
    job_repo.mark_succeeded(job, product_id=product.id, stats=stats)
    return {"product_external_id": product.external_id, **stats}


def process_ingestion_job(
    session: Session,
    job_id: UUID,
    *,
    raw_dir: Path | None = None,
) -> dict[str, Any]:
    job_repo = IngestionJobRepository(session)
    job = job_repo.get(job_id)
    if job is None:
        raise ValueError(f"Ingestion job not found: {job_id}")

    job_repo.mark_running(job)
    metadata = dict(job.metadata_ or {})
    product_html = metadata.get("product_html")
    reviews_file = metadata.get("reviews_file")
    if not product_html or not reviews_file:
        raise ValueError("Ingestion job metadata must include product_html and reviews_file.")

    # Heavy parsing dependencies stay lazy so health checks and lightweight CI do not load them.
    from .chunk import build_chunks
    from .ingest import load_reviews_csv, load_reviews_jsonl, parse_product_html

    base_dir = _resolved_raw_dir(raw_dir)
    product_path = base_dir / str(product_html)
    reviews_path = base_dir / str(reviews_file)
    product_blob = parse_product_html(product_path)
    reviews = (
        load_reviews_jsonl(reviews_path)
        if reviews_path.suffix.lower() == ".jsonl"
        else load_reviews_csv(reviews_path)
    )
    chunks = build_chunks(product_blob, reviews)

    return store_normalized_product(
        session,
        job_id=job_id,
        product_blob=product_blob,
        reviews=reviews,
        chunks=chunks,
        source_url=metadata.get("source_url"),
    )


def run_ingestion_job(job_id: str) -> dict[str, Any]:
    parsed_job_id = UUID(job_id)
    try:
        with session_scope() as session:
            return process_ingestion_job(session, parsed_job_id)
    except Exception as exc:
        with session_scope() as session:
            job = IngestionJobRepository(session).get(parsed_job_id)
            if job is not None:
                IngestionJobRepository(session).mark_failed(job, str(exc))
        raise
