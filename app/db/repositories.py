from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .models import (
    Chunk,
    Citation,
    IngestionJob,
    Product,
    ProductSection,
    QueryLog,
    Review,
)


def _parse_review_date(value: Any):
    if value is None:
        return None
    if hasattr(value, "isoformat") and not isinstance(value, str):
        return value
    try:
        return datetime.fromisoformat(str(value)[:10]).date()
    except ValueError:
        return None


class ProductRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_external_id(self, external_id: str) -> Product | None:
        stmt = select(Product).where(Product.external_id == external_id)
        return self.session.scalar(stmt)

    def get_by_id(self, product_id: UUID) -> Product | None:
        return self.session.get(Product, product_id)

    def list_products(self, limit: int = 100, offset: int = 0) -> Sequence[Product]:
        stmt = select(Product).order_by(Product.created_at.desc()).limit(limit).offset(offset)
        return self.session.scalars(stmt).all()

    def get_by_public_id(self, product_id: str) -> Product | None:
        try:
            parsed_id = UUID(product_id)
        except ValueError:
            return self.get_by_external_id(product_id)
        return self.get_by_id(parsed_id) or self.get_by_external_id(product_id)

    def corpus_counts(self, product_id: UUID) -> dict[str, int]:
        return {
            "sections": self.session.query(ProductSection).filter_by(product_id=product_id).count(),
            "reviews": self.session.query(Review).filter_by(product_id=product_id).count(),
            "chunks": self.session.query(Chunk).filter_by(product_id=product_id).count(),
        }

    def list_sections(self, product_id: UUID, limit: int = 50, offset: int = 0) -> Sequence[ProductSection]:
        stmt = (
            select(ProductSection)
            .where(ProductSection.product_id == product_id)
            .order_by(ProductSection.ordinal.asc(), ProductSection.created_at.asc())
            .limit(limit)
            .offset(offset)
        )
        return self.session.scalars(stmt).all()

    def list_reviews(self, product_id: UUID, limit: int = 20, offset: int = 0) -> Sequence[Review]:
        stmt = (
            select(Review)
            .where(Review.product_id == product_id)
            .order_by(Review.review_date.desc().nullslast(), Review.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return self.session.scalars(stmt).all()

    def list_chunks(self, product_id: UUID, limit: int = 20, offset: int = 0) -> Sequence[Chunk]:
        stmt = (
            select(Chunk)
            .where(Chunk.product_id == product_id)
            .order_by(Chunk.created_at.asc())
            .limit(limit)
            .offset(offset)
        )
        return self.session.scalars(stmt).all()

    def list_retrieval_chunks(self, product_id: UUID, limit: int = 1000) -> Sequence[Chunk]:
        stmt = (
            select(Chunk)
            .where(Chunk.product_id == product_id)
            .order_by(Chunk.created_at.asc())
            .limit(limit)
        )
        return self.session.scalars(stmt).all()

    def upsert_product(
        self,
        *,
        external_id: str,
        title: str,
        source_type: str = "manual",
        source_url: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Product:
        product = self.get_by_external_id(external_id)
        if product is None:
            product = Product(external_id=external_id, title=title)
            self.session.add(product)

        product.title = title
        product.source_type = source_type
        product.source_url = source_url
        product.metadata_ = metadata or {}
        return product

    def add_review(
        self,
        *,
        product: Product,
        source_id: str,
        body: str,
        title: str | None = None,
        rating: float | None = None,
        verified: bool = False,
        helpful_votes: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> Review:
        review = Review(
            product=product,
            source_id=source_id,
            title=title,
            body=body,
            rating=rating,
            verified=verified,
            helpful_votes=helpful_votes,
            metadata_=metadata or {},
        )
        self.session.add(review)
        return review

    def replace_product_corpus(
        self,
        *,
        product_blob: dict[str, Any],
        reviews: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
        source_type: str = "manual",
        source_url: str | None = None,
    ) -> Product:
        product = self.upsert_product(
            external_id=str(product_blob["product_id"]),
            title=str(product_blob.get("title") or product_blob["product_id"]),
            source_type=source_type,
            source_url=source_url,
            metadata={"headings": product_blob.get("headings", [])},
        )
        self.session.flush()

        self.session.execute(delete(ProductSection).where(ProductSection.product_id == product.id))
        self.session.execute(delete(Review).where(Review.product_id == product.id))
        self.session.execute(delete(Chunk).where(Chunk.product_id == product.id))

        sections = []
        for idx, spec in enumerate(product_blob.get("specs") or []):
            sections.append(
                ProductSection(
                    product=product,
                    source_id=f"spec:{idx}",
                    section_type="spec",
                    text=str(spec),
                    ordinal=idx,
                )
            )
        for idx, text in enumerate(product_blob.get("sections") or []):
            sections.append(
                ProductSection(
                    product=product,
                    source_id=f"section:{idx}",
                    section_type="product_page",
                    text=str(text),
                    ordinal=idx,
                )
            )

        review_rows = [
            Review(
                product=product,
                source_id=str(review.get("review_id") or f"review:{idx}"),
                title=str(review.get("title") or "") or None,
                body=str(review.get("body") or ""),
                rating=float(review["rating"]) if review.get("rating") is not None else None,
                review_date=_parse_review_date(review.get("date")),
                verified=bool(review.get("verified", False)),
                helpful_votes=int(review.get("helpful_votes") or 0),
                metadata_={
                    k: v
                    for k, v in review.items()
                    if k not in {"review_id", "title", "body", "rating", "date", "verified", "helpful_votes"}
                },
            )
            for idx, review in enumerate(reviews)
            if str(review.get("body") or review.get("title") or "").strip()
        ]

        chunk_rows = [
            Chunk(
                product=product,
                source_id=str(chunk["source_id"]),
                source_type=str(chunk["source_type"]),
                text=str(chunk["text"]),
                token_count=len(str(chunk["text"]).split()),
                metadata_=dict(chunk.get("meta") or {}),
            )
            for chunk in chunks
            if str(chunk.get("text") or "").strip()
        ]

        self.session.add_all([*sections, *review_rows, *chunk_rows])
        return product


class IngestionJobRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(
        self,
        *,
        source_type: str,
        input_uri: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IngestionJob:
        job = IngestionJob(
            source_type=source_type,
            input_uri=input_uri,
            status="queued",
            metadata_=metadata or {},
            stats={},
        )
        self.session.add(job)
        self.session.flush()
        return job

    def get(self, job_id: UUID) -> IngestionJob | None:
        return self.session.get(IngestionJob, job_id)

    def list_jobs(self, limit: int = 100, offset: int = 0) -> Sequence[IngestionJob]:
        stmt = select(IngestionJob).order_by(IngestionJob.created_at.desc()).limit(limit).offset(offset)
        return self.session.scalars(stmt).all()

    def mark_running(self, job: IngestionJob) -> None:
        job.status = "running"
        job.started_at = datetime.now(UTC)

    def mark_succeeded(
        self,
        job: IngestionJob,
        *,
        product_id: UUID,
        stats: dict[str, Any],
    ) -> None:
        job.status = "succeeded"
        job.product_id = product_id
        job.stats = stats
        job.error_message = None
        job.completed_at = datetime.now(UTC)

    def mark_failed(self, job: IngestionJob, error_message: str) -> None:
        job.status = "failed"
        job.error_message = error_message
        job.completed_at = datetime.now(UTC)


class QueryLogRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(
        self,
        *,
        product_id: UUID | None,
        request_id: str | None,
        question: str,
        answer: str | None,
        engine: str | None,
        confidence: float | None,
        latency_ms: int | None,
        metadata: dict[str, Any] | None = None,
        citations: list[dict[str, Any]] | None = None,
    ) -> QueryLog:
        query_log = QueryLog(
            product_id=product_id,
            request_id=request_id,
            question=question,
            answer=answer,
            engine=engine,
            confidence=confidence,
            latency_ms=latency_ms,
            metadata_=metadata or {},
        )
        self.session.add(query_log)
        self.session.flush()

        for citation in citations or []:
            chunk_id = citation.get("chunk_id")
            if isinstance(chunk_id, str):
                try:
                    chunk_id = UUID(chunk_id)
                except ValueError:
                    chunk_id = None
            self.session.add(
                Citation(
                    query_log=query_log,
                    chunk_id=chunk_id,
                    source_id=str(citation["source_id"]),
                    score=float(citation["score"]) if citation.get("score") is not None else None,
                    metadata_={
                        **dict(citation.get("meta") or citation.get("metadata") or {}),
                        "score_components": citation.get("score_components", {}),
                    },
                )
            )
        return query_log
