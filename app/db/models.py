from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


JSON_TYPE = JSON().with_variant(JSONB, "postgresql")
EMBEDDING_TYPE = Vector(768).with_variant(JSON(), "sqlite")


def uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)


def json_meta() -> Mapped[dict[str, Any]]:
    return mapped_column("metadata", JSON_TYPE, nullable=False, default=dict)


def created_at_col() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


def updated_at_col() -> Mapped[datetime]:
    return mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class Product(Base):
    __tablename__ = "products"

    id: Mapped[uuid.UUID] = uuid_pk()
    external_id: Mapped[str] = mapped_column(String(128), unique=True, index=True, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str] = mapped_column(String(64), nullable=False, default="manual")
    source_url: Mapped[str | None] = mapped_column(Text)
    metadata_: Mapped[dict[str, Any]] = json_meta()
    created_at: Mapped[datetime] = created_at_col()
    updated_at: Mapped[datetime] = updated_at_col()

    sections: Mapped[list[ProductSection]] = relationship(
        back_populates="product",
        cascade="all, delete-orphan",
    )
    reviews: Mapped[list[Review]] = relationship(back_populates="product", cascade="all, delete-orphan")
    chunks: Mapped[list[Chunk]] = relationship(back_populates="product", cascade="all, delete-orphan")


class ProductSection(Base):
    __tablename__ = "product_sections"
    __table_args__ = (
        UniqueConstraint("product_id", "source_id", name="uq_product_sections_product_source"),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    product_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("products.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    source_id: Mapped[str] = mapped_column(String(128), nullable=False)
    section_type: Mapped[str] = mapped_column(String(64), nullable=False)
    heading: Mapped[str | None] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    metadata_: Mapped[dict[str, Any]] = json_meta()
    created_at: Mapped[datetime] = created_at_col()

    product: Mapped[Product] = relationship(back_populates="sections")


class Review(Base):
    __tablename__ = "reviews"
    __table_args__ = (
        UniqueConstraint("product_id", "source_id", name="uq_reviews_product_source"),
        CheckConstraint("rating IS NULL OR (rating >= 0 AND rating <= 5)", name="ck_reviews_rating"),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    product_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("products.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    source_id: Mapped[str] = mapped_column(String(128), nullable=False)
    title: Mapped[str | None] = mapped_column(Text)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    rating: Mapped[float | None] = mapped_column(Float)
    review_date: Mapped[date | None] = mapped_column(Date)
    verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    helpful_votes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    metadata_: Mapped[dict[str, Any]] = json_meta()
    created_at: Mapped[datetime] = created_at_col()

    product: Mapped[Product] = relationship(back_populates="reviews")


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("product_id", "source_id", name="uq_chunks_product_source"),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    product_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("products.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    source_id: Mapped[str] = mapped_column(String(128), nullable=False)
    source_type: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer)
    embedding: Mapped[list[float] | None] = mapped_column(EMBEDDING_TYPE)
    metadata_: Mapped[dict[str, Any]] = json_meta()
    created_at: Mapped[datetime] = created_at_col()

    product: Mapped[Product] = relationship(back_populates="chunks")


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
            name="ck_ingestion_jobs_status",
        ),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    product_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("products.id", ondelete="SET NULL"),
        index=True,
    )
    status: Mapped[str] = mapped_column(String(32), index=True, nullable=False, default="queued")
    source_type: Mapped[str] = mapped_column(String(64), nullable=False)
    input_uri: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)
    stats: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, nullable=False, default=dict)
    metadata_: Mapped[dict[str, Any]] = json_meta()
    created_at: Mapped[datetime] = created_at_col()
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class QueryLog(Base):
    __tablename__ = "query_logs"

    id: Mapped[uuid.UUID] = uuid_pk()
    product_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("products.id", ondelete="SET NULL"),
        index=True,
    )
    request_id: Mapped[str | None] = mapped_column(String(128), index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str | None] = mapped_column(Text)
    engine: Mapped[str | None] = mapped_column(String(64))
    confidence: Mapped[float | None] = mapped_column(Float)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    metadata_: Mapped[dict[str, Any]] = json_meta()
    created_at: Mapped[datetime] = created_at_col()

    citations: Mapped[list[Citation]] = relationship(
        back_populates="query_log",
        cascade="all, delete-orphan",
    )


class Citation(Base):
    __tablename__ = "citations"

    id: Mapped[uuid.UUID] = uuid_pk()
    query_log_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("query_logs.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("chunks.id", ondelete="SET NULL"),
        index=True,
    )
    source_id: Mapped[str] = mapped_column(String(128), nullable=False)
    score: Mapped[float | None] = mapped_column(Float)
    metadata_: Mapped[dict[str, Any]] = json_meta()
    created_at: Mapped[datetime] = created_at_col()

    query_log: Mapped[QueryLog] = relationship(back_populates="citations")


class EvalDataset(Base):
    __tablename__ = "eval_datasets"

    id: Mapped[uuid.UUID] = uuid_pk()
    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    metadata_: Mapped[dict[str, Any]] = json_meta()
    created_at: Mapped[datetime] = created_at_col()

    examples: Mapped[list[EvalExample]] = relationship(
        back_populates="dataset",
        cascade="all, delete-orphan",
    )
    runs: Mapped[list[EvalRun]] = relationship(back_populates="dataset", cascade="all, delete-orphan")


class EvalExample(Base):
    __tablename__ = "eval_examples"

    id: Mapped[uuid.UUID] = uuid_pk()
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("eval_datasets.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    product_external_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answerable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    gold_spans: Mapped[list[dict[str, Any]]] = mapped_column(JSON_TYPE, nullable=False, default=list)
    metadata_: Mapped[dict[str, Any]] = json_meta()
    created_at: Mapped[datetime] = created_at_col()

    dataset: Mapped[EvalDataset] = relationship(back_populates="examples")


class EvalRun(Base):
    __tablename__ = "eval_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('running', 'succeeded', 'failed')",
            name="ck_eval_runs_status",
        ),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("eval_datasets.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="running")
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, nullable=False, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = created_at_col()
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    dataset: Mapped[EvalDataset] = relationship(back_populates="runs")
