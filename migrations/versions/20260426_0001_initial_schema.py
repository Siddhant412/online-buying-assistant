"""initial production schema

Revision ID: 20260426_0001
Revises:
Create Date: 2026-04-26 00:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision = "20260426_0001"
down_revision = None
branch_labels = None
depends_on = None


def _uuid_pk() -> sa.Column:
    return sa.Column(
        "id",
        postgresql.UUID(as_uuid=True),
        primary_key=True,
        server_default=sa.text("gen_random_uuid()"),
    )


def _timestamps(include_updated: bool = False) -> list[sa.Column]:
    cols = [
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        )
    ]
    if include_updated:
        cols.append(
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            )
        )
    return cols


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "products",
        _uuid_pk(),
        sa.Column("external_id", sa.String(length=128), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("source_type", sa.String(length=64), nullable=False, server_default="manual"),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        *_timestamps(include_updated=True),
        sa.UniqueConstraint("external_id", name="uq_products_external_id"),
    )
    op.create_index("ix_products_external_id", "products", ["external_id"])

    op.create_table(
        "product_sections",
        _uuid_pk(),
        sa.Column("product_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_id", sa.String(length=128), nullable=False),
        sa.Column("section_type", sa.String(length=64), nullable=False),
        sa.Column("heading", sa.Text(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        *_timestamps(),
        sa.ForeignKeyConstraint(["product_id"], ["products.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("product_id", "source_id", name="uq_product_sections_product_source"),
    )
    op.create_index("ix_product_sections_product_id", "product_sections", ["product_id"])

    op.create_table(
        "reviews",
        _uuid_pk(),
        sa.Column("product_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_id", sa.String(length=128), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("rating", sa.Float(), nullable=True),
        sa.Column("review_date", sa.Date(), nullable=True),
        sa.Column("verified", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("helpful_votes", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        *_timestamps(),
        sa.CheckConstraint("rating IS NULL OR (rating >= 0 AND rating <= 5)", name="ck_reviews_rating"),
        sa.ForeignKeyConstraint(["product_id"], ["products.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("product_id", "source_id", name="uq_reviews_product_source"),
    )
    op.create_index("ix_reviews_product_id", "reviews", ["product_id"])
    op.create_index("ix_reviews_review_date", "reviews", ["review_date"])

    op.create_table(
        "chunks",
        _uuid_pk(),
        sa.Column("product_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_id", sa.String(length=128), nullable=False),
        sa.Column("source_type", sa.String(length=64), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("embedding", Vector(768), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        *_timestamps(),
        sa.ForeignKeyConstraint(["product_id"], ["products.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("product_id", "source_id", name="uq_chunks_product_source"),
    )
    op.create_index("ix_chunks_product_id", "chunks", ["product_id"])
    op.create_index("ix_chunks_source_type", "chunks", ["source_type"])
    op.create_index(
        "ix_chunks_text_fts",
        "chunks",
        [sa.text("to_tsvector('english', text)")],
        postgresql_using="gin",
    )

    op.create_table(
        "ingestion_jobs",
        _uuid_pk(),
        sa.Column("product_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="queued"),
        sa.Column("source_type", sa.String(length=64), nullable=False),
        sa.Column("input_uri", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("stats", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        *_timestamps(),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
            name="ck_ingestion_jobs_status",
        ),
        sa.ForeignKeyConstraint(["product_id"], ["products.id"], ondelete="SET NULL"),
    )
    op.create_index("ix_ingestion_jobs_product_id", "ingestion_jobs", ["product_id"])
    op.create_index("ix_ingestion_jobs_status", "ingestion_jobs", ["status"])

    op.create_table(
        "query_logs",
        _uuid_pk(),
        sa.Column("product_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("request_id", sa.String(length=128), nullable=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("engine", sa.String(length=64), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        *_timestamps(),
        sa.ForeignKeyConstraint(["product_id"], ["products.id"], ondelete="SET NULL"),
    )
    op.create_index("ix_query_logs_product_id", "query_logs", ["product_id"])
    op.create_index("ix_query_logs_request_id", "query_logs", ["request_id"])

    op.create_table(
        "citations",
        _uuid_pk(),
        sa.Column("query_log_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("source_id", sa.String(length=128), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        *_timestamps(),
        sa.ForeignKeyConstraint(["chunk_id"], ["chunks.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["query_log_id"], ["query_logs.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_citations_query_log_id", "citations", ["query_log_id"])
    op.create_index("ix_citations_chunk_id", "citations", ["chunk_id"])

    op.create_table(
        "eval_datasets",
        _uuid_pk(),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        *_timestamps(),
        sa.UniqueConstraint("name", name="uq_eval_datasets_name"),
    )

    op.create_table(
        "eval_examples",
        _uuid_pk(),
        sa.Column("dataset_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("product_external_id", sa.String(length=128), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answerable", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("gold_spans", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="[]"),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        *_timestamps(),
        sa.ForeignKeyConstraint(["dataset_id"], ["eval_datasets.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_eval_examples_dataset_id", "eval_examples", ["dataset_id"])
    op.create_index("ix_eval_examples_product_external_id", "eval_examples", ["product_external_id"])

    op.create_table(
        "eval_runs",
        _uuid_pk(),
        sa.Column("dataset_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("model_name", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="running"),
        sa.Column("metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        sa.Column("error_message", sa.Text(), nullable=True),
        *_timestamps(),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("status IN ('running', 'succeeded', 'failed')", name="ck_eval_runs_status"),
        sa.ForeignKeyConstraint(["dataset_id"], ["eval_datasets.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_eval_runs_dataset_id", "eval_runs", ["dataset_id"])


def downgrade() -> None:
    op.drop_index("ix_eval_runs_dataset_id", table_name="eval_runs")
    op.drop_table("eval_runs")
    op.drop_index("ix_eval_examples_product_external_id", table_name="eval_examples")
    op.drop_index("ix_eval_examples_dataset_id", table_name="eval_examples")
    op.drop_table("eval_examples")
    op.drop_table("eval_datasets")
    op.drop_index("ix_citations_chunk_id", table_name="citations")
    op.drop_index("ix_citations_query_log_id", table_name="citations")
    op.drop_table("citations")
    op.drop_index("ix_query_logs_request_id", table_name="query_logs")
    op.drop_index("ix_query_logs_product_id", table_name="query_logs")
    op.drop_table("query_logs")
    op.drop_index("ix_ingestion_jobs_status", table_name="ingestion_jobs")
    op.drop_index("ix_ingestion_jobs_product_id", table_name="ingestion_jobs")
    op.drop_table("ingestion_jobs")
    op.drop_index("ix_chunks_text_fts", table_name="chunks")
    op.drop_index("ix_chunks_source_type", table_name="chunks")
    op.drop_index("ix_chunks_product_id", table_name="chunks")
    op.drop_table("chunks")
    op.drop_index("ix_reviews_review_date", table_name="reviews")
    op.drop_index("ix_reviews_product_id", table_name="reviews")
    op.drop_table("reviews")
    op.drop_index("ix_product_sections_product_id", table_name="product_sections")
    op.drop_table("product_sections")
    op.drop_index("ix_products_external_id", table_name="products")
    op.drop_table("products")
