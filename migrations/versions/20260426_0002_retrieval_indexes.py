"""add production retrieval indexes

Revision ID: 20260426_0002
Revises: 20260426_0001
Create Date: 2026-04-26 00:00:00.000000
"""

from __future__ import annotations

from alembic import op

revision = "20260426_0002"
down_revision = "20260426_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "ix_chunks_product_source_type",
        "chunks",
        ["product_id", "source_type"],
    )
    op.create_index(
        "ix_chunks_product_created_at",
        "chunks",
        ["product_id", "created_at"],
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_chunks_embedding_cosine
        ON chunks
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
        WHERE embedding IS NOT NULL
        """
    )
    op.create_index(
        "ix_query_logs_product_created_at",
        "query_logs",
        ["product_id", "created_at"],
    )
    op.create_index("ix_citations_source_id", "citations", ["source_id"])


def downgrade() -> None:
    op.drop_index("ix_citations_source_id", table_name="citations")
    op.drop_index("ix_query_logs_product_created_at", table_name="query_logs")
    op.execute("DROP INDEX IF EXISTS ix_chunks_embedding_cosine")
    op.drop_index("ix_chunks_product_created_at", table_name="chunks")
    op.drop_index("ix_chunks_product_source_type", table_name="chunks")
