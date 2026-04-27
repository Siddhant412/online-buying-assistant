from pathlib import Path


def test_initial_migration_defines_pgvector_schema() -> None:
    migration = Path("migrations/versions/20260426_0001_initial_schema.py").read_text(
        encoding="utf-8"
    )

    assert "CREATE EXTENSION IF NOT EXISTS vector" in migration
    assert "Vector(768)" in migration
    assert "products" in migration
    assert "chunks" in migration
    assert "eval_runs" in migration


def test_retrieval_index_migration_tunes_pgvector_and_query_logs() -> None:
    migration = Path("migrations/versions/20260426_0002_retrieval_indexes.py").read_text(
        encoding="utf-8"
    )

    assert "ivfflat" in migration
    assert "vector_cosine_ops" in migration
    assert "ix_chunks_product_source_type" in migration
    assert "ix_query_logs_product_created_at" in migration
