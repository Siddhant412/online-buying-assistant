from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, Product
from app.db.repositories import ProductRepository


def test_metadata_contains_production_tables() -> None:
    expected_tables = {
        "products",
        "product_sections",
        "reviews",
        "chunks",
        "ingestion_jobs",
        "query_logs",
        "citations",
        "eval_datasets",
        "eval_examples",
        "eval_runs",
    }

    assert expected_tables.issubset(Base.metadata.tables)


def test_product_repository_upserts_and_lists_products() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    with session_factory() as session:
        repo = ProductRepository(session)
        product = repo.upsert_product(
            external_id="pan",
            title="Original title",
            source_type="fixture",
            metadata={"category": "cookware"},
        )
        repo.add_review(
            product=product,
            source_id="R1",
            title="Great",
            body="Coating lasted 6 months.",
            rating=4,
            verified=True,
            helpful_votes=18,
        )
        session.commit()

        updated = repo.upsert_product(
            external_id="pan",
            title="Updated title",
            source_type="fixture",
            metadata={"category": "cookware"},
        )
        session.commit()

        assert updated.id == product.id
        assert repo.get_by_external_id("pan").title == "Updated title"
        assert [p.external_id for p in repo.list_products()] == ["pan"]
        assert session.query(Product).count() == 1
