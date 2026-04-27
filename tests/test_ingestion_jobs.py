from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import (
    Base,
    Chunk,
    Citation,
    IngestionJob,
    Product,
    ProductSection,
    QueryLog,
    Review,
)
from app.db.repositories import IngestionJobRepository
from app.ingestion_tasks import process_ingestion_job, store_normalized_product


def _sqlite_session_factory():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def test_ingestion_job_api_creates_lists_and_fetches_without_enqueue(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from app import db as db_package
    from app.api import app
    from app.db import session as db_session

    session_factory = _sqlite_session_factory()
    monkeypatch.setattr(db_session, "SessionLocal", session_factory)
    monkeypatch.setattr(db_package, "SessionLocal", session_factory)

    client = TestClient(app)
    create_response = client.post(
        "/api/v1/ingestion-jobs",
        json={
            "product_html": "pan.html",
            "reviews_file": "pan_reviews.csv",
            "enqueue": False,
        },
    )

    assert create_response.status_code == 200
    created = create_response.json()
    assert created["status"] == "queued"
    assert created["metadata"]["product_html"] == "pan.html"
    assert created["metadata"]["reviews_file"] == "pan_reviews.csv"

    list_response = client.get("/api/v1/ingestion-jobs")
    assert list_response.status_code == 200
    assert [job["job_id"] for job in list_response.json()["jobs"]] == [created["job_id"]]

    get_response = client.get(f"/api/v1/ingestion-jobs/{created['job_id']}")
    assert get_response.status_code == 200
    assert get_response.json()["job_id"] == created["job_id"]


def test_store_normalized_product_marks_job_succeeded_and_replaces_corpus() -> None:
    session_factory = _sqlite_session_factory()
    with session_factory() as session:
        job = IngestionJobRepository(session).create(
            source_type="raw_files",
            input_uri="pan.html:pan_reviews.csv",
            metadata={"product_html": "pan.html", "reviews_file": "pan_reviews.csv"},
        )
        session.commit()

        stats = store_normalized_product(
            session,
            job_id=job.id,
            product_blob={
                "product_id": "pan",
                "title": "Demo Pan",
                "headings": ["About this item"],
                "specs": ["Material: Aluminum"],
                "sections": ["Durable non-stick coating."],
            },
            reviews=[
                {
                    "review_id": "R1",
                    "title": "Great",
                    "body": "Coating lasted 6 months.",
                    "rating": 4,
                    "date": "2024-06-12",
                    "verified": True,
                    "helpful_votes": 18,
                }
            ],
            chunks=[
                {
                    "source_id": "R1",
                    "source_type": "review",
                    "text": "Great. Coating lasted 6 months.",
                    "meta": {"rating": 4},
                }
            ],
        )
        session.commit()

        assert stats == {
            "product_external_id": "pan",
            "sections": 1,
            "specs": 1,
            "reviews": 1,
            "chunks": 1,
        }
        assert session.get(IngestionJob, job.id).status == "succeeded"
        assert session.query(Product).count() == 1
        assert session.query(Review).count() == 1
        assert session.query(Chunk).count() == 1

        store_normalized_product(
            session,
            job_id=job.id,
            product_blob={
                "product_id": "pan",
                "title": "Demo Pan v2",
                "headings": [],
                "specs": [],
                "sections": ["Updated page text."],
            },
            reviews=[],
            chunks=[],
        )
        session.commit()

        assert session.query(Product).count() == 1
        assert session.query(Review).count() == 0
        assert session.query(Chunk).count() == 0
        assert session.query(Product).one().title == "Demo Pan v2"


def test_v1_product_catalog_reads_db_products(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from app import db as db_package
    from app.api import app
    from app.db import session as db_session

    session_factory = _sqlite_session_factory()
    monkeypatch.setattr(db_session, "SessionLocal", session_factory)
    monkeypatch.setattr(db_package, "SessionLocal", session_factory)

    with session_factory() as session:
        job = IngestionJobRepository(session).create(source_type="raw_files")
        store_normalized_product(
            session,
            job_id=job.id,
            product_blob={
                "product_id": "pan",
                "title": "Demo Pan",
                "headings": [],
                "specs": ["Material: Aluminum"],
                "sections": ["Durable non-stick coating."],
            },
            reviews=[
                {
                    "review_id": "R1",
                    "title": "Great",
                    "body": "Coating lasted 6 months.",
                    "rating": 4,
                    "date": "2024-06-12",
                    "verified": True,
                    "helpful_votes": 18,
                }
            ],
            chunks=[
                {
                    "source_id": "R1",
                    "source_type": "review",
                    "text": "Great. Coating lasted 6 months.",
                    "meta": {"rating": 4},
                }
            ],
        )
        session.commit()

    client = TestClient(app)
    list_response = client.get("/api/v1/products")
    assert list_response.status_code == 200
    products = list_response.json()["products"]
    assert products[0]["product_id"] == "pan"
    assert products[0]["counts"] == {"sections": 2, "reviews": 1, "chunks": 1}

    detail_response = client.get("/api/v1/products/pan")
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["title"] == "Demo Pan"
    assert [section["section_type"] for section in detail["sections"]] == ["spec", "product_page"]
    assert detail["reviews"][0]["source_id"] == "R1"
    assert detail["chunks"][0]["source_type"] == "review"

    assert client.get("/api/v1/products/pan/chunks").json()["chunks"][0]["source_id"] == "R1"
    assert client.get("/api/v1/products/pan/reviews").json()["reviews"][0]["source_id"] == "R1"


def test_v1_ask_uses_db_retrieval_and_logs_query(monkeypatch) -> None:
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    import app.db_retriever as db_retriever
    from app import db as db_package
    from app.api import app
    from app.db import session as db_session

    session_factory = _sqlite_session_factory()
    monkeypatch.setattr(db_session, "SessionLocal", session_factory)
    monkeypatch.setattr(db_package, "SessionLocal", session_factory)
    monkeypatch.setattr(
        db_retriever,
        "get_settings",
        lambda: SimpleNamespace(
            enable_semantic_retrieval=False,
            enable_reranking=False,
            retrieval_chunk_limit=1000,
        ),
    )

    with session_factory() as session:
        job = IngestionJobRepository(session).create(source_type="raw_files")
        store_normalized_product(
            session,
            job_id=job.id,
            product_blob={
                "product_id": "pan",
                "title": "Demo Pan",
                "headings": [],
                "specs": ["Material: Aluminum"],
                "sections": ["Durable non-stick coating."],
            },
            reviews=[
                {
                    "review_id": "R1",
                    "title": "Great",
                    "body": "Coating lasted 6 months with daily use.",
                    "rating": 4,
                    "date": "2024-06-12",
                    "verified": True,
                    "helpful_votes": 18,
                }
            ],
            chunks=[
                {
                    "source_id": "R1",
                    "source_type": "review",
                    "text": "Great. Coating lasted 6 months with daily use.",
                    "meta": {
                        "rating": 4,
                        "date": "2024-06-12",
                        "verified": True,
                        "helpful_votes": 18,
                    },
                }
            ],
        )
        session.commit()

    client = TestClient(app)
    response = client.post(
        "/api/v1/ask",
        headers={"x-request-id": "req-test-1"},
        json={
            "product_id": "pan",
            "question": "How long did the coating last?",
            "model": "extractive",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["result"]["engine"] == "extractive"
    assert "6 months" in payload["result"]["answer"]
    assert payload["evidence"][0]["source_id"] == "R1"
    assert payload["diagnostics"]["retrieval"]["returned"] == 1
    assert payload["diagnostics"]["retrieval"]["retrieval_backend"] == "python"
    assert payload["diagnostics"]["answer"]["engine"] == "extractive"

    with session_factory() as session:
        query_log = session.query(QueryLog).one()
        assert query_log.request_id == "req-test-1"
        assert query_log.question == "How long did the coating last?"
        assert query_log.answer == payload["result"]["answer"]
        assert query_log.engine == "extractive"
        assert query_log.metadata_["retrieval"]["retrieval_backend"] == "python"
        assert query_log.metadata_["answer"]["engine"] == "extractive"
        citation = session.query(Citation).one()
        assert citation.metadata_["score_components"]["base_score"] > 0


def test_process_ingestion_job_reads_raw_files_into_db(tmp_path) -> None:
    (tmp_path / "demo.html").write_text(
        """
        <html>
          <head><title>Demo Product</title></head>
          <body>
            <h1 id="productTitle">Demo Product</h1>
            <p>Durable coating for daily use.</p>
          </body>
        </html>
        """,
        encoding="utf-8",
    )
    (tmp_path / "demo_reviews.csv").write_text(
        "review_id,title,body,rating,date,verified,helpful_votes\n"
        "R1,Great,Coating lasted 6 months,4,2024-06-12,TRUE,18\n",
        encoding="utf-8",
    )

    session_factory = _sqlite_session_factory()
    with session_factory() as session:
        job = IngestionJobRepository(session).create(
            source_type="raw_files",
            input_uri="demo.html:demo_reviews.csv",
            metadata={"product_html": "demo.html", "reviews_file": "demo_reviews.csv"},
        )
        session.commit()

        stats = process_ingestion_job(session, job.id, raw_dir=tmp_path)
        session.commit()

        assert stats["product_external_id"] == "demo"
        assert stats["reviews"] == 1
        assert stats["chunks"] >= 1
        assert session.get(IngestionJob, job.id).status == "succeeded"
        assert session.query(Product).filter_by(external_id="demo").one().title == "Demo Product"
        assert session.query(ProductSection).count() >= 1
        assert session.query(Review).one().review_date.isoformat() == "2024-06-12"
