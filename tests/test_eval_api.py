from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base, EvalDataset, EvalRun
from app.db.repositories import IngestionJobRepository
from app.ingestion_tasks import store_normalized_product


def _sqlite_session_factory():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def _patch_db(monkeypatch, session_factory) -> None:
    from app import db as db_package
    from app.db import session as db_session

    monkeypatch.setattr(db_session, "SessionLocal", session_factory)
    monkeypatch.setattr(db_package, "SessionLocal", session_factory)


def _patch_retrieval(monkeypatch) -> None:
    import app.db_retriever as db_retriever

    monkeypatch.setattr(
        db_retriever,
        "get_settings",
        lambda: SimpleNamespace(
            enable_postgres_native_retrieval=False,
            enable_semantic_retrieval=False,
            enable_reranking=False,
            retrieval_chunk_limit=1000,
        ),
    )


def _seed_product(session_factory) -> None:
    with session_factory() as session:
        job = IngestionJobRepository(session).create(source_type="raw_files")
        store_normalized_product(
            session,
            job_id=job.id,
            product_blob={
                "product_id": "pan",
                "title": "Demo Pan",
                "headings": [],
                "specs": [],
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


def test_eval_dataset_and_run_api(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from app.api import app

    session_factory = _sqlite_session_factory()
    _patch_db(monkeypatch, session_factory)
    _patch_retrieval(monkeypatch)
    _seed_product(session_factory)

    client = TestClient(app)
    unauthenticated = client.post(
        "/api/v1/eval-datasets",
        json={
            "name": "fixture",
            "examples": [
                {
                    "product_id": "pan",
                    "question": "How long did the coating last?",
                    "answerable": True,
                    "gold_spans": [{"source_id": "R1", "relevance": 2}],
                }
            ],
        },
    )
    assert unauthenticated.status_code == 401

    create_dataset = client.post(
        "/api/v1/eval-datasets",
        headers={"x-api-key": "change-me"},
        json={
            "name": "fixture",
            "description": "Tiny CI fixture",
            "examples": [
                {
                    "product_id": "pan",
                    "question": "How long did the coating last?",
                    "answerable": True,
                    "gold_spans": [{"source_id": "R1", "relevance": 2}],
                }
            ],
        },
    )
    assert create_dataset.status_code == 200
    dataset = create_dataset.json()
    assert dataset["name"] == "fixture"
    assert dataset["example_count"] == 1
    assert dataset["examples"][0]["product_id"] == "pan"

    list_response = client.get("/api/v1/eval-datasets")
    assert list_response.status_code == 200
    assert list_response.json()["datasets"][0]["name"] == "fixture"

    run_response = client.post(
        "/api/v1/eval-runs",
        headers={"x-api-key": "change-me"},
        json={"dataset_name": "fixture", "model": "extractive", "k": 1},
    )
    assert run_response.status_code == 200
    run = run_response.json()
    assert run["status"] == "succeeded"
    assert run["metrics"]["evaluated_examples"] == 1
    assert run["metrics"]["retrieval"]["recall_at_1"] == 1.0
    assert run["metrics"]["answers"]["citation_support_rate"] == 0.0

    get_run = client.get(f"/api/v1/eval-runs/{run['id']}")
    assert get_run.status_code == 200
    assert get_run.json()["id"] == run["id"]

    with session_factory() as session:
        assert session.query(EvalDataset).count() == 1
        assert session.query(EvalRun).one().status == "succeeded"


def test_eval_dataset_can_import_jsonl_from_eval_dir(monkeypatch, tmp_path) -> None:
    from pathlib import Path

    from fastapi.testclient import TestClient

    from app.api import app

    session_factory = _sqlite_session_factory()
    _patch_db(monkeypatch, session_factory)

    eval_dir = Path.cwd() / "eval"
    source = eval_dir / "tmp_fixture_eval_dataset.jsonl"
    source.write_text(
        '{"product_id":"pan","question":"Q?","answerable":false,"gold_spans":[]}\n',
        encoding="utf-8",
    )
    try:
        client = TestClient(app)
        response = client.post(
            "/api/v1/eval-datasets",
            headers={"x-api-key": "change-me"},
            json={"name": "from-file", "source_path": "eval/tmp_fixture_eval_dataset.jsonl"},
        )
    finally:
        source.unlink(missing_ok=True)

    assert response.status_code == 200
    assert response.json()["metadata"]["source_path"] == "eval/tmp_fixture_eval_dataset.jsonl"
    assert response.json()["example_count"] == 1
