import json
from datetime import datetime
from time import perf_counter
from typing import Any
from uuid import UUID

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .errors import register_exception_handlers
from .logging_config import configure_logging
from .middleware import RequestContextMiddleware
from .settings import get_settings

load_dotenv()

settings = get_settings()
configure_logging(settings.log_level)

app = FastAPI(title="Online Buying Assistant API")
app.add_middleware(RequestContextMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
register_exception_handlers(app)

router = APIRouter()
v1_router = APIRouter(prefix="/api/v1")


class IngestReq(BaseModel):
    product_html: str
    reviews_csv: str


class AskReq(BaseModel):
    product_id: str
    question: str
    model: str | None = None


class AmazonIngestReq(BaseModel):
    query_or_url: str
    max_pages: int = 3
    sort: str = "recent"


class IngestionJobCreateReq(BaseModel):
    product_html: str
    reviews_file: str
    source_url: str | None = None
    enqueue: bool = True


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _job_payload(job: Any) -> dict[str, Any]:
    return {
        "job_id": str(job.id),
        "product_id": str(job.product_id) if job.product_id else None,
        "status": job.status,
        "source_type": job.source_type,
        "input_uri": job.input_uri,
        "error_message": job.error_message,
        "stats": job.stats or {},
        "metadata": job.metadata_ or {},
        "created_at": _iso(job.created_at),
        "started_at": _iso(job.started_at),
        "completed_at": _iso(job.completed_at),
    }


def _product_payload(product: Any, counts: dict[str, int] | None = None) -> dict[str, Any]:
    return {
        "id": str(product.id),
        "product_id": product.external_id,
        "external_id": product.external_id,
        "title": product.title,
        "source_type": product.source_type,
        "source_url": product.source_url,
        "metadata": product.metadata_ or {},
        "counts": counts or {},
        "created_at": _iso(product.created_at),
        "updated_at": _iso(product.updated_at),
    }


def _section_payload(section: Any) -> dict[str, Any]:
    return {
        "id": str(section.id),
        "source_id": section.source_id,
        "section_type": section.section_type,
        "heading": section.heading,
        "text": section.text,
        "ordinal": section.ordinal,
        "metadata": section.metadata_ or {},
    }


def _review_payload(review: Any) -> dict[str, Any]:
    return {
        "id": str(review.id),
        "source_id": review.source_id,
        "title": review.title,
        "body": review.body,
        "rating": review.rating,
        "date": review.review_date.isoformat() if review.review_date else None,
        "verified": review.verified,
        "helpful_votes": review.helpful_votes,
        "metadata": review.metadata_ or {},
    }


def _chunk_payload(chunk: Any) -> dict[str, Any]:
    return {
        "id": str(chunk.id),
        "source_id": chunk.source_id,
        "source_type": chunk.source_type,
        "text": chunk.text,
        "token_count": chunk.token_count,
        "metadata": chunk.metadata_ or {},
    }


def _answer_from_hits(question: str, model: str | None, hits: list[dict]) -> dict[str, Any]:
    from .answerer import answer_from_passages
    from .faithfulness import evaluate_answer
    from .llm_answerer import generate_llm_answer
    from .temporal import analyze_temporal_conflict

    answer_diagnostics: dict[str, Any] = {"requested_model": model}
    if model == "extractive":
        ans = answer_from_passages(question, hits, max_sents=2)
        ans["engine"] = "extractive"
        answer_diagnostics["engine"] = "extractive"
    else:
        try:
            ans = generate_llm_answer(question, hits, model_name=model)
            ans["engine"] = "llm"
            answer_diagnostics.update(
                {
                    "engine": "llm",
                    "provider": ans.get("provider"),
                    "model": ans.get("model"),
                    "provider_latency_ms": ans.get("provider_latency_ms"),
                }
            )
        except Exception as exc:
            ans = answer_from_passages(question, hits, max_sents=2)
            ans["engine"] = "extractive"
            answer_diagnostics.update(
                {
                    "engine": "extractive",
                    "fallback_reason": str(exc),
                }
            )

    analysis = analyze_temporal_conflict(hits)
    faith = evaluate_answer(ans.get("answer", ""), hits)

    if faith["overall_supported_rate"] < 0.75:
        ans["confidence"] = round(max(0.3, ans.get("confidence", 0.6) * 0.9), 2)

    return {
        "question": question,
        "result": ans,
        "evidence": hits,
        "consensus": analysis,
        "faithfulness": faith,
        "diagnostics": {"answer": answer_diagnostics},
    }


@app.get("/healthz")
def healthz():
    return {"status": "ok", "environment": settings.environment}


@app.get("/readyz")
def readyz():
    return {"status": "ok"}


@router.post("/ingest")
def ingest(req: IngestReq):
    from .chunk import chunk_processed
    from .indexer import build_indexes
    from .ingest import ingest_one

    path = ingest_one(req.product_html, req.reviews_csv)
    chunks = chunk_processed(path)
    if not chunks:
        raise HTTPException(status_code=422, detail="Ingestion produced no indexable chunks.")
    build_indexes(chunks[0]["product_id"])
    title = (chunks[0].get("meta") or {}).get("title") if chunks else None
    return {
        "status": "ok",
        "product_id": chunks[0]["product_id"],
        "title": title or chunks[0]["product_id"],
        "chunks": len(chunks),
    }


@router.get("/products")
def products():
    from .indexer import list_products

    return {"products": list_products()}


@router.get("/products_meta")
def products_meta():
    from .indexer import list_products_meta

    return {"products": list_products_meta()}


@router.post("/ingest_amazon")
def ingest_amazon(req: AmazonIngestReq):
    from .chunk import chunk_processed
    from .config import RAW_DIR
    from .indexer import build_indexes
    from .ingest import ingest_one
    from .scraper import scrape_amazon_sync

    try:
        blob = scrape_amazon_sync(req.query_or_url, max_pages=req.max_pages, sort=req.sort)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    asin = blob["asin"]

    html_path = RAW_DIR / f"{asin}.html"
    reviews_path = RAW_DIR / f"{asin}_reviews.jsonl"
    html_path.write_text(blob["html"], encoding="utf-8")
    with reviews_path.open("w", encoding="utf-8") as f:
        for r in blob["reviews"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    path = ingest_one(html_path.name, reviews_path.name)
    chunks = chunk_processed(path)
    if not chunks:
        raise HTTPException(status_code=422, detail="Amazon ingest produced no indexable chunks.")
    build_indexes(chunks[0]["product_id"])
    reviews_n = len(blob.get("reviews") or [])
    resp = {
        "status": "ok",
        "product_id": chunks[0]["product_id"],
        "title": (chunks[0].get("meta") or {}).get("title") if chunks else None,
        "chunks": len(chunks),
        "asin": asin,
        "reviews": reviews_n,
        "url": blob.get("url", ""),
    }
    if reviews_n == 0:
        resp["warning"] = "No reviews were scraped. Amazon may have served a bot-check page or blocked review access."
    return resp


@router.post("/ask")
def ask(req: AskReq):
    from .retriever import HybridRetriever

    try:
        retr = HybridRetriever(req.product_id)
    except Exception as e:
        raise HTTPException(400, f"Index not found for product_id={req.product_id}: {e}") from e

    hits = retr.search(req.question, k_dense=30, k_out=8)
    return _answer_from_hits(req.question, req.model, hits)


@v1_router.get("/products")
def list_v1_products(limit: int = 100, offset: int = 0):
    from .db.repositories import ProductRepository
    from .db.session import session_scope

    with session_scope() as session:
        repo = ProductRepository(session)
        products = repo.list_products(limit=limit, offset=offset)
        return {
            "products": [
                _product_payload(product, counts=repo.corpus_counts(product.id))
                for product in products
            ]
        }


@v1_router.get("/products/{product_id}")
def get_v1_product(product_id: str):
    from .db.repositories import ProductRepository
    from .db.session import session_scope

    with session_scope() as session:
        repo = ProductRepository(session)
        product = repo.get_by_public_id(product_id)
        if product is None:
            raise HTTPException(status_code=404, detail="Product not found.")
        return {
            **_product_payload(product, counts=repo.corpus_counts(product.id)),
            "sections": [_section_payload(section) for section in repo.list_sections(product.id)],
            "reviews": [_review_payload(review) for review in repo.list_reviews(product.id)],
            "chunks": [_chunk_payload(chunk) for chunk in repo.list_chunks(product.id)],
        }


@v1_router.get("/products/{product_id}/chunks")
def list_v1_product_chunks(product_id: str, limit: int = 50, offset: int = 0):
    from .db.repositories import ProductRepository
    from .db.session import session_scope

    with session_scope() as session:
        repo = ProductRepository(session)
        product = repo.get_by_public_id(product_id)
        if product is None:
            raise HTTPException(status_code=404, detail="Product not found.")
        return {"chunks": [_chunk_payload(chunk) for chunk in repo.list_chunks(product.id, limit, offset)]}


@v1_router.get("/products/{product_id}/reviews")
def list_v1_product_reviews(product_id: str, limit: int = 50, offset: int = 0):
    from .db.repositories import ProductRepository
    from .db.session import session_scope

    with session_scope() as session:
        repo = ProductRepository(session)
        product = repo.get_by_public_id(product_id)
        if product is None:
            raise HTTPException(status_code=404, detail="Product not found.")
        return {
            "reviews": [_review_payload(review) for review in repo.list_reviews(product.id, limit, offset)]
        }


@v1_router.post("/ingestion-jobs")
def create_ingestion_job(req: IngestionJobCreateReq):
    from .db.repositories import IngestionJobRepository
    from .db.session import session_scope

    metadata = {
        "product_html": req.product_html,
        "reviews_file": req.reviews_file,
        "source_url": req.source_url,
    }

    with session_scope() as session:
        repo = IngestionJobRepository(session)
        job = repo.create(
            source_type="raw_files",
            input_uri=f"{req.product_html}:{req.reviews_file}",
            metadata=metadata,
        )
        payload = _job_payload(job)

    if not req.enqueue:
        return payload

    try:
        from .queue import enqueue_ingestion_job

        rq_job_id = enqueue_ingestion_job(payload["job_id"])
    except Exception as e:
        with session_scope() as session:
            repo = IngestionJobRepository(session)
            failed_job = repo.get(UUID(payload["job_id"]))
            if failed_job is not None:
                repo.mark_failed(failed_job, f"Failed to enqueue ingestion job: {e}")
                payload = _job_payload(failed_job)
        raise HTTPException(status_code=503, detail=payload["error_message"]) from e

    with session_scope() as session:
        repo = IngestionJobRepository(session)
        enqueued_job = repo.get(UUID(payload["job_id"]))
        if enqueued_job is None:
            raise HTTPException(status_code=404, detail="Ingestion job not found after enqueue.")
        enqueued_job.metadata_ = {**(enqueued_job.metadata_ or {}), "rq_job_id": rq_job_id}
        payload = _job_payload(enqueued_job)
    return payload


@v1_router.get("/ingestion-jobs")
def list_ingestion_jobs(limit: int = 100, offset: int = 0):
    from .db.repositories import IngestionJobRepository
    from .db.session import session_scope

    with session_scope() as session:
        jobs = IngestionJobRepository(session).list_jobs(limit=limit, offset=offset)
        return {"jobs": [_job_payload(job) for job in jobs]}


@v1_router.get("/ingestion-jobs/{job_id}")
def get_ingestion_job(job_id: UUID):
    from .db.repositories import IngestionJobRepository
    from .db.session import session_scope

    with session_scope() as session:
        job = IngestionJobRepository(session).get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Ingestion job not found.")
        return _job_payload(job)


@v1_router.post("/ask")
def ask_v1(req: AskReq, request: Request):
    from .db.repositories import QueryLogRepository
    from .db.session import session_scope
    from .db_retriever import DatabaseHybridRetriever

    started = perf_counter()
    with session_scope() as session:
        try:
            retriever = DatabaseHybridRetriever(session, req.product_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        hits = retriever.search(req.question, k_dense=30, k_out=8)
        response = _answer_from_hits(req.question, req.model, hits)
        latency_ms = int((perf_counter() - started) * 1000)

        result = response["result"]
        query_log = QueryLogRepository(session).create(
            product_id=retriever.product.id,
            request_id=getattr(request.state, "request_id", None),
            question=req.question,
            answer=result.get("answer"),
            engine=result.get("engine"),
            confidence=result.get("confidence"),
            latency_ms=latency_ms,
            metadata={
                "model": req.model,
                "retrieval": retriever.diagnostics,
                "answer": response.get("diagnostics", {}).get("answer", {}),
            },
            citations=hits,
        )
        response["query_log_id"] = str(query_log.id)
        answer_diagnostics = response.get("diagnostics", {}).get("answer", {})
        response["diagnostics"] = {
            "latency_ms": latency_ms,
            "retrieval": retriever.diagnostics,
            "answer": answer_diagnostics,
        }
        return response


app.include_router(router)
app.include_router(v1_router)


def main() -> None:
    import uvicorn

    uvicorn.run("app.api:app", host="0.0.0.0", port=8000)
