from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .ingest import ingest_one
from .chunk import chunk_processed
from .indexer import build_indexes, list_products, list_products_meta
from .retriever import HybridRetriever
from .llm_answerer import generate_llm_answer
from .answerer import answer_from_passages
from .temporal import analyze_temporal_conflict
from pathlib import Path
import json
from dotenv import load_dotenv
from .faithfulness import evaluate_answer
from .scraper import scrape_amazon_sync
from .config import RAW_DIR

load_dotenv()

app = FastAPI(title="Product QA RAG (MVP)")

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

@app.post("/ingest")
def ingest(req: IngestReq):
    path = ingest_one(req.product_html, req.reviews_csv)
    chunks = chunk_processed(path)
    build_indexes(chunks[0]["product_id"])
    title = (chunks[0].get("meta") or {}).get("title") if chunks else None
    return {"status":"ok", "product_id": chunks[0]["product_id"], "title": title or chunks[0]["product_id"], "chunks": len(chunks)}

@app.get("/products")
def products():
    return {"products": list_products()}

@app.get("/products_meta")
def products_meta():
    return {"products": list_products_meta()}

@app.post("/ingest_amazon")
def ingest_amazon(req: AmazonIngestReq):
    # scrape
    try:
        blob = scrape_amazon_sync(req.query_or_url, max_pages=req.max_pages, sort=req.sort)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    asin = blob["asin"]
    # persist raw
    html_path = RAW_DIR / f"{asin}.html"
    reviews_path = RAW_DIR / f"{asin}_reviews.jsonl"
    html_path.write_text(blob["html"], encoding="utf-8")
    with reviews_path.open("w", encoding="utf-8") as f:
        for r in blob["reviews"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # ingest normalized
    path = ingest_one(html_path.name, reviews_path.name)
    chunks = chunk_processed(path)
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

@app.post("/ask")
def ask(req: AskReq):
    try:
        retr = HybridRetriever(req.product_id)
    except Exception as e:
        raise HTTPException(400, f"Index not found for product_id={req.product_id}: {e}")

    hits = retr.search(req.question, k_dense=30, k_out=8)

    try:
        ans = generate_llm_answer(req.question, hits, model_name=req.model)
        ans["engine"] = "llm"
    except Exception as e:
        ans = answer_from_passages(req.question, hits, max_sents=2)
        ans["engine"] = "extractive"

    analysis = analyze_temporal_conflict(hits)

    faith = evaluate_answer(ans.get("answer",""), hits)

    if faith["overall_supported_rate"] < 0.75:
        ans["confidence"] = round(max(0.3, ans.get("confidence", 0.6) * 0.9), 2)

    return {
        "question": req.question,
        "result": ans,
        "evidence": hits,
        "consensus": analysis,
        "faithfulness": faith
    }
