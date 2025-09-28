from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .ingest import ingest_one
from .chunk import chunk_processed
from .indexer import build_indexes, list_products
from .retriever import HybridRetriever
from .answerer import answer_from_passages
from .temporal import analyze_temporal_conflict
from pathlib import Path

app = FastAPI(title="Product QA RAG (MVP)")

class IngestReq(BaseModel):
    product_html: str
    reviews_csv: str

class AskReq(BaseModel):
    product_id: str
    question: str

@app.post("/ingest")
def ingest(req: IngestReq):
    path = ingest_one(req.product_html, req.reviews_csv)
    chunks = chunk_processed(path)
    build_indexes(chunks[0]["product_id"])
    return {"status":"ok", "product_id": chunks[0]["product_id"], "chunks": len(chunks)}

@app.get("/products")
def products():
    return {"products": list_products()}

@app.post("/ask")
def ask(req: AskReq):
    try:
        retr = HybridRetriever(req.product_id)
    except Exception as e:
        raise HTTPException(400, f"Index not found for product_id={req.product_id}: {e}")
    hits = retr.search(req.question, k_dense=30, k_out=8)
    ans = answer_from_passages(req.question, hits, max_sents=2)
    analysis = analyze_temporal_conflict(hits)
    return {"question": req.question, "result": ans, "evidence": hits, "consensus": analysis}
