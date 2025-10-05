from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .ingest import ingest_one
from .chunk import chunk_processed
from .indexer import build_indexes, list_products
from .retriever import HybridRetriever
from .llm_answerer import generate_llm_answer
from .answerer import answer_from_passages
from .temporal import analyze_temporal_conflict
from pathlib import Path
from dotenv import load_dotenv
from .faithfulness import evaluate_answer

load_dotenv()

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

    # Try LLM, on any error, fall back to extractive answer
    try:
        ans = generate_llm_answer(req.question, hits)
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
