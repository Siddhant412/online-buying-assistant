import json, pickle
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
from .config import PROC_DIR, INDEX_DIR, EMB_MODEL_NAME

def _load_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def build_indexes(product_id: str):
    jsonl = PROC_DIR / f"{product_id}_chunks.jsonl"
    docs = list(_load_jsonl(jsonl))
    texts = [d["text"] for d in docs]

    # --- BM25 ---
    tok_docs = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tok_docs)
    with open(INDEX_DIR / f"{product_id}.bm25.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "docs": docs}, f)

    # --- Dense + FAISS ---
    emb_model = SentenceTransformer(EMB_MODEL_NAME)
    emb = emb_model.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb.astype('float32'))

    faiss.write_index(index, str(INDEX_DIR / f"{product_id}.faiss"))
    np.save(INDEX_DIR / f"{product_id}.emb.npy", emb)
    with open(INDEX_DIR / f"{product_id}.docs.pkl", "wb") as f:
        pickle.dump(docs, f)

def list_products():
    return sorted({p.name.split("_chunks.jsonl")[0] for p in PROC_DIR.glob("*_chunks.jsonl")})
