import pickle, json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from .config import INDEX_DIR, EMB_MODEL_NAME, RERANKER_NAME
import torch
device = "mps" if torch.backends.mps.is_available() else "cpu"

def _load_bm25(pid):
    with open(INDEX_DIR / f"{pid}.bm25.pkl", "rb") as f:
        obj = pickle.load(f)
    return obj["bm25"], obj["docs"]

def _load_dense(pid):
    index = faiss.read_index(str(INDEX_DIR / f"{pid}.faiss"))
    with open(INDEX_DIR / f"{pid}.docs.pkl", "rb") as f:
        docs = pickle.load(f)
    return index, docs

class HybridRetriever:
    def __init__(self, product_id: str, w_sparse=0.5, w_dense=0.5):
        self.pid = product_id
        self.w_sparse = w_sparse
        self.w_dense = w_dense
        self.bm25, self.docs_b = _load_bm25(product_id)
        self.faiss, self.docs_d = _load_dense(product_id)
        self.emb_model = SentenceTransformer(EMB_MODEL_NAME)
        self.reranker = CrossEncoder(RERANKER_NAME, device=device)

    def _prior(self, d):
        m = d.get("meta", {})
        if d["source_type"] != "review": return 0.0
        helpful = float(m.get("helpful_votes", 0))
        verified = 1.0 if m.get("verified") else 0.0
        rating = float(m.get("rating", 0))
        return 0.001*helpful + 0.2*verified + 0.02*rating

    def search(self, query: str, k_dense=30, k_out=10, pool_size=40):
        n = self.faiss.ntotal
        k_dense = max(1, min(k_dense, n))

        # dense
        qv = self.emb_model.encode([query], normalize_embeddings=True)
        D, I = self.faiss.search(qv.astype('float32'), k_dense)

        dense_scores = {}
        for idx, s in zip(I[0], D[0]):
            if int(idx) >= 0 and np.isfinite(s) and s > -1e10:
                dense_scores[int(idx)] = float(s)

        # sparse
        toks = query.lower().split()
        bm_scores_arr = self.bm25.get_scores(toks)
        top_sparse_idx = np.argsort(bm_scores_arr)[::-1][:k_dense]
        smax = float(bm_scores_arr[top_sparse_idx].max()) if len(top_sparse_idx) else 1.0
        if smax == 0.0: smax = 1.0
        sparse_scores = {int(i): float(bm_scores_arr[i] / smax) for i in top_sparse_idx}

        # fuse + prior
        prelim = []
        seen_sid = set()
        for idx in set(dense_scores) | set(sparse_scores):
            d = self.docs_d[idx]
            s = 0.5 * dense_scores.get(idx, 0.0) + 0.5 * sparse_scores.get(idx, 0.0)
            score = s + self._prior(d)
            sid = d["source_id"]
            if sid in seen_sid:
                continue
            seen_sid.add(sid)
            prelim.append((idx, score))

        prelim.sort(key=lambda x: x[1], reverse=True)
        pool = prelim[:min(pool_size, len(prelim))]

        if not pool:
            return []

        pairs = [(query, self.docs_d[idx]["text"]) for idx, _ in pool]
        rerank_scores = self.reranker.predict(pairs, convert_to_numpy=True)

        ranked = []
        for (idx, base_score), rr in zip(pool, rerank_scores):
            d = self.docs_d[idx]
            combined = 0.85 * float(rr) + 0.15 * float(base_score)
            ranked.append((idx, combined))

        ranked.sort(key=lambda x: x[1], reverse=True)

        out = []
        non_review_budget = 1
        for idx, score in ranked:
            d = self.docs_d[idx]
            if d["source_type"] != "review":
                if non_review_budget <= 0:
                    continue
                non_review_budget -= 1
            out.append({
                "score": float(score),
                "text": d["text"],
                "source_id": d["source_id"],
                "source_type": d["source_type"],
                "meta": d.get("meta", {})
            })
            if len(out) >= k_out:
                break
        return out
