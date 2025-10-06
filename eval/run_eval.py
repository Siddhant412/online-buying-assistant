from __future__ import annotations
import argparse, json, sys, math
from pathlib import Path
from typing import Dict, List
from statistics import mean

sys.path.append(str(Path(__file__).resolve().parents[1]))  # add project root to PYTHONPATH
from app.retriever import HybridRetriever
from app.llm_answerer import generate_llm_answer
from app.answerer import answer_from_passages
from eval.metrics import recall_at_k, ndcg_at_k, citation_support_rate, predicted_unanswerable

def load_dataset(path: Path) -> List[Dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            items.append(json.loads(line))
    return items

def ranked_ids_from_hits(hits: List[Dict]) -> List[str]:
    return [h["source_id"] for h in hits]

def gold_rel_map(gold_spans: List[Dict]) -> Dict[str, float]:
    out = {}
    for s in gold_spans:
        rid = s["source_id"]
        rel = float(s.get("relevance", 1.0))
        out[rid] = max(out.get(rid, 0.0), rel)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True, help="JSONL with product_id, question, answerable, gold_spans[]")
    ap.add_argument("--k", type=int, default=5, help="k for Recall@k and nDCG@k")
    ap.add_argument("--disable_llm", action="store_true", help="force extractive fallback")
    args = ap.parse_args()

    data = load_dataset(args.dataset)

    r_recall, r_ndcg = [], []
    sent_csr = []     # citation support rate per-sentence
    tn_flags, fp_flags = [], []

    for ex in data:
        pid = ex["product_id"]
        q   = ex["question"]
        answerable = bool(ex.get("answerable", True))
        gold = ex.get("gold_spans", [])
        gold_ids = [g["source_id"] for g in gold]
        gold_rel = gold_rel_map(gold)

        # retrieval (using current retriever, which includes reranker)
        retr = HybridRetriever(pid)
        hits = retr.search(q, k_dense=30, k_out=max(args.k, 10))
        ranked_ids = ranked_ids_from_hits(hits)

        # retrieval metrics (skip on unanswerables for recall/ndcg)
        if answerable and gold_ids:
            r_recall.append(recall_at_k(gold_ids, ranked_ids, args.k))
            r_ndcg.append(ndcg_at_k(ranked_ids, gold_rel, args.k))

        # answer
        if args.disable_llm:
            ans = answer_from_passages(q, hits, max_sents=2)
        else:
            try:
                ans = generate_llm_answer(q, hits)
            except Exception:
                ans = answer_from_passages(q, hits, max_sents=2)

        atext = ans.get("answer","")

        # citation support rate
        if answerable and gold_ids:
            sent_csr.append(citation_support_rate(atext, gold_ids))
        # unanswerable calibration
        pred_unans = predicted_unanswerable(atext)
        if not answerable:
            tn_flags.append(1 if pred_unans else 0)
        else:
            fp_flags.append(1 if pred_unans else 0)

    def avg(xs):
        xs = [x for x in xs if not (isinstance(x,float) and (math.isnan(x) or math.isinf(x)))]
        return (sum(xs)/len(xs)) if xs else float("nan")

    print("\n=== Retrieval ===")
    print(f"Recall@{args.k}: {avg(r_recall):.3f}   (over {len(r_recall)} answerable items)")
    print(f"nDCG@{args.k}:   {avg(r_ndcg):.3f}     (over {len(r_ndcg)} answerable items)")

    print("\n=== Answer Quality ===")
    print(f"Citation support rate (per sentence): {avg(sent_csr):.3f}  (answerable items)")

    print("\n=== Unanswerables ===")
    if tn_flags:
        print(f"True Negative rate (correctly said 'insufficient evidence'): {avg(tn_flags):.3f}  (over {len(tn_flags)} unanswerables)")
    else:
        print("No unanswerables in dataset.")
    if fp_flags:
        print(f"False Positive rate (said 'insufficient' when answerable): {avg(fp_flags):.3f}  (over {len(fp_flags)} answerables)")
    else:
        print("No answerables flagged as 'insufficient' to evaluate FP rate.")

if __name__ == "__main__":
    main()
