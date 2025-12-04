from __future__ import annotations
import argparse, json, sys, math, os, csv
from pathlib import Path
from typing import Dict, List, Tuple, Any

sys.path.append(str(Path(__file__).resolve().parents[1]))  # add project root to PYTHONPATH
from app.retriever import HybridRetriever
from app.llm_answerer import generate_llm_answer
from app.answerer import answer_from_passages
from eval.metrics import recall_at_k, ndcg_at_k, citation_support_rate, predicted_unanswerable

DEFAULT_DATASET = Path(os.getenv("EVAL_DATASET", "eval/groundtruth_dataset.jsonl"))
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")


def load_dataset(path: Path) -> List[Dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
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


def avg(xs):
    xs = [x for x in xs if not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))]
    return (sum(xs) / len(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        action="append",
        type=Path,
        help="Repeatable JSONL paths with product_id, question, answerable, gold_spans[]. Defaults to EVAL_DATASET env.",
    )
    ap.add_argument("--k", type=int, default=5, help="k for Recall@k and nDCG@k")
    ap.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of Ollama models to compare. Use 'extractive' to include the extractive baseline.",
    )
    ap.add_argument(
        "--ollama_url",
        type=str,
        default=None,
        help="Override Ollama URL (defaults to OLLAMA_URL env).",
    )
    ap.add_argument(
        "--include_extractive",
        action="store_true",
        help="Also evaluate the extractive baseline alongside LLM models.",
    )
    ap.add_argument("--disable_llm", action="store_true", help="Force extractive fallback only (ignores --models).")
    ap.add_argument("--export_csv", type=Path, help="Optional path to write per-example results as CSV")
    ap.add_argument("--export_json", type=Path, help="Optional path to write per-example results as JSON")
    args = ap.parse_args()

    dataset_paths = args.dataset or [DEFAULT_DATASET]

    data = []
    dataset_sizes = {}
    for path in dataset_paths:
        items = load_dataset(path)
        dataset_sizes[path.name] = len(items)
        for it in items:
            it["_dataset"] = path.name
            data.append(it)

    if not data:
        print("No examples found across datasets. Exiting.")
        sys.exit(1)

    total_answerable = sum(1 for ex in data if ex.get("answerable", True))
    total_unanswerable = len(data) - total_answerable

    # Precompute retrieval once (model-agnostic)
    hits_cache: List[Tuple[Dict, List[Dict], List[str]]] = []
    r_recall, r_ndcg = [], []
    for ex in data:
        pid = ex["product_id"]
        q = ex["question"]
        answerable = bool(ex.get("answerable", True))
        gold = ex.get("gold_spans", [])
        gold_ids = [g["source_id"] for g in gold]
        gold_rel = gold_rel_map(gold)

        retr = HybridRetriever(pid)
        hits = retr.search(q, k_dense=30, k_out=max(args.k, 10))
        ranked_ids = ranked_ids_from_hits(hits)
        hits_cache.append((ex, hits, ranked_ids))

        if answerable and gold_ids:
            r_recall.append(recall_at_k(gold_ids, ranked_ids, args.k))
            r_ndcg.append(ndcg_at_k(ranked_ids, gold_rel, args.k))

    print("\nDatasets:")
    for name, n in dataset_sizes.items():
        print(f"  {name}: {n} examples")

    print("\n=== Retrieval (model-agnostic) ===")
    print(f"Recall@{args.k}: {avg(r_recall):.3f}   (over {len(r_recall)} answerable items)")
    print(f"nDCG@{args.k}:   {avg(r_ndcg):.3f}     (over {len(r_ndcg)} answerable items)")
    print(f"Answerable items: {total_answerable} | Unanswerable items: {total_unanswerable}")

    # Build model list
    if args.disable_llm:
        models = ["extractive"]
    else:
        if args.models:
            models = [m.strip() for m in args.models.split(",") if m.strip()]
        else:
            models = [DEFAULT_MODEL]
        if args.include_extractive and "extractive" not in models:
            models.append("extractive")
        if not models:
            models = ["extractive"]

    # Evaluate answers per model
    summary_rows = []
    per_example_rows: List[Dict[str, Any]] = []
    for model in models:
        sent_csr = []
        tn_flags, fp_flags = [], []
        llm_errors = 0
        calib_records = []

        for ex, hits, _ranked_ids in hits_cache:
            pid = ex["product_id"]
            q = ex["question"]
            answerable = bool(ex.get("answerable", True))
            gold = ex.get("gold_spans", [])
            gold_ids = [g["source_id"] for g in gold]

            if model == "extractive":
                ans = answer_from_passages(q, hits, max_sents=2)
            else:
                try:
                    ans = generate_llm_answer(q, hits, model_name=model, ollama_url=args.ollama_url)
                except Exception:
                    llm_errors += 1
                    ans = answer_from_passages(q, hits, max_sents=2)

            atext = ans.get("answer", "")

            csr_ex = citation_support_rate(atext, gold_ids) if (answerable and gold_ids) else None
            if csr_ex is not None:
                sent_csr.append(csr_ex)

            pred_unans = predicted_unanswerable(atext)
            if not answerable:
                tn_flags.append(1 if pred_unans else 0)
            else:
                fp_flags.append(1 if pred_unans else 0)

            confidence = float(ans.get("confidence", 0.0))
            # simple per-example "accuracy": citation support > 0 for answerables, correct insufficient for unanswerables
            if answerable:
                acc = 1.0 if (csr_ex is not None and csr_ex > 0) else 0.0
            else:
                acc = 1.0 if pred_unans else 0.0
            calib_records.append({"conf": confidence, "acc": acc})

            per_example_rows.append({
                "dataset": ex.get("_dataset", ""),
                "product_id": pid,
                "question": q,
                "answerable": answerable,
                "gold_ids": ";".join(gold_ids),
                "model": model,
                "confidence": confidence,
                "pred_unanswerable": bool(pred_unans),
                "citation_support_rate": csr_ex if csr_ex is not None else "",
                "accuracy": acc
            })

        csr_val = avg(sent_csr)
        tn_rate = avg(tn_flags)
        fp_rate = avg(fp_flags)

        summary_rows.append({
            "model": model,
            "csr": csr_val,
            "csr_n": len(sent_csr),
            "tn": sum(tn_flags),
            "tn_n": len(tn_flags),
            "tn_rate": tn_rate,
            "fp": sum(fp_flags),
            "fp_n": len(fp_flags),
            "fp_rate": fp_rate,
            "llm_errors": llm_errors
        })

        print(f"\n=== Answers: model={model} ===")
        if llm_errors:
            print(f"(Fallbacks to extractive due to {llm_errors} LLM errors)")
        print(f"Citation support rate (per sentence): {csr_val:.3f}  (over {len(sent_csr)} answerable items)")
        if len(tn_flags):
            print(f"Unanswerables TN: {sum(tn_flags)}/{len(tn_flags)} ({tn_rate:.3f})")
        else:
            print("Unanswerables TN: n/a (no unanswerables)")
        if len(fp_flags):
            print(f"Answerables flagged 'insufficient' (FP): {sum(fp_flags)}/{len(fp_flags)} ({fp_rate:.3f})")
        else:
            print("Answerables flagged 'insufficient' (FP): n/a (no answerables)")

    # Compact comparison table
    if summary_rows:
        print("\n=== Comparison Summary ===")
        print("model\tcsr\tTN (unans)\tFP (ans)\terrors")
        for row in summary_rows:
            tn_txt = f"{row['tn']}/{row['tn_n']} ({row['tn_rate']:.3f})" if row["tn_n"] else "n/a"
            fp_txt = f"{row['fp']}/{row['fp_n']} ({row['fp_rate']:.3f})" if row["fp_n"] else "n/a"
            err_txt = str(row["llm_errors"]) if row["llm_errors"] else "-"
            print(f"{row['model']}\t{row['csr']:.3f}\t{tn_txt}\t{fp_txt}\t{err_txt}")

    # Confidence calibration bins
    if per_example_rows:
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
        print("\n=== Confidence Calibration (accuracy by confidence bins) ===")
        for model in models:
            recs = [r for r in per_example_rows if r["model"] == model]
            if not recs:
                continue
            print(f"\nModel: {model}")
            for i in range(len(bins) - 1):
                lo, hi = bins[i], bins[i + 1]
                bucket = [r for r in recs if lo <= float(r["confidence"]) < hi]
                acc = avg([r["accuracy"] for r in bucket]) if bucket else float("nan")
                print(f"  {lo:.1f}–{hi:.1f}: n={len(bucket)}, acc={acc:.3f}")

    # Exports
    if args.export_csv and per_example_rows:
        with args.export_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_example_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_example_rows)
        print(f"\nWrote CSV: {args.export_csv}")
    if args.export_json and per_example_rows:
        with args.export_json.open("w", encoding="utf-8") as f:
            json.dump(per_example_rows, f, ensure_ascii=False, indent=2)
        print(f"Wrote JSON: {args.export_json}")


if __name__ == "__main__":
    main()
