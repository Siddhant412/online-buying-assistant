import os, json, textwrap, sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import streamlit as st
import requests
from dotenv import load_dotenv

# ensure project root is ahead of the UI dir so `import app.*` resolves to the package, not this file
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path = [str(ROOT)] + [p for p in sys.path if p != str(SCRIPT_DIR) and p != str(ROOT)]

from app.retriever import HybridRetriever
from app.llm_answerer import generate_llm_answer
from app.answerer import answer_from_passages
from eval.metrics import recall_at_k, ndcg_at_k, citation_support_rate, predicted_unanswerable

load_dotenv()
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
DEFAULT_DATASETS = [
    "eval/groundtruth_dataset.jsonl",
    "eval/groundtruth_dataset_additional.jsonl"
]

def api_get_products():
    r = requests.get(f"{API_URL}/products", timeout=15)
    r.raise_for_status()
    return r.json().get("products", [])

def api_ingest(product_html: str, reviews_csv: str):
    r = requests.post(f"{API_URL}/ingest", json={"product_html": product_html, "reviews_csv": reviews_csv}, timeout=120)
    r.raise_for_status()
    return r.json()

def api_ask(pid: str, q: str, model: str | None = None):
    payload = {"product_id": pid, "question": q}
    if model:
        payload["model"] = model
    r = requests.post(f"{API_URL}/ask", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

def badge_for_citation(c):
    stype = c.get("source_type")
    sid = c.get("source_id")
    meta = c.get("meta", {})
    if stype == "review":
        rating = meta.get("rating", "")
        date = meta.get("date", "")
        verified = meta.get("verified", False)
        hv = meta.get("helpful_votes", 0)
        vtxt = "✅ verified" if verified else "⚪ unverified"
        return f"`[{sid}]`  ⭐{rating}  {vtxt}  {date}  👍{hv}"
    elif stype == "spec":
        return f"`[{sid}]` spec"
    else:
        return f"`[{sid}]` {stype}"

def render_evidence(evi):
    for i, p in enumerate(evi, 1):
        with st.expander(f"{i}. [{p['source_id']}] {p['source_type']}  —  score: {p['score']:.3f}", expanded=(i == 1)):
            meta = p.get("meta", {})
            if p["source_type"] == "review":
                st.write(f"⭐ {meta.get('rating','?')}  |  "
                        f"{'✅ verified' if meta.get('verified') else '⚪︎ unverified'}  |  "
                        f"date: {meta.get('date','?')}  |  👍 {meta.get('helpful_votes',0)}")
            txt = p["text"].strip()
            if len(txt) > 1000:
                txt = txt[:1000] + " …"
            st.write(txt)

def render_consensus(cons):
    st.caption(cons.get("summary",""))
    by_win = cons.get("by_window", [])
    if by_win:
        df = pd.DataFrame(by_win)[["window","pos","neg","neu"]].set_index("window")
        st.bar_chart(df)

# eval helpers
def _load_dataset(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            obj["_dataset"] = os.path.basename(path)
            items.append(obj)
    return items

def _gold_rel_map(gold_spans):
    out = {}
    for s in gold_spans:
        rid = s["source_id"]
        rel = float(s.get("relevance", 1.0))
        out[rid] = max(out.get(rid, 0.0), rel)
    return out

def _avg(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and (pd.isna(x) or pd.isnull(x)))]
    return (sum(xs)/len(xs)) if xs else 0.0

# UI
st.set_page_config(page_title="Product QA Autopilot", layout="wide")
st.title("🛒 Product QA Autopilot (MVP)")

with st.sidebar:
    st.subheader("Settings")
    API_URL = st.text_input("FastAPI URL", API_URL)
    st.caption("Tip: Keep your FastAPI server running in another terminal:\n`uvicorn app.api:app --reload`")
    st.divider()
    st.subheader("Quick links")
    if st.button("Refresh products"):
        st.session_state["_products"] = api_get_products()

tabs = st.tabs(["Ask", "Ingest", "Eval"])
# preload product list
if "_products" not in st.session_state:
    try:
        st.session_state["_products"] = api_get_products()
    except Exception as e:
        st.session_state["_products"] = []
        st.sidebar.error(f"Could not reach API: {e}")

MODEL_CHOICES = ["llama3.1:8b", "mistral:7b", "qwen2:7b", "phi3:14b", "extractive"]

with tabs[0]:
    col1, col2, col3 = st.columns([2, 3, 2])
    with col1:
        products = st.session_state.get("_products", [])
        pid = st.selectbox("Product ID", options=(products or ["(none)"]), index=0)
        manual_pid = st.text_input("…or type a product_id", value="" if products else "")
        chosen_pid = manual_pid.strip() or (pid if pid != "(none)" else "")
    with col2:
        q = st.text_input("Your question", value="How long does the non-stick coating last?")
    with col3:
        model_choice = st.selectbox("Model", options=MODEL_CHOICES, index=0, help="Select LLM or extractive fallback.")

    ask_btn = st.button("Ask", type="primary", disabled=not (chosen_pid and q))
    if ask_btn:
        try:
            chosen_model = None if model_choice == "extractive" else model_choice
            res = api_ask(chosen_pid, q, model=chosen_model)
            result = res.get("result", {})
            evidence = res.get("evidence", [])
            consensus = res.get("consensus", {})
            faith = res.get("faithfulness", {})

            # answer block
            st.subheader("Answer")
            st.write(result.get("answer","(no answer)"))
            engine = result.get("engine","?")
            st.caption(f"Engine: {engine} | Confidence: {result.get('confidence',0):.2f}")

            if faith:
                st.subheader("Faithfulness")
                st.caption(f"Supported sentences: {faith.get('overall_supported_rate',0):.2f}")
                for i, s in enumerate(faith.get("sentences", []), 1):
                    status = "✅ supported" if s["supported"] else "⚠️ needs evidence"
                    st.write(f"{i}. {status} — cites {s.get('cited_ids',[])} — {s.get('notes','')}")

            # citations
            cits = result.get("citations", [])
            if cits:
                st.write("Citations:")
                st.write("  ")
                st.write("  ".join([badge_for_citation(c) for c in cits]))

            st.divider()

            colA, colB = st.columns([3,2])
            with colA:
                st.subheader("Top Evidence")
                render_evidence(evidence)
            with colB:
                st.subheader("Consensus Timeline")
                render_consensus(consensus)

        except Exception as e:
            st.error(f"Request failed: {e}")

with tabs[1]:
    st.write("Provide a saved product HTML and a reviews CSV (will be written to `data/raw/`).")
    up_html = st.file_uploader("Product page HTML (.html)", type=["html","htm"], key="html")
    up_csv = st.file_uploader("Reviews CSV (.csv)", type=["csv"], key="csv")

    default_html_name = "pan.html"
    default_csv_name = "pan_reviews.csv"
    name_html = st.text_input("Save as (HTML filename)", value=default_html_name)
    name_csv = st.text_input("Save as (CSV filename)", value=default_csv_name)

    if st.button("Save files & Ingest", type="primary", disabled=not (up_html and up_csv and name_html and name_csv)):
        try:
            # write to data/raw
            raw_dir = os.path.join(os.getcwd(), "data", "raw")
            os.makedirs(raw_dir, exist_ok=True)
            html_path = os.path.join(raw_dir, name_html)
            csv_path  = os.path.join(raw_dir, name_csv)
            with open(html_path, "wb") as f:
                f.write(up_html.read())
            with open(csv_path, "wb") as f:
                f.write(up_csv.read())

            resp = api_ingest(name_html, name_csv)
            st.success(f"Ingested: {resp}")
            # refresh product list
            st.session_state["_products"] = api_get_products()
        except Exception as e:
            st.error(f"Ingest failed: {e}")

with tabs[2]:
    st.subheader("Model Evaluation Dashboard")
    st.caption("Runs local eval with selected models and datasets.")

    eval_models = st.multiselect(
        "Select models",
        options=["llama3.1:8b", "mistral:7b", "qwen2:7b", "phi3:14b", "extractive"],
        default=["llama3.1:8b", "mistral:7b", "qwen2:7b", "phi3:14b"],
    )
    eval_datasets = st.multiselect(
        "Select datasets",
        options=DEFAULT_DATASETS,
        default=DEFAULT_DATASETS,
    )
    k_val = st.slider("k for retrieval metrics", min_value=3, max_value=10, value=5, step=1)
    run_btn = st.button("Start evaluation", type="primary", disabled=not (eval_models and eval_datasets))

    if run_btn:
        try:
            with st.spinner("Running evaluation..."):
                # load data
                data = []
                dataset_sizes = {}
                for ds in eval_datasets:
                    items = _load_dataset(ds)
                    dataset_sizes[os.path.basename(ds)] = len(items)
                    data.extend(items)

                if not data:
                    st.error("No examples loaded.")
                else:
                    # retrieval once
                    r_recall, r_ndcg = [], []
                    hits_cache = []
                    total_answerable = sum(1 for ex in data if ex.get("answerable", True))
                    total_unanswerable = len(data) - total_answerable

                    for ex in data:
                        pid = ex["product_id"]
                        q = ex["question"]
                        answerable = bool(ex.get("answerable", True))
                        gold = ex.get("gold_spans", [])
                        gold_ids = [g["source_id"] for g in gold]
                        gold_rel = _gold_rel_map(gold)

                        retr = HybridRetriever(pid)
                        hits = retr.search(q, k_dense=30, k_out=max(k_val, 10))
                        ranked_ids = [h["source_id"] for h in hits]
                        hits_cache.append((ex, hits, ranked_ids))

                        if answerable and gold_ids:
                            r_recall.append(recall_at_k(gold_ids, ranked_ids, k_val))
                            r_ndcg.append(ndcg_at_k(ranked_ids, gold_rel, k_val))

                    st.markdown("**Datasets:**")
                    st.write(dataset_sizes)
                    st.metric("Recall@k", f"{_avg(r_recall):.3f}", help="Answerable items only")
                    st.metric("nDCG@k", f"{_avg(r_ndcg):.3f}", help="Answerable items only")
                    st.caption(f"Answerable={total_answerable}, Unanswerable={total_unanswerable}")

                    rows = []
                    per_example_rows = []
                    for model in eval_models:
                        sent_csr = []
                        tn_flags, fp_flags = [], []
                        llm_errors = 0
                        for ex, hits, _ranked_ids in hits_cache:
                            q = ex["question"]
                            answerable = bool(ex.get("answerable", True))
                            gold = ex.get("gold_spans", [])
                            gold_ids = [g["source_id"] for g in gold]

                            if model == "extractive":
                                ans = answer_from_passages(q, hits, max_sents=2)
                            else:
                                try:
                                    ans = generate_llm_answer(q, hits, model_name=model)
                                except Exception:
                                    llm_errors += 1
                                    ans = answer_from_passages(q, hits, max_sents=2)

                            atext = ans.get("answer", "")
                            if answerable and gold_ids:
                                sent_csr.append(citation_support_rate(atext, gold_ids))
                            pred_unans = predicted_unanswerable(atext)
                            if not answerable:
                                tn_flags.append(1 if pred_unans else 0)
                            else:
                                fp_flags.append(1 if pred_unans else 0)

                            csr_ex = citation_support_rate(atext, gold_ids) if (answerable and gold_ids) else None
                            confidence = float(ans.get("confidence", 0.0))
                            if answerable:
                                acc = 1.0 if (csr_ex is not None and csr_ex > 0) else 0.0
                            else:
                                acc = 1.0 if pred_unans else 0.0

                            per_example_rows.append({
                                "dataset": ex.get("_dataset", ""),
                                "product_id": ex.get("product_id"),
                                "question": q,
                                "answerable": answerable,
                                "gold_ids": ";".join(gold_ids),
                                "model": model,
                                "confidence": confidence,
                                "pred_unanswerable": bool(pred_unans),
                                "citation_support_rate": csr_ex if csr_ex is not None else None,
                                "accuracy": acc
                            })

                        csr = _avg(sent_csr)
                        tn_rate = _avg(tn_flags) if tn_flags else None
                        fp_rate = _avg(fp_flags) if fp_flags else None

                        rows.append({
                            "Model": model,
                            "Citation support": round(csr, 3),
                            "TN (unans)": f"{sum(tn_flags)}/{len(tn_flags)}" if tn_flags else "n/a",
                            "TN rate": round(tn_rate, 3) if tn_rate is not None else "n/a",
                            "FP (ans)": f"{sum(fp_flags)}/{len(fp_flags)}" if fp_flags else "n/a",
                            "FP rate": round(fp_rate, 3) if fp_rate is not None else "n/a",
                            "LLM errors": llm_errors
                        })

                    df = pd.DataFrame(rows)
                    st.subheader("Model comparison")
                    st.dataframe(df, use_container_width=True)

                    # charts
                    chart_df = df[["Model", "Citation support"]].set_index("Model")
                    st.bar_chart(chart_df)

                    # confidence calibration
                    if per_example_rows:
                        st.subheader("Confidence calibration")
                        cal_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
                        cal_table = []
                        for model in eval_models:
                            recs = [r for r in per_example_rows if r["model"] == model]
                            for i in range(len(cal_bins) - 1):
                                lo, hi = cal_bins[i], cal_bins[i+1]
                                bucket = [r for r in recs if lo <= float(r["confidence"]) < hi]
                                acc = _avg([r["accuracy"] for r in bucket]) if bucket else None
                                cal_table.append({
                                    "Model": model,
                                    "Bin": f"{lo:.1f}-{hi:.1f}",
                                    "n": len(bucket),
                                    "Accuracy": round(acc, 3) if acc is not None else None
                                })
                        cal_df = pd.DataFrame(cal_table)
                        st.dataframe(cal_df, use_container_width=True)

                    # exports
                    if per_example_rows:
                        csv_buf = pd.DataFrame(per_example_rows).to_csv(index=False)
                        json_buf = json.dumps(per_example_rows, ensure_ascii=False, indent=2)
                        st.download_button("Download results CSV", data=csv_buf, file_name="eval_results.csv", mime="text/csv")
                        st.download_button("Download results JSON", data=json_buf, file_name="eval_results.json", mime="application/json")

        except Exception as e:
            st.error(f"Eval failed: {e}")
