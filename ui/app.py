import os, json, textwrap, sys, re, subprocess, tempfile
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

from eval.metrics import recall_at_k, ndcg_at_k, citation_support_rate, predicted_unanswerable

load_dotenv()
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
AMAZON_INGEST_TIMEOUT_SEC = int(os.getenv("AMAZON_INGEST_TIMEOUT_SEC", "600"))
DEFAULT_DATASETS = [
    "eval/groundtruth_dataset.jsonl",
    "eval/groundtruth_dataset_additional.jsonl"
]

_ASIN_RE = re.compile(r"(?i)(?:dp/|product-reviews/|asin=)?([A-Z0-9]{10})")

def extract_asin(text: str) -> str | None:
    m = _ASIN_RE.search((text or "").strip())
    return m.group(1).upper() if m else None

def api_get_products():
    try:
        r = requests.get(f"{API_URL}/products", timeout=15)
        r.raise_for_status()
        return r.json().get("products", [])
    except Exception:
        return []

def api_get_products_meta():
    """
    Returns list of {"product_id": ..., "title": ...}. Falls back to /products.
    """
    try:
        r = requests.get(f"{API_URL}/products_meta", timeout=15)
        r.raise_for_status()
        items = r.json().get("products", [])
        out = []
        for it in items:
            pid = (it or {}).get("product_id")
            title = (it or {}).get("title")
            if pid:
                out.append({"product_id": pid, "title": title or pid})
        return out
    except Exception:
        return [{"product_id": pid, "title": pid} for pid in (api_get_products() or [])]

def api_ingest(product_html: str, reviews_csv: str):
    r = requests.post(f"{API_URL}/ingest", json={"product_html": product_html, "reviews_csv": reviews_csv}, timeout=120)
    r.raise_for_status()
    return r.json()

def api_ingest_amazon(query_or_url: str, max_pages: int = 3, sort: str = "recent"):
    payload = {"query_or_url": query_or_url, "max_pages": max_pages, "sort": sort}
    r = requests.post(f"{API_URL}/ingest_amazon", json=payload, timeout=AMAZON_INGEST_TIMEOUT_SEC)
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
        hv = meta.get("helpful_votes", 0)
        return f"`[{sid}]`  ⭐{rating}  {date}  👍{hv}"
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

def render_qa_response(res: dict):
    result = res.get("result", {})
    evidence = res.get("evidence", [])
    consensus = res.get("consensus", {})
    faith = res.get("faithfulness", {})

    st.subheader("Answer")
    st.write(result.get("answer", "(no answer)"))

    meta_cols = st.columns(3)
    with meta_cols[0]:
        st.metric("Engine", result.get("engine", "?"))
    with meta_cols[1]:
        st.metric("Confidence", f"{float(result.get('confidence', 0.0)):.2f}")
    with meta_cols[2]:
        st.metric("Evidence items", str(len(evidence)))

    cits = result.get("citations", [])
    if cits:
        st.caption("Citations")
        st.write("  ".join([badge_for_citation(c) for c in cits]))

    if faith:
        with st.expander("Faithfulness details", expanded=False):
            st.caption(f"Supported sentences: {faith.get('overall_supported_rate',0):.2f}")
            for i, s in enumerate(faith.get("sentences", []), 1):
                status = "✅ supported" if s["supported"] else "⚠️ needs evidence"
                st.write(f"{i}. {status} — cites {s.get('cited_ids',[])} — {s.get('notes','')}")

    st.divider()
    colA, colB = st.columns([3, 2])
    with colA:
        st.subheader("Top Evidence")
        render_evidence(evidence)
    with colB:
        st.subheader("Consensus Timeline")
        render_consensus(consensus)

def _default_select_index(options: list[str], preferred: str | None) -> int:
    if preferred and preferred in options:
        return options.index(preferred)
    return 0

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

def _arrow_safe_eval_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Streamlit uses Arrow for dataframe rendering; mixed-type object columns can crash.
    Normalize common eval columns into Arrow-friendly dtypes.
    """
    if df is None or df.empty:
        return df
    out = df.copy()

    # Replace empty strings with nulls to avoid Arrow trying (and failing) to coerce types.
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].replace({"": None})

    for col in ("confidence", "citation_support_rate", "accuracy"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "answerable" in out.columns:
        try:
            out["answerable"] = out["answerable"].astype("boolean")
        except Exception:
            pass

    return out

def _run_eval_subprocess(datasets: list[str], models: list[str], k: int) -> tuple[int, str, list[dict]]:
    """
    Runs eval/run_eval.py in a subprocess so native-library crashes won't take down Streamlit.
    Returns (exit_code, combined_output, per_example_rows_json).
    """
    root = Path(__file__).resolve().parents[1]
    script = root / "eval" / "run_eval.py"

    # Prefer the repo's venv python if Streamlit is launched outside the venv.
    py_candidates = [
        root / ".venv" / "bin" / "python",
        root / "venv" / "bin" / "python",
    ]
    py = next((p for p in py_candidates if p.exists()), Path(sys.executable))

    with tempfile.NamedTemporaryFile(prefix="eval_results_", suffix=".json", delete=False) as tmp:
        out_json = tmp.name

    cmd = [str(py), "-X", "faulthandler", str(script), "--k", str(k), "--export_json", out_json]
    for ds in datasets:
        cmd.extend(["--dataset", ds])
    if models:
        cmd.extend(["--models", ",".join(models)])

    env = os.environ.copy()
    # Make eval more stable across machines by default (do not override explicit user env).
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("RERANK_DEVICE", "cpu")
    env.setdefault("EMB_DEVICE", "cpu")
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    env.setdefault("PYTHONFAULTHANDLER", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    header = f"[eval] python={py}\n[eval] cmd={' '.join(cmd)}\n[eval] exit_code={proc.returncode}"
    combined = header + ("\n\n" + (proc.stdout or "") if proc.stdout else "") + ("\n" + proc.stderr if proc.stderr else "")

    rows: list[dict] = []
    try:
        if Path(out_json).exists():
            rows = json.loads(Path(out_json).read_text(encoding="utf-8"))
    except Exception:
        rows = []
    finally:
        try:
            Path(out_json).unlink(missing_ok=True)
        except Exception:
            pass

    return proc.returncode, combined.strip(), rows

st.set_page_config(page_title="Product QA Autopilot", layout="wide")
st.title("🛒 Product Buying Assistant")

tabs = st.tabs(["Amazon", "Ingest", "Eval"])
# preload product list
if "_products" not in st.session_state:
    try:
        st.session_state["_products"] = api_get_products()
    except Exception as e:
        st.session_state["_products"] = []
        st.sidebar.error(f"Could not reach API: {e}")

MODEL_CHOICES = ["llama3.1:8b", "mistral:7b", "qwen2:7b", "extractive"]

with tabs[0]:
    st.subheader("Amazon Product")

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("### 1) Get product data")
        with st.form("amz_scrape_form", clear_on_submit=False):
            query = st.text_input(
                "Keyword / Amazon URL / ASIN",
                value=st.session_state.get("amz_query", ""),
                key="amz_query_input",
            )
            max_pages = st.slider(
                "Max review pages",
                1,
                10,
                int(st.session_state.get("amz_max_pages", 3)),
                key="amz_max_pages_slider",
            )
            sort = st.selectbox("Review sort", options=["recent", "helpful"], index=0, key="amz_sort")

            run_scrape = st.form_submit_button("Ingest from Amazon", type="primary")

        if run_scrape:
            if not (query or "").strip():
                st.warning("Enter a keyword, Amazon URL, or ASIN.")
            else:
                try:
                    st.session_state["amz_query"] = query
                    st.session_state["amz_max_pages"] = max_pages
                    with st.spinner("Scraping and ingesting..."):
                        resp = api_ingest_amazon(query, max_pages=max_pages, sort=sort)
                    st.success(f"Indexed {resp.get('asin')} with {resp.get('reviews')} reviews.")
                    if resp.get("warning"):
                        st.warning(resp["warning"])
                    st.session_state["_products"] = api_get_products()
                    st.session_state["amz_product_id"] = resp.get("product_id")
                    st.session_state["amz_asin"] = resp.get("asin")
                    st.session_state["amz_url"] = resp.get("url")
                    st.session_state["amz_reviews"] = resp.get("reviews")
                    st.session_state["amz_title"] = resp.get("title") or resp.get("product_id")
                except Exception as e:
                    st.error(f"Amazon ingest failed: {e}")


        # Summary card
        pid = st.session_state.get("amz_product_id")
        if pid:
            st.markdown("### Current Amazon product")
            # Prefer the title returned from ingest, else fall back to ID.
            title = st.session_state.get("amz_title") or pid
            st.write(title)
            st.caption(f"Product ID: `{pid}`")
            if st.session_state.get("amz_url"):
                url = st.session_state.get("amz_url")
                st.write(f"Source URL: {url}")
                try:
                    st.link_button("Open on Amazon", url)
                except Exception:
                    pass
            if st.session_state.get("amz_reviews") is not None:
                st.write(f"Reviews indexed: {st.session_state.get('amz_reviews')}")

    with right:
        st.markdown("### 2) Ask questions")

        products_meta = api_get_products_meta()
        title_map = {p["product_id"]: p.get("title") or p["product_id"] for p in products_meta}
        pid_options = [p["product_id"] for p in products_meta] or ["(none)"]
        preferred_pid = st.session_state.get("amz_product_id")

        def _fmt_pid(pid: str) -> str:
            return title_map.get(pid, pid)

        pid = st.selectbox(
            "Indexed product",
            options=pid_options,
            index=_default_select_index(pid_options, preferred_pid) if pid_options != ["(none)"] else 0,
            help="Select the product to query (defaults to the most recently ingested Amazon product).",
            key="amz_pid_select",
            format_func=_fmt_pid,
        )
        if pid != "(none)":
            st.caption(f"Product ID: `{pid}`")
        model_choice = st.selectbox("Model", options=MODEL_CHOICES, index=0, key="amz_model_select")

        with st.form("amz_ask_form", clear_on_submit=False):
            q = st.text_input("Question", value=st.session_state.get("amz_question", ""), key="amz_question_input")

            ask = st.form_submit_button("Ask", type="primary")

        if ask:
            if pid == "(none)":
                st.warning("Index a product first, then select it here.")
            elif not (q or "").strip():
                st.warning("Enter a question.")
            else:
                try:
                    st.session_state["amz_question"] = q
                    chosen_model = None if model_choice == "extractive" else model_choice
                    with st.spinner("Thinking..."):
                        res = api_ask(pid, q, model=chosen_model)
                    st.session_state["amz_last_answer"] = res
                except Exception as e:
                    st.error(f"Request failed: {e}")

        last = st.session_state.get("amz_last_answer")
        if last:
            render_qa_response(last)

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
        options=["llama3.1:8b", "mistral:7b", "qwen2:7b", "extractive"],
        default=["llama3.1:8b", "mistral:7b", "qwen2:7b"],
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
                code, output, per_rows = _run_eval_subprocess(eval_datasets, eval_models, k_val)
            if output:
                with st.expander("Eval logs", expanded=(code != 0)):
                    st.code(output, language="text")
            if code != 0:
                st.error("Eval failed. See logs above.")
            else:
                st.success("Eval completed.")

            if not per_rows:
                st.warning("No per-example results were returned (export_json empty).")
            else:
                df = _arrow_safe_eval_df(pd.DataFrame(per_rows))
                if not df.empty:
                    st.subheader("Per-example results")
                    st.dataframe(df, width="stretch")

                    # Lightweight summaries if expected columns exist
                    if {"model", "citation_support_rate", "accuracy"}.issubset(df.columns):
                        st.subheader("Summary")
                        summ = (
                            df.groupby("model", dropna=False)
                            .agg(
                                n=("accuracy", "count"),
                                avg_accuracy=("accuracy", "mean"),
                                avg_citation_support=("citation_support_rate", "mean"),
                            )
                            .reset_index()
                        )
                        st.dataframe(_arrow_safe_eval_df(summ), width="stretch")

                    # exports
                    st.download_button(
                        "Download results JSON",
                        data=json.dumps(per_rows, ensure_ascii=False, indent=2),
                        file_name="eval_results.json",
                        mime="application/json",
                    )
        except Exception as e:
            st.error(f"Eval failed: {e}")
