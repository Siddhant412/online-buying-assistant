import os, json
from typing import List, Dict
import streamlit as st
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
DEFAULT_DATASET = os.getenv("EVAL_DATASET", "eval/groundtruth_dataset.jsonl")

st.set_page_config(page_title="Gold Set Annotator", layout="wide")
st.title("📝 Gold Set Annotator (source_id labels)")

def api_get_products() -> List[str]:
    r = requests.get(f"{API_URL}/products", timeout=15)
    r.raise_for_status()
    return r.json().get("products", [])

def api_ask(pid: str, q: str) -> Dict:
    r = requests.post(f"{API_URL}/ask", json={"product_id": pid, "question": q}, timeout=60)
    r.raise_for_status()
    return r.json()

def append_jsonl(path: str, obj: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def read_tail(path: str, n: int = 20) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_json(path, lines=True).tail(n)

# sidebar
with st.sidebar:
    st.subheader("Settings")
    API_URL = st.text_input("FastAPI URL", API_URL)
    dataset_path = st.text_input("Dataset file (.jsonl)", DEFAULT_DATASET)
    if st.button("Refresh product list"):
        st.session_state["_products"] = api_get_products()
    st.divider()
    st.caption("Run API in another terminal:\n`uvicorn app.api:app --reload`")

if "_products" not in st.session_state:
    try:
        st.session_state["_products"] = api_get_products()
    except Exception as e:
        st.session_state["_products"] = []
        st.sidebar.error(f"API unreachable: {e}")

# main UI
colL, colR = st.columns([2, 5])

with colL:
    products = st.session_state.get("_products", [])
    pid = st.selectbox("Product ID", options=(products or ["(none)"]))
    manual_pid = st.text_input("…or type product_id", "")
    chosen_pid = manual_pid.strip() or (pid if pid != "(none)" else "")
    q = st.text_input("Question", "How long does the non-stick coating last?")
    fetch = st.button("Retrieve evidence", type="primary", disabled=not (chosen_pid and q))

with colR:
    if fetch:
        try:
            res = api_ask(chosen_pid, q)
            st.session_state["last_pid"] = chosen_pid
            st.session_state["last_q"] = q
            st.session_state["evidence"] = res.get("evidence", [])
        except Exception as e:
            st.error(f"Request failed: {e}")

    ev = st.session_state.get("evidence", [])
    if ev:
        st.subheader("Top Evidence (select supporting sources)")
        st.caption("Tip: relevance 2 = primary; 1 = secondary; 0 = not supporting")
        selections = []
        for i, p in enumerate(ev, 1):
            with st.expander(f"{i}. [{p['source_id']}] {p['source_type']} — score {p['score']:.3f}", expanded=(i <= 3)):
                meta = p.get("meta", {})
                if p["source_type"] == "review":
                    st.write(f"⭐ {meta.get('rating','?')} | "
                             f"{'✅ verified' if meta.get('verified') else '⚪︎ unverified'} | "
                             f"date: {meta.get('date','?')} | 👍 {meta.get('helpful_votes',0)}")
                st.write(p["text"])
                rel = st.selectbox(
                    f"Relevance for [{p['source_id']}]", options=[0,1,2], index=0, key=f"rel_{p['source_id']}"
                )
                if rel > 0:
                    selections.append({"source_id": p["source_id"], "relevance": rel})

        st.divider()
        cols = st.columns(3)
        with cols[0]:
            answerable = st.toggle("Answerable?", value=bool(selections), help="Off = should be 'insufficient evidence'")
        with cols[1]:
            st.write("")
            st.write("")
            if st.button("Append to dataset", type="primary"):
                try:
                    item = {
                        "product_id": st.session_state.get("last_pid", chosen_pid),
                        "question": st.session_state.get("last_q", q),
                        "answerable": bool(answerable),
                        "gold_spans": selections if answerable else []
                    }
                    append_jsonl(dataset_path, item)
                    st.success(f"Appended 1 line to {dataset_path}")
                except Exception as e:
                    st.error(f"Write failed: {e}")
        with cols[2]:
            st.write("")
            st.write("")
            if st.button("Clear selections"):
                for p in ev:
                    st.session_state[f"rel_{p['source_id']}"] = 0
                st.experimental_rerun()

    st.subheader("Dataset preview (tail)")
    df_tail = read_tail(dataset_path, n=20)
    if not df_tail.empty:
        st.dataframe(df_tail, use_container_width=True)
    else:
        st.caption("No dataset yet — label a few items and append.")
