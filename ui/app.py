import os, json, textwrap
from datetime import datetime
import pandas as pd
import streamlit as st
import requests
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

def api_get_products():
    r = requests.get(f"{API_URL}/products", timeout=15)
    r.raise_for_status()
    return r.json().get("products", [])

def api_ingest(product_html: str, reviews_csv: str):
    r = requests.post(f"{API_URL}/ingest", json={"product_html": product_html, "reviews_csv": reviews_csv}, timeout=120)
    r.raise_for_status()
    return r.json()

def api_ask(pid: str, q: str):
    r = requests.post(f"{API_URL}/ask", json={"product_id": pid, "question": q}, timeout=60)
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

tabs = st.tabs(["Ask", "Ingest"])
# preload product list
if "_products" not in st.session_state:
    try:
        st.session_state["_products"] = api_get_products()
    except Exception as e:
        st.session_state["_products"] = []
        st.sidebar.error(f"Could not reach API: {e}")

with tabs[0]:
    col1, col2 = st.columns([2, 5])
    with col1:
        products = st.session_state.get("_products", [])
        pid = st.selectbox("Product ID", options=(products or ["(none)"]), index=0)
        manual_pid = st.text_input("…or type a product_id", value="" if products else "")
        chosen_pid = manual_pid.strip() or (pid if pid != "(none)" else "")
    with col2:
        q = st.text_input("Your question", value="How long does the non-stick coating last?")

    ask_btn = st.button("Ask", type="primary", disabled=not (chosen_pid and q))
    if ask_btn:
        try:
            res = api_ask(chosen_pid, q)
            result = res.get("result", {})
            evidence = res.get("evidence", [])
            consensus = res.get("consensus", {})

            # answer block
            st.subheader("Answer")
            st.write(result.get("answer","(no answer)"))
            st.caption(f"Confidence: {result.get('confidence', 0):.2f}")

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
