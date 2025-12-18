from bs4 import BeautifulSoup
from readability import Document
import pandas as pd
from datetime import datetime, date
from pathlib import Path
import json
import re
from .config import RAW_DIR, PROC_DIR

_WS_RE = re.compile(r"\s+")

def _clean_title(s: str) -> str:
    return _WS_RE.sub(" ", (s or "").strip())

def parse_product_html(html_path: Path) -> dict:
    raw = html_path.read_text(encoding="utf-8", errors="ignore")
    doc = Document(raw)
    summary_html = doc.summary(html_partial=True)
    soup = BeautifulSoup(summary_html, "lxml")

    title = ""
    try:
        raw_soup = BeautifulSoup(raw, "lxml")
        # Amazon: preferred title element
        t_el = raw_soup.select_one("#productTitle") or raw_soup.select_one("span#productTitle")
        if t_el:
            title = _clean_title(t_el.get_text(" ", strip=True))
        if not title:
            og = raw_soup.select_one("meta[property='og:title']") or raw_soup.select_one("meta[name='og:title']")
            if og and og.get("content"):
                title = _clean_title(str(og.get("content")))
        if not title and raw_soup.title:
            title = _clean_title(raw_soup.title.get_text(" ", strip=True))
    except Exception:
        title = ""

    # readability title fallback
    if not title:
        try:
            title = _clean_title(doc.short_title() or doc.title() or "")
        except Exception:
            title = ""
    if not title:
        title = html_path.stem

    headings = [h.get_text(" ", strip=True) for h in soup.select("h1,h2,h3")]
    paras = [p.get_text(" ", strip=True) for p in soup.select("p,li") if p.get_text(strip=True)]

    specs = []
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in row.find_all(["th","td"])]
            if len(cells) == 2:
                specs.append(f"{cells[0]}: {cells[1]}")
            elif cells:
                specs.append(" | ".join(cells))

    return {
        "product_id": html_path.stem,
        "title": title,
        "headings": headings,
        "specs": specs,
        "sections": paras
    }

def _normalize_reviews(reviews: list[dict]) -> list[dict]:
    out: list[dict] = []
    for r in reviews:
        rr = dict(r)

        # date normalization
        d = rr.get("date")
        if isinstance(d, (datetime, date)):
            rr["date"] = d.isoformat()
        elif d is not None:
            rr["date"] = str(d)

        # type normalization
        rr["verified"] = bool(rr.get("verified", False))
        try:
            rr["helpful_votes"] = int(rr.get("helpful_votes", 0) or 0)
        except Exception:
            rr["helpful_votes"] = 0
        try:
            rr["rating"] = int(float(rr.get("rating", 0) or 0))
        except Exception:
            rr["rating"] = 0

        rr["source_type"] = "review"
        out.append(rr)
    return out

def load_reviews_csv(csv_path: Path) -> list[dict]:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["verified"] = df["verified"].astype(bool)
    df["helpful_votes"] = pd.to_numeric(df["helpful_votes"], errors="coerce").fillna(0).astype(int)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0).astype(int)

    reviews = df.to_dict(orient="records")
    return _normalize_reviews(reviews)

def load_reviews_jsonl(jsonl_path: Path) -> list[dict]:
    reviews: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            reviews.append(json.loads(line))
    return _normalize_reviews(reviews)

def save_processed(product_blob: dict, reviews: list[dict]) -> Path:
    out = {
        "product": product_blob,
        "reviews": reviews,
        "created_at": datetime.utcnow().isoformat()
    }
    out_path = PROC_DIR / f"{product_blob['product_id']}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path

def ingest_one(product_html_filename: str, reviews_csv_filename: str) -> Path:
    html_path = RAW_DIR / product_html_filename
    reviews_path = RAW_DIR / reviews_csv_filename
    product = parse_product_html(html_path)
    if reviews_path.suffix.lower() == ".jsonl":
        reviews = load_reviews_jsonl(reviews_path)
    else:
        reviews = load_reviews_csv(reviews_path)
    return save_processed(product, reviews)
