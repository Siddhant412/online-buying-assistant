from bs4 import BeautifulSoup
from readability import Document
import pandas as pd
from datetime import datetime, date
from pathlib import Path
import json
from .config import RAW_DIR, PROC_DIR

def parse_product_html(html_path: Path) -> dict:
    raw = html_path.read_text(encoding="utf-8", errors="ignore")
    doc = Document(raw)
    summary_html = doc.summary(html_partial=True)
    soup = BeautifulSoup(summary_html, "lxml")

    title = (soup.find("title").get_text(strip=True)
             if soup.find("title") else html_path.stem)

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

def load_reviews_csv(csv_path: Path) -> list[dict]:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["verified"] = df["verified"].astype(bool)
    df["helpful_votes"] = pd.to_numeric(df["helpful_votes"], errors="coerce").fillna(0).astype(int)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0).astype(int)

    reviews = df.to_dict(orient="records")
    for r in reviews:
        if isinstance(r.get("date"), (datetime, date)):
            r["date"] = r["date"].isoformat()
        elif r.get("date") is not None:
            r["date"] = str(r["date"])
        r["source_type"] = "review"
    return reviews

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
    csv_path = RAW_DIR / reviews_csv_filename
    product = parse_product_html(html_path)
    reviews = load_reviews_csv(csv_path)
    return save_processed(product, reviews)
