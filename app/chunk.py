import re
from typing import List, Dict
from .config import PROC_DIR
import json
from pathlib import Path

def _split_text(text: str, min_tok=80, max_tok=180) -> List[str]:
    # simple sentence splitter, then pack into token-sized chunks
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, cur, cur_len = [], [], 0
    for s in sents:
        if not s: continue
        est = max(1, len(s.split()))  # cheap token proxy
        if cur_len + est > max_tok and cur:
            chunks.append(" ".join(cur))
            cur, cur_len = [], 0
        cur.append(s)
        cur_len += est
    if cur:
        chunks.append(" ".join(cur))
    # ensure min size by merging tiny tails
    out = []
    buf = ""
    for c in chunks:
        if len(c.split()) < min_tok and buf:
            buf = buf + " " + c
        else:
            if buf: out.append(buf)
            buf = c
    if buf: out.append(buf)
    return out

def chunk_processed(json_path: Path) -> list[Dict]:
    blob = json.loads(json_path.read_text(encoding="utf-8"))
    prod = blob["product"]
    reviews = blob["reviews"]

    chunks = []

    # specs chunk (longer)
    if prod.get("specs"):
        spec_text = "\n".join(prod["specs"])
        for i, ch in enumerate(_split_text(spec_text, min_tok=200, max_tok=600)):
            chunks.append({
                "product_id": prod["product_id"],
                "chunk_id": f"S{i}",
                "source_id": f"S{i}",
                "source_type": "spec",
                "text": ch,
                "meta": {"title": prod.get("title")}
            })

    # section paragraphs as smaller chunks
    for i, ch in enumerate(_split_text("\n".join(prod.get("sections", [])), min_tok=120, max_tok=220)):
        chunks.append({
            "product_id": prod["product_id"],
            "chunk_id": f"P{i}",
            "source_id": f"P{i}",
            "source_type": "product_page",
            "text": ch,
            "meta": {"title": prod.get("title")}
        })

    # reviews: keep boundaries intact (one review = one chunk)
    for r in reviews:
        text = ". ".join([str(r.get("title","")).strip(), str(r.get("body","")).strip()]).strip(". ")
        if not text: continue
        chunks.append({
            "product_id": prod["product_id"],
            "chunk_id": r["review_id"],
            "source_id": r["review_id"],
            "source_type": "review",
            "text": text,
            "meta": {
                "rating": int(r.get("rating",0)),
                "date": str(r.get("date","")),
                "verified": bool(r.get("verified",False)),
                "helpful_votes": int(r.get("helpful_votes",0))
            }
        })

    # save
    out_path = PROC_DIR / f"{prod['product_id']}_chunks.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return chunks
