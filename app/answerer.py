from typing import List, Dict
import re
from statistics import mean

_DURATION_RE = re.compile(r'(\d+)\s*(?:month|months|mo)\b', re.I)

def _extract_months(text: str) -> List[int]:
    return [int(m.group(1)) for m in _DURATION_RE.finditer(text or "")]

def answer_from_passages(question: str, passages: List[Dict], max_sents: int = 2) -> Dict:
    if not passages:
        return {"answer": "Insufficient evidence.", "citations": [], "confidence": 0.0}

    reviews = [p for p in passages if p["source_type"] == "review"][:3]
    non_reviews = [p for p in passages if p["source_type"] != "review"][:1]
    chosen = reviews + non_reviews
    if not chosen:
        return {"answer": "Insufficient evidence.", "citations": [], "confidence": 0.0}

    months = []
    for p in reviews:
        months.extend(_extract_months(p["text"]))

    citations = [{"source_id": p["source_id"], "source_type": p["source_type"], "meta": p.get("meta", {})} for p in chosen]

    if len(months) >= 2 and (max(months) != min(months)):
        ans = f"Reports vary: about {min(months)}–{max(months)} months based on user reviews."
        conf = 0.55
    elif len(months) >= 1:
        avg = round(mean(months))
        ans = f"Roughly {avg} months based on user reviews."
        conf = 0.6
    else:
        texts = []
        for p in chosen[:max_sents]:
            first = p["text"].strip()
            if ". " in first:
                first = first.split(". ", 1)[1].strip()
            texts.append(first.split(".")[0])
        ans = " ".join([t for t in texts if t]) or "Insufficient evidence."
        conf = 0.4

    return {"answer": ans, "citations": citations, "confidence": round(conf, 2)}
