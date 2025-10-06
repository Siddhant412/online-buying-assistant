from __future__ import annotations
from typing import List, Dict, Iterable
import math
import re

_CIT_RE = re.compile(r"\[([A-Za-z0-9_-]+)\]")

def recall_at_k(gold_ids: Iterable[str], ranked_ids: List[str], k: int) -> float:
    """binary recall@k over source_ids."""
    gold = set(gold_ids)
    if not gold:
        return float("nan")
    topk = set(ranked_ids[:k])
    hit = len(gold & topk) > 0
    return 1.0 if hit else 0.0

def dcg_at_k(relevances: List[float], k: int) -> float:
    dcg = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        dcg += (2**rel - 1) / math.log2(i + 1)
    return dcg

def ndcg_at_k(ranked_ids: List[str], gold_rel: Dict[str, float], k: int) -> float:
    """graded nDCG@k using per-id relevance in gold_rel (defaults to 1 if missing)."""
    rels = [float(gold_rel.get(rid, 0.0)) for rid in ranked_ids[:k]]
    dcg = dcg_at_k(rels, k)
    ideal_rels = sorted(gold_rel.values(), reverse=True)
    idcg = dcg_at_k(ideal_rels, min(k, len(ideal_rels)))
    return (dcg / idcg) if idcg > 0 else float("nan")

def extract_sentence_citations(text: str) -> List[List[str]]:
    """returns list of citation-id lists per sentence."""
    sents = re.split(r'(?<=[.!?])\s+', (text or "").strip())
    out = []
    for s in sents:
        if not s.strip():
            continue
        out.append(_CIT_RE.findall(s))
    return out

def citation_support_rate(answer_text: str, gold_ids: Iterable[str]) -> float:
    """
    fraction of sentences where at least one citation matches a gold source_id.
    if no sentences found, returns 0.0.
    """
    gold = set(gold_ids)
    sents = extract_sentence_citations(answer_text)
    if not sents:
        return 0.0
    ok = 0
    for tags in sents:
        ok += 1 if (set(tags) & gold) else 0
    return ok / len(sents)

def predicted_unanswerable(answer_text: str) -> bool:
    t = (answer_text or "").lower()
    return "insufficient evidence" in t or "not enough evidence" in t
