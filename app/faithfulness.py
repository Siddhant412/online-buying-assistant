from __future__ import annotations
from typing import List, Dict, Tuple
import re

STOP = {
    "the","a","an","and","or","but","if","so","of","to","in","on","for",
    "is","are","was","were","be","been","it","this","that","with","as",
    "by","from","at","than","then","their","its","his","her","they","you","i",
    "about","over","after","before","into","out","up","down","off"
}

CIT_RE = re.compile(r"\[([A-Za-z0-9_-]+)\]")
SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

# simple numeric/claim patterns worth verifying explicitly
RE_MONTHS = re.compile(r"\b(\d+)\s*(?:month|months|mo)\b", re.I)
RE_TEMP   = re.compile(r"\b(\d+)\s*°?\s*(?:f|c)\b", re.I)
RE_TIME   = re.compile(r"\b(\d+)\s*(?:day|days|week|weeks|year|years|yr|yrs)\b", re.I)

def _norm_tokens(text: str) -> set:
    toks = re.findall(r"[a-zA-Z0-9°]+", (text or "").lower())
    return {t for t in toks if t not in STOP and len(t) > 1}

def _gather_cited_text(cited_ids: List[str], passages: List[Dict]) -> str:
    blob = []
    allowed = {p["source_id"]: p for p in passages}
    for cid in cited_ids:
        if cid in allowed:
            blob.append(allowed[cid]["text"])
    return "\n".join(blob)

def _numbers_in(text: str, pattern: re.Pattern) -> List[str]:
    return [m.group(0).lower() for m in pattern.finditer(text or "")]

def check_sentence_support(sentence: str, cited_ids: List[str], passages: List[Dict]) -> Dict:
    """
    Returns a dict with:
      - supported: bool
      - coverage: float (0..1 token overlap ratio)
      - missing_numbers: list[str]  (numbers in the sentence not found in cited text)
      - notes: str
    """
    cited_text = _gather_cited_text(cited_ids, passages)
    if not cited_text:
        return {"supported": False, "coverage": 0.0, "missing_numbers": [], "notes": "No cited text available."}

    sent_tokens = _norm_tokens(sentence)
    cited_tokens = _norm_tokens(cited_text)

    # token coverage: fraction of non-stopword sentence tokens that appear in cited text
    overlap = sent_tokens & cited_tokens
    coverage = (len(overlap) / max(1, len(sent_tokens)))

    # explicit number/claim checks
    missing = []
    for pat in (RE_MONTHS, RE_TEMP, RE_TIME):
        nums = _numbers_in(sentence, pat)
        for n in nums:
            if n.lower() not in (cited_text.lower()):
                missing.append(n)

    # Decide support with simple rules:
    # - good coverage or there are no explicit numbers to verify
    # - there is at least 1 citation tag
    supported = bool(cited_ids) and (coverage >= 0.35) and (len(missing) == 0)

    note_bits = [f"overlap={len(overlap)}/{len(sent_tokens)} ({coverage:.2f})"]
    if missing:
        note_bits.append(f"missing_numbers={missing}")
    return {
        "supported": supported,
        "coverage": round(coverage, 2),
        "missing_numbers": missing,
        "notes": "; ".join(note_bits)
    }

def evaluate_answer(answer_text: str, passages: List[Dict]) -> Dict:
    """
    Splits the answer into sentences, extracts [ID] tags per sentence,
    returns per-sentence support judgments.
    """
    if not answer_text:
        return {"sentences": [], "overall_supported_rate": 0.0}

    sentences = [s for s in SENT_SPLIT.split(answer_text.strip()) if s.strip()]
    results = []
    supported_count = 0

    for s in sentences:
        cited_ids = CIT_RE.findall(s)
        r = check_sentence_support(s, cited_ids, passages)
        r.update({"sentence": s, "cited_ids": cited_ids})
        results.append(r)
        if r["supported"]:
            supported_count += 1

    rate = supported_count / max(1, len(results))
    return {"sentences": results, "overall_supported_rate": round(rate, 2)}
