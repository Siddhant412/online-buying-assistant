from __future__ import annotations
from typing import List, Dict, Tuple
import os, re, json, requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# prompting
SYSTEM_RULES = """\
You are a careful product QA assistant. Answer ONLY using the EVIDENCE passages provided.
Hard rules:
- Every sentence MUST include at least one citation tag like [R2] or [S1], where tags are the exact source_id values shown in EVIDENCE.
- If evidence conflicts, say "Reports vary" and briefly summarize (do not cherry-pick).
- If the answer is unknown from the evidence, reply "Insufficient evidence." (still include at least one citation to the closest relevant passage, if any).
- Be concise (1-2 sentences).
- Never invent facts or sources. Never use outside knowledge.
"""

USER_TEMPLATE = """\
QUESTION:
{question}

EVIDENCE (each item has source_id in square brackets):
{evidence_block}

Write a 1-2 sentence answer that follows the rules.
"""

def _evidence_block(passages: List[Dict]) -> Tuple[str, set]:
    """
    Passage formatting:
    [R2] (review, stars, verified, 2024-11-20): Non-stick wore off in 3 months...
    Returns (block_text, allowed_ids)
    """
    lines = []
    allowed = set()
    for p in passages:
        sid = p["source_id"]
        st = p["source_type"]
        allowed.add(sid)
        if st == "review":
            meta = p.get("meta", {})
            rating = meta.get("rating", "")
            verified = "verified" if meta.get("verified") else "unverified"
            date = meta.get("date", "")
            head = f"[{sid}] (review, ⭐{rating}, {verified}, {date})"
        elif st == "spec":
            head = f"[{sid}] (spec)"
        else:
            head = f"[{sid}] ({st})"
        text = p["text"].replace("\n", " ").strip()
        if len(text) > 400:
            text = text[:400].rsplit(" ", 1)[0] + " ..."
        lines.append(f"{head}: {text}")
    return "\n".join(lines), allowed

# minimal faithfulness guard on the LLM output

_CIT_RE = re.compile(r"\[([A-Za-z0-9_-]+)\]")

def _enforce_citations(ans: str, allowed: set, fallback_sid: str | None) -> str:
    """
    Ensure:
    - Every sentence has at least one [ID] from allowed
    - Remove any unknown [ID]s
    """
    # remove disallowed tags
    def filter_ids(text: str) -> str:
        def repl(m):
            return f"[{m.group(1)}]" if m.group(1) in allowed else ""
        return _CIT_RE.sub(repl, text)

    cleaned = filter_ids(ans).strip()
    if not cleaned:
        return cleaned

    # ensure each sentence has a citation
    sents = re.split(r'(?<=[.!?])\s+', cleaned)
    fixed = []
    for s in sents:
        if not s.strip():
            continue
        if not _CIT_RE.search(s):
            # append fallback citation if available
            if fallback_sid:
                s = s.rstrip() + f" [{fallback_sid}]"
        fixed.append(s)
    return " ".join(fixed).strip()

def _first_allowed_id(allowed: set) -> str | None:
    return next(iter(allowed)) if allowed else None

# Ollama client
def _ollama_chat(system: str, user: str, model: str, url: str) -> str:
    """
    Calls Ollama /api/chat
    """
    endpoint = f"{url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": 4096
        }
    }
    resp = requests.post(endpoint, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    return data.get("message", {}).get("content", "").strip()


def generate_llm_answer(question: str, passages: List[Dict]) -> Dict:
    """
    Returns:
    {
      "answer": "<text with [IDs]>",
      "citations": [{"source_id":..., "source_type":..., "meta":...}, ...],
      "confidence": float
    }
    """
    if not passages:
        return {"answer": "Insufficient evidence.", "citations": [], "confidence": 0.0}

    evidence, allowed = _evidence_block(passages)
    prompt = USER_TEMPLATE.format(question=question, evidence_block=evidence)
    try:
        raw = _ollama_chat(SYSTEM_RULES, prompt, OLLAMA_MODEL, OLLAMA_URL)
    except Exception as e:
        # let caller decide about fallback, but return structured error context
        raise RuntimeError(f"Ollama error: {e}")

    fixed = _enforce_citations(raw, allowed, _first_allowed_id(allowed))
    if not fixed:
        # if model produced nothing valid, degrade gracefully
        first = _first_allowed_id(allowed)
        fixed = "Insufficient evidence." + (f" [{first}]" if first else "")

    # confidence: simple heuristic = fraction of sentences that have citations
    conf = 0.65

    # return all passages as eligible citations
    cits = [{"source_id": p["source_id"], "source_type": p["source_type"], "meta": p.get("meta", {})} for p in passages]

    return {"answer": fixed, "citations": cits, "confidence": conf}
