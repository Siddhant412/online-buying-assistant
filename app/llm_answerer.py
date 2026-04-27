from __future__ import annotations

import re
from typing import Any

from .providers import get_llm_provider

SYSTEM_RULES = """\
You are a careful product QA assistant. Answer ONLY using the EVIDENCE passages provided.
Hard rules:
- Every sentence MUST include at least one citation tag like [R2] or [S1], where tags are the exact source_id values shown in EVIDENCE.
- If evidence conflicts, say "Reports vary" and briefly summarize without cherry-picking.
- If the answer is unknown from the evidence, reply "Insufficient evidence." and cite the closest relevant passage if one exists.
- Be concise: 1-2 sentences.
- Never invent facts or sources. Never use outside knowledge.
"""

USER_TEMPLATE = """\
QUESTION:
{question}

EVIDENCE (each item has source_id in square brackets):
{evidence_block}

Write a 1-2 sentence answer that follows the rules.
"""

_CIT_RE = re.compile(r"\[([A-Za-z0-9_-]+)\]")


def _evidence_block(passages: list[dict[str, Any]]) -> tuple[str, set[str]]:
    lines = []
    allowed = set()
    for passage in passages:
        source_id = passage["source_id"]
        source_type = passage["source_type"]
        allowed.add(source_id)
        if source_type == "review":
            metadata = passage.get("meta", {})
            rating = metadata.get("rating", "")
            verified = "verified" if metadata.get("verified") else "unverified"
            date = metadata.get("date", "")
            header = f"[{source_id}] (review, rating={rating}, {verified}, {date})"
        elif source_type == "spec":
            header = f"[{source_id}] (spec)"
        else:
            header = f"[{source_id}] ({source_type})"
        text = passage["text"].replace("\n", " ").strip()
        if len(text) > 400:
            text = text[:400].rsplit(" ", 1)[0] + " ..."
        lines.append(f"{header}: {text}")
    return "\n".join(lines), allowed


def _enforce_citations(answer: str, allowed: set[str], fallback_source_id: str | None) -> str:
    def replace_unknown(match: re.Match[str]) -> str:
        source_id = match.group(1)
        return f"[{source_id}]" if source_id in allowed else ""

    cleaned = _CIT_RE.sub(replace_unknown, answer).strip()
    if not cleaned:
        return cleaned

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    fixed = []
    for sentence in sentences:
        if not sentence.strip():
            continue
        if not _CIT_RE.search(sentence) and fallback_source_id:
            sentence = sentence.rstrip() + f" [{fallback_source_id}]"
        fixed.append(sentence)
    return " ".join(fixed).strip()


def _first_allowed_id(allowed: set[str]) -> str | None:
    return next(iter(allowed)) if allowed else None


def generate_llm_answer(
    question: str,
    passages: list[dict[str, Any]],
    model_name: str | None = None,
) -> dict[str, Any]:
    if not passages:
        return {"answer": "Insufficient evidence.", "citations": [], "confidence": 0.0}

    evidence, allowed = _evidence_block(passages)
    prompt = USER_TEMPLATE.format(question=question, evidence_block=evidence)
    try:
        provider_response = get_llm_provider().chat(SYSTEM_RULES, prompt, model_name=model_name)
    except Exception as exc:
        raise RuntimeError(f"LLM provider error: {exc}") from exc

    fixed = _enforce_citations(
        provider_response.content,
        allowed,
        _first_allowed_id(allowed),
    )
    if not fixed:
        first = _first_allowed_id(allowed)
        fixed = "Insufficient evidence." + (f" [{first}]" if first else "")

    citations = [
        {
            "source_id": passage["source_id"],
            "source_type": passage["source_type"],
            "meta": passage.get("meta", {}),
        }
        for passage in passages
    ]

    return {
        "answer": fixed,
        "citations": citations,
        "confidence": 0.65,
        "provider": provider_response.provider,
        "model": provider_response.model,
        "provider_latency_ms": provider_response.latency_ms,
    }
