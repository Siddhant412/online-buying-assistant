from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from .answerer import answer_from_passages
from .db.repositories import EvalDatasetRepository, EvalRunRepository
from .db_retriever import DatabaseHybridRetriever
from .llm_answerer import generate_llm_answer


def load_eval_examples_jsonl(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def _gold_rel_map(gold_spans: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for span in gold_spans:
        source_id = str(span["source_id"])
        relevance = float(span.get("relevance", 1.0))
        out[source_id] = max(out.get(source_id, 0.0), relevance)
    return out


def _avg(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return round(sum(finite) / len(finite), 4) if finite else None


def _recall_at_k(gold_ids: list[str], ranked_ids: list[str], k: int) -> float:
    if not gold_ids:
        return float("nan")
    return 1.0 if set(gold_ids) & set(ranked_ids[:k]) else 0.0


def _dcg_at_k(relevances: list[float], k: int) -> float:
    dcg = 0.0
    for index, relevance in enumerate(relevances[:k], start=1):
        dcg += (2**relevance - 1) / math.log2(index + 1)
    return dcg


def _ndcg_at_k(ranked_ids: list[str], gold_rel: dict[str, float], k: int) -> float:
    relevance = [float(gold_rel.get(source_id, 0.0)) for source_id in ranked_ids[:k]]
    dcg = _dcg_at_k(relevance, k)
    ideal = sorted(gold_rel.values(), reverse=True)
    idcg = _dcg_at_k(ideal, min(k, len(ideal)))
    return dcg / idcg if idcg > 0 else float("nan")


def _citation_support_rate(answer_text: str, gold_ids: list[str]) -> float:
    import re

    sentences = [s for s in re.split(r"(?<=[.!?])\s+", (answer_text or "").strip()) if s.strip()]
    if not sentences:
        return 0.0
    gold = set(gold_ids)
    supported = 0
    for sentence in sentences:
        citations = set(re.findall(r"\[([A-Za-z0-9_-]+)\]", sentence))
        supported += 1 if citations & gold else 0
    return supported / len(sentences)


def _predicted_unanswerable(answer_text: str) -> bool:
    text = (answer_text or "").lower()
    return "insufficient evidence" in text or "not enough evidence" in text


def _answer(question: str, hits: list[dict[str, Any]], model_name: str) -> tuple[dict[str, Any], bool]:
    if model_name == "extractive":
        return answer_from_passages(question, hits, max_sents=2), False
    try:
        return generate_llm_answer(question, hits, model_name=model_name), False
    except Exception:
        return answer_from_passages(question, hits, max_sents=2), True


def run_eval(
    session: Session,
    *,
    dataset_id: UUID,
    model_name: str = "extractive",
    k: int = 5,
    max_examples: int | None = None,
) -> dict[str, Any]:
    dataset_repo = EvalDatasetRepository(session)
    dataset = dataset_repo.get(dataset_id)
    if dataset is None:
        raise ValueError(f"Eval dataset not found: {dataset_id}")

    limit = max_examples or 10_000
    examples = list(dataset_repo.list_examples(dataset.id, limit=limit))

    recall_values: list[float] = []
    ndcg_values: list[float] = []
    csr_values: list[float] = []
    true_negative_flags: list[float] = []
    false_positive_flags: list[float] = []
    per_example: list[dict[str, Any]] = []
    llm_fallbacks = 0
    skipped = 0

    for example in examples:
        gold_spans = list(example.gold_spans or [])
        gold_ids = [str(span["source_id"]) for span in gold_spans]
        answerable = bool(example.answerable)

        try:
            retriever = DatabaseHybridRetriever(session, example.product_external_id)
            hits = retriever.search(example.question, k_dense=30, k_out=max(k, 10))
        except Exception as exc:
            skipped += 1
            per_example.append(
                {
                    "example_id": str(example.id),
                    "product_id": example.product_external_id,
                    "question": example.question,
                    "status": "skipped",
                    "error": str(exc),
                }
            )
            continue

        ranked_ids = [hit["source_id"] for hit in hits]
        gold_rel = _gold_rel_map(gold_spans)
        if answerable and gold_ids:
            recall_values.append(_recall_at_k(gold_ids, ranked_ids, k))
            ndcg_values.append(_ndcg_at_k(ranked_ids, gold_rel, k))

        answer, used_fallback = _answer(example.question, hits, model_name)
        llm_fallbacks += 1 if used_fallback else 0
        answer_text = str(answer.get("answer") or "")

        citation_support = None
        if answerable and gold_ids:
            citation_support = _citation_support_rate(answer_text, gold_ids)
            csr_values.append(citation_support)

        predicted_unanswerable = _predicted_unanswerable(answer_text)
        if not answerable:
            true_negative_flags.append(1.0 if predicted_unanswerable else 0.0)
        else:
            false_positive_flags.append(1.0 if predicted_unanswerable else 0.0)

        per_example.append(
            {
                "example_id": str(example.id),
                "product_id": example.product_external_id,
                "question": example.question,
                "status": "evaluated",
                "answerable": answerable,
                "gold_ids": gold_ids,
                "ranked_ids": ranked_ids[:k],
                "answer": answer_text,
                "confidence": float(answer.get("confidence", 0.0)),
                "citation_support_rate": citation_support,
                "predicted_unanswerable": predicted_unanswerable,
                "retrieval": retriever.diagnostics,
            }
        )

    return {
        "dataset": {"id": str(dataset.id), "name": dataset.name},
        "model": model_name,
        "k": k,
        "total_examples": len(examples),
        "evaluated_examples": len(examples) - skipped,
        "skipped_examples": skipped,
        "retrieval": {
            f"recall_at_{k}": _avg(recall_values),
            f"ndcg_at_{k}": _avg(ndcg_values),
            "answerable_with_gold": len(recall_values),
        },
        "answers": {
            "citation_support_rate": _avg(csr_values),
            "true_negative_rate": _avg(true_negative_flags),
            "false_positive_rate": _avg(false_positive_flags),
            "llm_fallbacks": llm_fallbacks,
        },
        "examples": per_example,
    }


def create_eval_run(
    session: Session,
    *,
    dataset_id: UUID,
    model_name: str = "extractive",
    k: int = 5,
    max_examples: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    run_repo = EvalRunRepository(session)
    run = run_repo.create(dataset_id=dataset_id, model_name=model_name)
    try:
        metrics = run_eval(
            session,
            dataset_id=dataset_id,
            model_name=model_name,
            k=k,
            max_examples=max_examples,
        )
        run_repo.mark_succeeded(run, metrics)
        return run, metrics
    except Exception as exc:
        run_repo.mark_failed(run, str(exc))
        raise
