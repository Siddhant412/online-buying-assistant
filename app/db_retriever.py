from __future__ import annotations

import math
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from .db.models import Chunk, Product
from .db.repositories import ProductRepository
from .providers import (
    LexicalRerankerProvider,
    get_embedding_provider,
    get_reranker_provider,
    lexical_relevance,
)
from .settings import get_settings


def _metadata_float(metadata: dict[str, Any], key: str) -> float:
    try:
        return float(metadata.get(key) or 0)
    except (TypeError, ValueError):
        return 0.0


def _review_prior(source_type: str, metadata: dict[str, Any]) -> float:
    if source_type != "review":
        return 0.0
    helpful = _metadata_float(metadata, "helpful_votes")
    verified = 1.0 if metadata.get("verified") else 0.0
    rating = _metadata_float(metadata, "rating")
    return 0.001 * helpful + 0.2 * verified + 0.02 * rating


def _dot(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    return float(sum(a * b for a, b in zip(left, right, strict=True)))


def _normalize(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    if max_score <= 0:
        return {key: 0.0 for key in scores}
    return {key: value / max_score for key, value in scores.items()}


def _postgres_json_number(key: str) -> str:
    return (
        f"CASE WHEN c.metadata->>'{key}' ~ '^-?[0-9]+(\\.[0-9]+)?$' "
        f"THEN (c.metadata->>'{key}')::double precision ELSE 0.0 END"
    )


class DatabaseHybridRetriever:
    def __init__(self, session: Session, product_id: str) -> None:
        self.session = session
        self.settings = get_settings()
        self.product = self._load_product(product_id)
        self.diagnostics: dict[str, Any] = {}

    def _load_product(self, product_id: str) -> Product:
        product = ProductRepository(self.session).get_by_public_id(product_id)
        if product is None:
            raise ValueError(f"Product not found: {product_id}")
        return product

    def _dialect_name(self) -> str:
        bind = self.session.get_bind()
        return bind.dialect.name if bind is not None else "unknown"

    def _product_has_embeddings(self) -> bool:
        if self._dialect_name() != "postgresql":
            return False
        exists_sql = text(
            """
            SELECT EXISTS (
                SELECT 1
                FROM chunks
                WHERE product_id = CAST(:product_id AS uuid)
                  AND embedding IS NOT NULL
            )
            """
        )
        return bool(self.session.execute(exists_sql, {"product_id": str(self.product.id)}).scalar())

    def _query_embedding(self, query: str, *, require_stored_embeddings: bool = False) -> list[float] | None:
        if not getattr(self.settings, "enable_semantic_retrieval", True):
            self.diagnostics["semantic_retrieval"] = "disabled"
            return None

        if require_stored_embeddings and not self._product_has_embeddings():
            self.diagnostics["semantic_retrieval"] = "skipped_no_chunk_embeddings"
            return None

        try:
            vector = get_embedding_provider().embed_texts([query])[0]
        except Exception as exc:
            self.diagnostics["semantic_retrieval"] = f"failed: {exc}"
            return None

        self.diagnostics["semantic_retrieval"] = "enabled"
        return vector

    def _dense_scores(self, query: str, chunks: list[Chunk]) -> dict[int, float]:
        embedded = {idx: chunk.embedding for idx, chunk in enumerate(chunks) if chunk.embedding}
        if not embedded:
            self.diagnostics["semantic_retrieval"] = "skipped_no_chunk_embeddings"
            return {}

        query_vector = self._query_embedding(query)
        if query_vector is None:
            return {}

        scores = {
            idx: _dot(query_vector, list(vector))
            for idx, vector in embedded.items()
            if len(query_vector) == len(vector)
        }
        if not scores and self.diagnostics.get("semantic_retrieval") == "enabled":
            self.diagnostics["semantic_retrieval"] = "skipped_dim_mismatch"
        return _normalize(scores)

    def _rerank_hits(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not candidates:
            return []

        passages = [candidate["text"] for candidate in candidates]
        try:
            reranker = get_reranker_provider()
            rerank_scores = reranker.rerank(query, passages)
            self.diagnostics["reranker"] = reranker.__class__.__name__
        except Exception as exc:
            reranker = LexicalRerankerProvider()
            rerank_scores = reranker.rerank(query, passages)
            self.diagnostics["reranker"] = f"lexical_fallback: {exc}"

        ranked = []
        for candidate, rerank_score in zip(candidates, rerank_scores, strict=False):
            base_score = float(candidate["score"])
            candidate["score"] = 0.85 * float(rerank_score) + 0.15 * base_score
            candidate["score_components"]["rerank_score"] = float(rerank_score)
            ranked.append(candidate)
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked

    def _apply_output_policy(self, ranked: list[dict[str, Any]], k_out: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen_source_ids: set[str] = set()
        non_review_budget = 1
        for hit in ranked:
            if hit["source_id"] in seen_source_ids:
                continue
            if hit["source_type"] != "review":
                if non_review_budget <= 0:
                    continue
                non_review_budget -= 1
            seen_source_ids.add(hit["source_id"])
            out.append(hit)
            if len(out) >= k_out:
                break
        return out

    def _search_postgres(
        self,
        query: str,
        *,
        k_dense: int,
        k_out: int,
        pool_size: int,
    ) -> list[dict[str, Any]]:
        query_vector = self._query_embedding(query, require_stored_embeddings=True)
        candidate_limit = max(pool_size, k_dense, k_out)
        sparse_limit = max(candidate_limit, 20)
        dense_limit = max(candidate_limit, 20)

        helpful_expr = _postgres_json_number("helpful_votes")
        rating_expr = _postgres_json_number("rating")
        prior_expr = (
            "CASE WHEN c.source_type = 'review' THEN "
            f"(0.001 * {helpful_expr}) + "
            "(0.2 * CASE WHEN lower(coalesce(c.metadata->>'verified', 'false')) "
            "IN ('true', '1', 'yes') THEN 1.0 ELSE 0.0 END) + "
            f"(0.02 * {rating_expr}) "
            "ELSE 0.0 END"
        )

        if query_vector is not None:
            dense_cte = """
            dense AS (
                SELECT
                    c.id,
                    0.0::double precision AS sparse_score,
                    c.embedding <=> CAST(:query_embedding AS vector) AS dense_distance
                FROM chunks c
                WHERE c.product_id = CAST(:product_id AS uuid)
                  AND c.embedding IS NOT NULL
                ORDER BY c.embedding <=> CAST(:query_embedding AS vector)
                LIMIT :dense_limit
            ),
            unioned AS (
                SELECT id, sparse_score, NULL::double precision AS dense_distance FROM sparse
                UNION ALL
                SELECT id, sparse_score, dense_distance FROM dense
            ),
            """
            dense_score_expr = (
                "CASE WHEN cs.dense_distance IS NULL THEN 0.0 "
                "ELSE greatest(0.0, 1.0 - cs.dense_distance) END"
            )
        else:
            dense_cte = """
            unioned AS (
                SELECT id, sparse_score, NULL::double precision AS dense_distance FROM sparse
            ),
            """
            dense_score_expr = "0.0"

        retrieval_sql = text(
            f"""
            WITH q AS (
                SELECT websearch_to_tsquery('english', :query) AS tsq
            ),
            sparse AS (
                SELECT
                    c.id,
                    ts_rank_cd(to_tsvector('english', c.text), q.tsq)::double precision AS sparse_score,
                    NULL::double precision AS dense_distance
                FROM chunks c
                CROSS JOIN q
                WHERE c.product_id = CAST(:product_id AS uuid)
                  AND to_tsvector('english', c.text) @@ q.tsq
                ORDER BY sparse_score DESC
                LIMIT :sparse_limit
            ),
            {dense_cte}
            candidate_scores AS (
                SELECT
                    id,
                    max(sparse_score) AS sparse_score,
                    min(dense_distance) AS dense_distance
                FROM unioned
                GROUP BY id
            )
            SELECT
                c.id,
                c.source_id,
                c.source_type,
                c.text,
                c.metadata,
                coalesce(cs.sparse_score, 0.0) AS sparse_score,
                {dense_score_expr} AS dense_score,
                {prior_expr} AS prior_score,
                (
                    0.45 * least(coalesce(cs.sparse_score, 0.0), 1.0)
                    + 0.45 * ({dense_score_expr})
                    + ({prior_expr})
                ) AS base_score
            FROM candidate_scores cs
            JOIN chunks c ON c.id = cs.id
            ORDER BY base_score DESC
            LIMIT :candidate_limit
            """
        )
        params: dict[str, Any] = {
            "query": query,
            "product_id": str(self.product.id),
            "sparse_limit": sparse_limit,
            "dense_limit": dense_limit,
            "candidate_limit": candidate_limit,
        }
        if query_vector is not None:
            params["query_embedding"] = query_vector

        rows = self.session.execute(retrieval_sql, params).mappings().all()
        candidates = [
            {
                "chunk_id": str(row["id"]),
                "score": float(row["base_score"]),
                "text": row["text"],
                "source_id": row["source_id"],
                "source_type": row["source_type"],
                "meta": dict(row["metadata"] or {}),
                "score_components": {
                    "sparse_score": float(row["sparse_score"] or 0.0),
                    "dense_score": float(row["dense_score"] or 0.0),
                    "prior_score": float(row["prior_score"] or 0.0),
                    "base_score": float(row["base_score"] or 0.0),
                },
            }
            for row in rows
        ]
        ranked = self._rerank_hits(query, candidates)
        out = self._apply_output_policy(ranked, k_out)
        self.diagnostics.update(
            {
                "retrieval_backend": "postgres_native",
                "chunks_considered": len(candidates),
                "candidates": len(candidates),
                "returned": len(out),
                "sparse_limit": sparse_limit,
                "dense_limit": dense_limit if query_vector is not None else 0,
            }
        )
        return out

    def _search_python(
        self,
        query: str,
        *,
        k_dense: int,
        k_out: int,
        pool_size: int,
    ) -> list[dict[str, Any]]:
        repo = ProductRepository(self.session)
        chunks = list(repo.list_retrieval_chunks(self.product.id, limit=self.settings.retrieval_chunk_limit))
        if not chunks:
            self.diagnostics.update({"retrieval_backend": "python", "chunks_considered": 0})
            return []

        sparse_scores = _normalize(
            {idx: lexical_relevance(query, chunk.text) for idx, chunk in enumerate(chunks)}
        )
        dense_scores = self._dense_scores(query, chunks)

        candidates: list[dict[str, Any]] = []
        for idx, chunk in enumerate(chunks):
            metadata = chunk.metadata_ or {}
            sparse = sparse_scores.get(idx, 0.0)
            dense = dense_scores.get(idx, 0.0)
            if sparse <= 0 and dense <= 0:
                continue
            weight = 0.5 if dense_scores else 1.0
            prior = _review_prior(chunk.source_type, metadata)
            score = (weight * sparse) + ((1.0 - weight) * dense) + prior
            if not math.isfinite(score):
                continue
            candidates.append(
                {
                    "chunk_id": str(chunk.id),
                    "score": float(score),
                    "text": chunk.text,
                    "source_id": chunk.source_id,
                    "source_type": chunk.source_type,
                    "meta": metadata,
                    "score_components": {
                        "sparse_score": sparse,
                        "dense_score": dense,
                        "prior_score": prior,
                        "base_score": score,
                    },
                }
            )

        candidates.sort(key=lambda item: item["score"], reverse=True)
        pool = candidates[: min(pool_size, max(k_dense, len(candidates)))]
        ranked = self._rerank_hits(query, pool)
        out = self._apply_output_policy(ranked, k_out)

        self.diagnostics.update(
            {
                "retrieval_backend": "python",
                "chunks_considered": len(chunks),
                "candidates": len(candidates),
                "returned": len(out),
            }
        )
        return out

    def search(self, query: str, k_dense: int = 30, k_out: int = 10, pool_size: int = 40) -> list[dict]:
        if (
            self._dialect_name() == "postgresql"
            and getattr(self.settings, "enable_postgres_native_retrieval", True)
        ):
            try:
                return self._search_postgres(query, k_dense=k_dense, k_out=k_out, pool_size=pool_size)
            except Exception as exc:
                self.diagnostics["postgres_native_error"] = str(exc)

        return self._search_python(query, k_dense=k_dense, k_out=k_out, pool_size=pool_size)
