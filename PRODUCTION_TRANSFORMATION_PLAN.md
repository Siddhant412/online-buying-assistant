# Production Transformation Plan: Online Buying Assistant

## Governance

This document is the source of truth for transforming this repository from a course MVP into a production-style AI application.

Implementation should follow this plan unless a change is genuinely needed. If an implementation detail needs to change, the change must be proposed to the project owner first, and this document may only be updated after approval.

## Summary

Transform the course MVP into production-style AI application over 3-4 weeks. The target is a polished local/deployable system that shows senior engineering judgment: clean architecture, durable storage, async ingestion, robust RAG evaluation, provider-agnostic LLMs, observability, tests, Docker, CI, and a modern web UI.

Default stack: FastAPI, Postgres + pgvector, Redis + RQ, Next.js, Docker Compose, Pydantic Settings, Alembic, pytest, ruff, mypy, OpenTelemetry, and provider abstraction for Ollama plus OpenAI-compatible APIs.

## Current Implementation Status

Last updated: April 26, 2026

Completed:
- Production packaging and developer tooling: `pyproject.toml`, `uv.lock`, Makefile, ruff, mypy, pytest, pre-commit, Dockerfile, Docker Compose, and CI workflow.
- Runtime foundation: Pydantic settings, structured logging, request context middleware, health/readiness endpoints, CORS configuration, and structured error handling.
- Production data layer: SQLAlchemy models, repository layer, Alembic environment, and initial Postgres + pgvector migration for products, sections, reviews, chunks, ingestion jobs, query logs, citations, eval datasets, examples, and runs.
- Ingestion job foundation: DB-backed job creation/list/read endpoints, Redis/RQ enqueue hook, worker entrypoint, raw-file ingestion task, normalized product persistence, and corpus replacement semantics.
- API versioning foundation: explicit `/api/v1` router with DB-backed product catalog endpoints and DB-backed `/api/v1/ask`.
- Retrieval migration foundation: DB chunk retriever, Postgres-native full-text/vector hybrid retrieval, SQLite/Python fallback, optional reranker provider, retrieval index migration, and persisted query/citation logs.
- Provider foundation: local sentence-transformers embedding provider, configurable embedding writes, helper for persisting missing chunk embeddings, Ollama chat provider, and OpenAI-compatible chat provider.
- MVP compatibility fixes: lazy heavy imports for lightweight API startup, extractive model routing, date normalization, and empty-ingest validation.
- Verification baseline: unit/API/integration tests for answerer behavior, LLM provider citation enforcement, health/readiness, DB models, migrations, ingestion jobs, DB-backed product catalog reads, DB-backed ask/query logging, and temporal logic.

In progress:
- Expanding evaluation storage into API workflows and CI-friendly eval runs.
- Hardening operational controls around write auth, rate-limit hooks, and request/input limits.

Next planned slice:
- Add `/api/v1/eval-datasets` and `/api/v1/eval-runs` endpoints backed by the existing eval tables.
- Persist small fixture eval runs from CI without requiring heavyweight local model downloads.
- Add API key enforcement for write endpoints and documented operational limits.

## Key Changes

- Restructure backend into production layers: API routers, domain models, services, repositories, providers, background jobs, and shared config/errors/logging.
- Replace ad hoc `requirements.txt` setup with `pyproject.toml`, pinned lockfile, `ruff`, `mypy`, `pytest`, `pre-commit`, and Makefile/task commands.
- Add Docker Compose for local production simulation: `api`, `worker`, `postgres` with pgvector, `redis`, `frontend`, and optional `ollama`.
- Move runtime state from ignored local files into Postgres:
  - products, reviews, product page sections, chunks, embeddings, ingestion jobs, query logs, citations, eval runs.
  - use Alembic migrations for schema changes.
- Replace FAISS/BM25 file indexes with hybrid retrieval using Postgres full-text search + pgvector semantic search + cross-encoder reranking.
- Keep Amazon scraping as an optional best-effort connector, not the reliable production path.
  - Reliable ingestion paths become file upload, JSONL/CSV import, and API-based product/review import.
  - Scraper outputs normalized artifacts and records blocked/partial states instead of failing ambiguously.
- Add provider abstraction:
  - `LLMProvider`: Ollama and OpenAI-compatible chat providers.
  - `EmbeddingProvider`: local sentence-transformers first, extensible hosted provider.
  - `RerankerProvider`: local cross-encoder with configurable disable/fallback mode.
- Version API under `/api/v1`.
  - Core endpoints: health/readiness, products, ingestion jobs, ask, eval runs, datasets.
  - Use stable request/response schemas with request IDs, structured errors, citation objects, and confidence/faithfulness metadata.
- Add background ingestion pipeline:
  - API creates an ingestion job.
  - RQ worker parses, normalizes, chunks, embeds, stores, and marks status.
  - UI polls job status.
- Build a modern Next.js frontend as the primary product surface:
  - Product ingestion dashboard.
  - Product library.
  - Ask experience with answer, cited evidence, confidence, retrieval diagnostics, and temporal review trends.
  - Eval dashboard with retrieval and answer metrics.
  - Keep Streamlit only as optional internal/legacy tooling.
- Add observability:
  - structured JSON logs with request/job IDs.
  - OpenTelemetry instrumentation for API, DB, worker, and model calls.
  - `/healthz` and `/readyz`.
  - lightweight local dashboards/log inspection in docs.
- Add security and operational hygiene:
  - `.env.example`, secret-free config, CORS allowlist, API key auth for write endpoints, input size limits, timeouts, and rate-limit hooks.
  - no raw secret/config in git.
- Upgrade documentation:
  - architecture diagram, local quickstart, Docker quickstart, API examples, eval reproduction, design decisions, and production limitations.

## Implementation Roadmap

- Week 1: Foundation
  - Add project packaging/tooling, config system, logging, error model, tests skeleton, Docker Compose, CI.
  - Introduce Postgres schema and Alembic migrations.
  - Preserve current behavior behind compatibility scripts while new services are added.
- Week 2: Data + RAG Core
  - Implement ingestion jobs, normalized persistence, chunking service, embedding storage, hybrid retrieval, reranking, provider abstraction, and `/api/v1/ask`.
  - Fix known MVP issues: fresh clone setup, extractive model routing, date normalization, empty-ingest handling.
- Week 3: UI + Evaluation
  - Build Next.js app for ingest, product browsing, asking, evidence inspection, temporal charts, and eval dashboard.
  - Expand evaluation runner to store runs, compare models, export reports, and run in CI on small fixtures.
- Week 4: Production Polish
  - Add auth/rate-limit hooks, observability, docs, seeded demo dataset, final acceptance tests, cleanup legacy code paths, and portfolio-ready README.

## Test Plan

- Unit tests:
  - HTML parsing, CSV/JSONL review normalization, date parsing, chunking, citation extraction, faithfulness checks, temporal analysis.
- Integration tests:
  - ingestion job creates product/reviews/chunks/embeddings.
  - `/api/v1/ask` returns answer, evidence, citations, confidence, and structured diagnostics.
  - provider fallback from LLM failure to extractive answer works.
- Retrieval/eval tests:
  - small seeded fixture verifies recall@k, nDCG@k, citation support, and unanswerable detection.
  - CI runs fast eval with tiny embeddings or mocked providers.
- API tests:
  - health/readiness, validation errors, missing product, empty ingest, malformed upload, rate-limit/auth behavior.
- UI acceptance tests:
  - ingest product, poll job, ask question, inspect evidence, view eval result.
- Operational tests:
  - `docker compose up` starts all services.
  - fresh clone can seed demo data and answer a question without manual index files.

## Assumptions

- Target is production-grade quality, not a fully hosted enterprise SaaS.
- Local Docker Compose is the main deployment target; cloud deployment docs can be added but not implemented first.
- Primary UI will be Next.js; Streamlit becomes optional/internal.
- Postgres + pgvector replaces local FAISS artifacts as the production retrieval store.
- Amazon scraping remains optional and best-effort; reliable ingestion is upload/API import.
- LLM architecture supports Ollama locally and OpenAI-compatible hosted providers through one interface.
