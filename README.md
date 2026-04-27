# Online Buying Assistant

[Google Drive folder (report, video demo)](https://drive.google.com/drive/folders/1RQfFxSdwmV5KtkuROgQ014Ix9i_RDrnO?usp=sharing)

Evidence-grounded Product QnA over product pages + reviews using a hybrid RAG pipeline (BM25 + dense FAISS + cross-encoder reranking) with a local LLM (Ollama) and a Streamlit UI.

## Production Transformation

The roadmap for turning this course MVP into a production-grade project is tracked in [`PRODUCTION_TRANSFORMATION_PLAN.md`](PRODUCTION_TRANSFORMATION_PLAN.md). Treat that file as the source of truth for transformation work.

## Quickstart

### 1) Install

```bash
uv sync --extra rag --extra scrape --extra ui --extra dev
```

This project targets Python 3.11 for the productionized environment. If you use `pip` instead of `uv`, install with a Python 3.11 virtual environment and `python -m pip install -r requirements.txt`.

### 2) Configure

Create `.env` in the project root:

```bash
# Storage and jobs
DATABASE_URL=postgresql+psycopg://oba:oba@localhost:5432/oba
REDIS_URL=redis://localhost:6379/0
API_KEY=change-me

# Backend URL used by Streamlit
API_URL=http://127.0.0.1:8000

# Ollama (optional, for LLM answers)
OLLAMA_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.1:8b

# Optional DB retrieval controls
ENABLE_EMBEDDING_WRITES=0
ENABLE_POSTGRES_NATIVE_RETRIEVAL=1
ENABLE_SEMANTIC_RETRIEVAL=1
ENABLE_RERANKING=0
RETRIEVAL_CHUNK_LIMIT=1000

# LLM provider
LLM_PROVIDER=ollama
OPENAI_BASE_URL=
OPENAI_API_KEY=
OPENAI_MODEL=

# (Optional) evaluation default dataset
EVAL_DATASET=eval/groundtruth_dataset.jsonl
```

### 3) Run the backend (FastAPI)

```bash
make run-api
```

### 4) Run the production frontend (Next.js)

```bash
make frontend-install
make run-frontend
```

The frontend targets Node.js 24+. Open `http://localhost:3000`. The Next.js app proxies API calls to FastAPI and injects `API_KEY` server-side for protected write endpoints.

### 5) Run the legacy UI (Streamlit)

```bash
make run-ui
```

## Local Services and Migrations

Start the production-style local services:

```bash
docker compose up --build
```

Services:
- FastAPI: `http://localhost:8000`
- Next.js frontend: `http://localhost:3000`
- Postgres + pgvector: `localhost:5432`
- Redis: `localhost:6379`

Apply database migrations:

```bash
make migrate
```

Run the background worker locally:

```bash
make run-worker
```

Create a DB-backed ingestion job for raw files under `data/raw/`:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/ingestion-jobs \
  -H "x-api-key: change-me" \
  -H "Content-Type: application/json" \
  -d '{"product_html":"pan.html","reviews_file":"pan_reviews.csv","enqueue":true}'
```

Poll job status:

```bash
curl http://127.0.0.1:8000/api/v1/ingestion-jobs/<job_id>
```

Read the DB-backed product catalog after a job succeeds:

```bash
curl http://127.0.0.1:8000/api/v1/products
curl http://127.0.0.1:8000/api/v1/products/<product_id>
curl http://127.0.0.1:8000/api/v1/products/<product_id>/chunks
curl http://127.0.0.1:8000/api/v1/products/<product_id>/reviews
```

Ask over the DB-backed product corpus:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"product_id":"<product_id>","question":"How long does the coating last?","model":"extractive"}'
```

`/api/v1/ask` uses the DB-backed retriever, persists query logs and citation rows, and includes retrieval/model diagnostics. On Postgres it uses native full-text search plus pgvector similarity when embeddings exist. Embedding writes are disabled by default to keep fresh local setup lightweight; set `ENABLE_EMBEDDING_WRITES=1` after installing the `rag` extra to populate pgvector embeddings during ingestion.

Create a DB-backed eval dataset:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/eval-datasets \
  -H "x-api-key: change-me" \
  -H "Content-Type: application/json" \
  -d '{"name":"groundtruth","source_path":"eval/groundtruth_dataset.jsonl","replace":true}'
```

Run a CI-friendly extractive eval:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/eval-runs \
  -H "x-api-key: change-me" \
  -H "Content-Type: application/json" \
  -d '{"dataset_name":"groundtruth","model":"extractive","k":5}'
```

Write endpoints under `/api/v1` require `x-api-key` or `Authorization: Bearer <API_KEY>`.

Legacy MVP file-index endpoints remain available during the migration at `/ingest`, `/products`, `/products_meta`, `/ask`, and `/ingest_amazon`.

## LLM Setup (Ollama)

```bash
ollama serve
ollama pull llama3.1:8b
```

Then ensure `.env` has `OLLAMA_URL` and `OLLAMA_MODEL`.

To use an OpenAI-compatible chat API instead:

```bash
LLM_PROVIDER=openai-compatible
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=<your-key>
OPENAI_MODEL=<model-name>
```

## Amazon Ingest (Playwright)

The app supports `POST /ingest_amazon` (used by the Amazon tab in the UI). Amazon may block automation; scraping reliability depends on the network/account status.

Recommended `.env` settings:

```bash
SCRAPE_HEADLESS=0
SCRAPE_PERSISTENT=1
SCRAPE_INTERACTIVE_LOGIN_TIMEOUT_SEC=300
SCRAPE_DEBUG=1
```

If installed via `requirements.txt`, you still need browser binaries:

```bash
python -m playwright install chromium
```

## Evaluation (Reproduce Results)

Gold datasets live in:
- `eval/groundtruth_dataset.jsonl`
- `eval/groundtruth_dataset_additional.jsonl`

### Run eval

Runs retrieval once per example, then evaluates answer models:

```bash
python eval/run_eval.py \
  --dataset eval/groundtruth_dataset.jsonl \
  --dataset eval/groundtruth_dataset_additional.jsonl \
  --k 5 \
  --models llama3.1:8b,mistral:7b,qwen2:7b
```

### Convenience script (rebuild indexes + eval)

This re-ingests sample products under `data/raw/`, rebuilds indexes, then runs eval:

```bash
python scripts/rebuild_and_eval.py \
  --datasets eval/groundtruth_dataset.jsonl eval/groundtruth_dataset_additional.jsonl \
  --models llama3.1:8b,mistral:7b,qwen2:7b \
  --k 5
```
