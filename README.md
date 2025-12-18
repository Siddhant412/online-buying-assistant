# Online Buying Assistant

Evidence-grounded Product QnA over product pages + reviews using a hybrid RAG pipeline (BM25 + dense FAISS + cross-encoder reranking) with a local LLM (Ollama) and a Streamlit UI.

## Quickstart

### 1) Install

```bash
python -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

### 2) Configure

Create `.env` in the project root:

```bash
# Backend URL used by Streamlit
API_URL=http://127.0.0.1:8000

# Ollama (optional, for LLM answers)
OLLAMA_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.1:8b

# (Optional) evaluation default dataset
EVAL_DATASET=eval/groundtruth_dataset.jsonl
```

### 3) Run the backend (FastAPI)

```bash
uvicorn app.api:app --reload --port 8000
```

### 4) Run the UI (Streamlit)

```bash
streamlit run ui/app.py
```

## LLM Setup (Ollama)

```bash
ollama serve
ollama pull llama3.1:8b
```

Then ensure `.env` has `OLLAMA_URL` and `OLLAMA_MODEL`.

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
