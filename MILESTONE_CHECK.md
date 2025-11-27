# Online Buying Assistant (RAG QA System)

## 1. Achievements so far

- Implemented an end‑to‑end ingestion -> indexing -> retrieval -> answering pipeline.
  - Ingestion of product HTML + reviews CSV into a normalized JSON format.
  - Text chunking for specs, product page content, and reviews.
  - Hybrid BM25 + dense + cross‑encoder retrieval and scoring.
  - Answer generation using either a local LLM (Ollama) or an extractive fallback.
  - Temporal sentiment analysis over reviews to detect changes in user experience over time.
  - Faithfulness checking of answers against cited evidence.

- Exposed a clean API surface for the system via FastAPI:
  - `POST /ingest` – ingest and index a new product.
  - `GET /products` – list currently indexed products.
  - `POST /ask` – run retrieval + answer generation + temporal & faithfulness analysis.

- Built a user‑facing UI for interactive QA and ingestion:
  - Streamlit UI with “Ask” and “Ingest” tabs, including evidence viewers and consensus timeline charts.

- Built an evaluation + annotation toolkit:
  - Streamlit UI for creating a gold dataset of questions and supporting source_ids.
  - Ground truth dataset JSONL with labeled examples for two products.
  - Evaluation script to compute retrieval and answer‑quality metrics.

- Set up configuration and environment:
  - Centralized paths, models, and index directories.
  - `.env` configuration for Ollama, API URL, evaluation dataset, and temporal review windows.

## 2. Module status overview

**Core pipeline**

- `app/ingest.py` – **Fully functional**
  - Parses HTML product pages and review CSVs.
  - Saves normalized JSON blobs in `data/processed`.

- `app/chunk.py` – **Fully functional**
  - Creates semantically meaningful chunks for specs, content, and reviews.
  - Persists chunked data as JSONL for downstream indexing.

- `app/indexer.py` – **Fully functional**
  - Builds BM25, dense embedding, and FAISS indexes for each product.
  - Already used to index sample products (`pan`, `airfryer`) under `data/index`.

- `app/retriever.py` (HybridRetriever) – **Fully functional**
  - Combines sparse BM25, dense FAISS, and cross‑encoder re‑ranking.
  - Adds priors favoring more trustworthy reviews (verified, helpful, better‑rated).

- `app/answerer.py` (extractive baseline) – **Fully functional**
  - Produces answers based purely on selected passages, focusing on durations in months.
  - Serves as a robust fallback when the LLM is unavailable.

- `app/llm_answerer.py` (LLM‑based answerer via Ollama) – **Functional, depends on local model setup**
  - Requires a running Ollama server and the configured model.
  - Includes guardrails for citation enforcement and “insufficient evidence” behavior.

- `app/temporal.py` (temporal review analysis) – **Fully functional**
  - Buckets reviews into configurable time windows and summarizes sentiment trends.

- `app/faithfulness.py` (faithfulness checker) – **Fully functional (heuristic)**
  - Token‑overlap and numeric‑consistency based scoring of answer sentences relative to cited text.

**APIs & UIs**

- `app/api.py` (FastAPI app) – **Fully functional**
  - Provides the ingestion and question‑answering endpoints used by all frontends and tools.

- `ui/app.py` (main QA UI) – **Fully functional**
  - Fully wired into the FastAPI backend for asking questions and ingesting new products.
  - Displays evidence, temporal consensus, citations, and faithfulness diagnostics.

- `eval/annotator.py` (gold annotator UI) – **Fully functional**
  - Allows interactive labeling of supporting passages for given questions.
  - Appends new examples to the evaluation dataset.

- `eval/run_eval.py` + `eval/metrics.py` (evaluation pipeline) – **Functional; baseline metrics implemented**
  - Computes retrieval metrics (Recall@k, nDCG@k).
  - Computes answer‑quality metrics (citation support rate, unanswerable detection).
  - Can be run with or without the LLM (using `--disable_llm`).

## 3. Baseline modules: functionality and evaluation

### 3.1 Identified baseline modules

- Ingestion + chunking: `app/ingest.py`, `app/chunk.py`
- Indexing + hybrid retrieval (BM25 + dense + cross‑encoder): `app/indexer.py`, `app/retriever.py`
- Extractive answerer: `app/answerer.py`
- Core FastAPI endpoints: `app/api.py`
- Evaluation scripts and metrics: `eval/run_eval.py`, `eval/metrics.py`
- Annotation UI and QA UI: `eval/annotator.py`, `ui/app.py`

The LLM answerer (`app/llm_answerer.py`), faithfulness checker (`app/faithfulness.py`), and temporal trend module (`app/temporal.py`) extend this baseline with more advanced reasoning, calibration, and diagnostics.

### 3.2 Baseline functionality

- **Ingestion & chunking**
  - Robustly parses arbitrary product HTML into structured fields (title, specs, headings, sections).
  - Normalizes diverse review CSVs into a consistent schema with dates, ratings, verification flags, and helpfulness counts.
  - Produces granular chunks that preserve review boundaries and product metadata for better retrieval control.

- **Hybrid retrieval**
  - Uses BM25 for lexical coverage and dense embeddings for semantic similarity.
  - FAISS enables scalable dense nearest‑neighbor search over product chunks.
  - A cross‑encoder reranker refines a candidate pool of passages, improving precision for small k.
  - A simple prior biases toward more trustworthy reviews (verified, highly rated, helpful).

- **Extractive answers**
  - For duration‑type questions (e.g., coating lifetime), identifies numeric mentions (months) in top reviews and summarizes them into ranges or averages.
  - When numeric patterns are absent, surfaces representative sentences directly from evidence, while still signaling “Insufficient evidence.” when needed.

- **APIs and UIs**
  - The FastAPI endpoints provide a clear contract for programmatic or UI‑driven access.
  - The QA UI surfaces not only the answer, but also:
    - Supporting passages.
    - A temporal breakdown of review sentiment.
    - Citations and confidence estimates.
  - The annotator UI allows quick curation of ground truth for evaluation tasks.

### 3.3 Baseline evaluation

- **Retrieval quality**
  - Evaluated using Recall@k and nDCG@k over a small labeled dataset (`eval/groundtruth_dataset.jsonl`).
  - The dataset currently includes examples for two products (`pan`, `airfryer`) with both answerable and unanswerable queries.

- **Answer quality**
  - Citation support rate: fraction of answer sentences where at least one citation matches a gold supporting passage.
  - Unanswerable detection: whether the system correctly outputs “insufficient evidence” when the gold label is unanswerable.
  - For baseline runs, the script can be invoked with `--disable_llm` to evaluate the extractive answerer alone.

- **Screenshots**
![App Screenshot](https://drive.google.com/uc?export=view&id=1IWyC138ieLjvpC8VvtIPh4IKujvH5jbR)


![App Screenshot](https://drive.google.com/uc?export=view&id=1K2UEi9SWcFdTeXBdOT9UhVwTrsEzMfF0)


## 4. References and sources used

Code and design are grounded in the following libraries, tools, and concepts:

- **Retrieval and embeddings**
  - `sentence-transformers` – sentence and passage embedding models, e.g. `"intfloat/e5-base-v2"`.
  - FAISS – dense vector index for similarity search.
  - `rank_bm25` – BM25 implementation for lexical retrieval.
  - CrossEncoder models for reranking candidate passages.

- **Web and APIs**
  - FastAPI – backend framework for REST endpoints.
  - Streamlit – rapid UIs for QA and annotation.
  - `requests` – HTTP client for UI ↔ API communication.

- **LLM serving**
  - Ollama – local LLM server accessed via `/api/chat`, configured via `.env`.

- **Data processing**
  - `pandas` – CSV loading and data manipulation for reviews.
  - `readability-lxml` + `BeautifulSoup` – extracting main content and structured information from product HTML.

- **General Python & tooling**
  - Python standard library modules for JSON, datetime handling, path management, statistics, and regex (`app/`, `eval/` modules throughout).


## 5. Challenges and plans to tackle

### 5.1 Challenges

- **Local model and resource requirements**
  - The system depends on local models (SentenceTransformer, cross‑encoder, Ollama LLM) that can be memory and compute intensive especially on CPU‑only machines.
  - Cross‑encoder reranking adds latency for each query particularly as the candidate pool grows.

- **Limited evaluation dataset**
  - The current ground truth dataset is small and focused on two products (`pan`, `airfryer`).
  - This limits the robustness of retrieval and answer quality metrics.

- **Heuristic faithfulness checking**
  - The faithfulness module relies on token overlap and simple numeric pattern checks and may miss more subtle forms of hallucination.

- **Narrow domain coverage**
  - The current examples focus on cookware and small appliances; performance and behavior on very different product categories are not yet validated.


### 5.2 Plans to overcome challenges and move forward

- **Scalability and performance**
  - Introduce configuration for smaller / faster models (both encoder and cross‑encoder) for low‑resource environments.
  - Cache query embeddings, reranker scores especially for repeated questions.
  - Experiment with reducing the reranker pool size to trade off a bit of quality for latency.

- **Richer evaluation**
  - Expand `eval/groundtruth_dataset.jsonl` with more products and diverse question types (factual, comparative, unanswerable).
  - Add separate splits (train/dev/test).
  - Track metric trends over time as retrieval, prompting or models change.

- **Improved faithfulness and calibration**
  - Iterate on the faithfulness heuristics to:
    - Better capture paraphrases and synonyms (e.g. via embeddings on answer segments).
    - Detect unsupported numeric claims more robustly.
  - Use `evaluate_answer` outputs not only to adjust confidence but also to inform UI warnings, loggings.

- **Domain generalization and robustness**
  - Ingest and index more varied product categories to test how well the current chunking, retrieval, and answer strategies generalize.
  - Adjust chunking strategies for pages with very different structure.

