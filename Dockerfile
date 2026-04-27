FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir --upgrade pip uv

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --extra rag --extra scrape --extra postgres --extra worker --extra observability --no-dev --no-install-project

COPY app ./app
COPY eval ./eval
COPY scripts ./scripts
COPY ui ./ui
COPY data/raw ./data/raw

RUN uv sync --frozen --extra rag --extra scrape --extra postgres --extra worker --extra observability --no-dev

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
