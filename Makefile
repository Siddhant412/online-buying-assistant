.PHONY: help sync sync-dev lint format typecheck test migrate migration run-api run-worker run-ui docker-up docker-down docker-logs

PYTHON ?= python
UV ?= uv

help:
	@echo "Available commands:"
	@echo "  make sync        Install runtime dependencies with uv"
	@echo "  make sync-dev    Install development dependencies with uv"
	@echo "  make lint        Run ruff checks"
	@echo "  make format      Format Python code with ruff"
	@echo "  make typecheck   Run mypy"
	@echo "  make test        Run pytest"
	@echo "  make migrate     Apply Alembic migrations"
	@echo "  make migration   Create an Alembic migration, e.g. make migration MSG='add table'"
	@echo "  make run-api     Start FastAPI locally"
	@echo "  make run-worker  Start the RQ worker locally"
	@echo "  make run-ui      Start the legacy Streamlit UI"
	@echo "  make docker-up   Start local services with Docker Compose"

sync:
	$(UV) sync --extra rag --extra scrape --extra ui --extra postgres --extra worker --extra observability

sync-dev:
	$(UV) sync --extra dev

lint:
	$(UV) run ruff check app/api.py app/db app/db_retriever.py app/embeddings.py app/errors.py app/ingestion_tasks.py app/logging_config.py app/middleware.py app/providers.py app/queue.py app/settings.py app/worker.py migrations tests

format:
	$(UV) run ruff format app/api.py app/db app/db_retriever.py app/embeddings.py app/errors.py app/ingestion_tasks.py app/logging_config.py app/middleware.py app/providers.py app/queue.py app/settings.py app/worker.py migrations tests

typecheck:
	$(UV) run mypy --follow-imports=skip app/api.py app/db app/db_retriever.py app/embeddings.py app/errors.py app/ingestion_tasks.py app/logging_config.py app/middleware.py app/providers.py app/queue.py app/settings.py app/worker.py

test:
	$(UV) run pytest

migrate:
	$(UV) run alembic upgrade head

migration:
	$(UV) run alembic revision --autogenerate -m "$(MSG)"

run-api:
	$(UV) run uvicorn app.api:app --reload --port 8000

run-worker:
	$(UV) run oba-worker

run-ui:
	$(UV) run streamlit run ui/app.py

docker-up:
	docker compose up --build

docker-down:
	docker compose down --remove-orphans

docker-logs:
	docker compose logs -f
