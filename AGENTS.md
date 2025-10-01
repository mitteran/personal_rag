# Repository Guidelines

## Project Structure & Module Organization
Keep Python source under `src/` with focused subpackages: `src/ingestion/` for loaders, `src/vectorstore/` for pgvector clients, and `src/pipeline/` for LangChain flows. Store notebooks under `notebooks/` and raw corpora in `data/raw/`. Place processed chunks in `data/processed/`, keep secrets out of the repo, and drive shared settings from `config/settings.yaml`.

## Build, Test, and Development Commands
Use a local virtual environment: `python -m venv .venv && source .venv/bin/activate`. Install dependencies with `pip install -r requirements.txt`. Expose ingestion, indexing, and query flows via `src/pipeline/cli.py`; run `python -m src.pipeline.cli ingest data/raw` or `python -m src.pipeline.cli query "your question"`. After adding content, rerun ingest with `--reindex` to rebuild the vector database.

## Coding Style & Naming Conventions
Follow PEP 8 with Black formatting (`black src tests`). Enforce linting with Ruff (`ruff check src tests`). Prefer type hints and run `mypy src`. Name LangChain components with intent (`QueryRouterChain`, `EpubIngestionTask`). Keep modules lightweight and compose via dependency injection so agents can be swapped easily.

## Testing Guidelines
Add unit tests inside `tests/` mirroring the package layout (e.g., `tests/vectorstore/test_chroma.py`). Use `pytest` as the main runner (`pytest -q`). Stub external services with fixtures and spin up a temporary vector store under `tmp/` for integration runs. Track coverage with `pytest --cov=src --cov-report=term-missing`; keep ingestion and retrieval modules at ≥85%.

## Commit & Pull Request Guidelines
Adopt Conventional Commits (`feat: add epub loader`, `fix: correct retriever scoring`). Keep commits scoped and include runnable examples when possible. PR descriptions must summarize the RAG workflow impact, link issues, list new commands, and include before/after retrieval snippets or screenshots. Request review from another agent before merging.

## Data Handling & Security
Never commit proprietary documents or API credentials. Store environment variables in `.env` (set `OPENAI_API_KEY` for inference) and load them via `python-dotenv`. If sharing vector stores, sanitize embeddings and verify source permissions before distribution.

## Vector Store Requirements
Use PostgreSQL 15+ with the `pgvector` extension as the canonical vector database. Provision it via Docker Compose (`docker compose up db`) and enable the extension with `CREATE EXTENSION IF NOT EXISTS vector;`. Keep connection details in `config/settings.yaml` and `.env`. When experimenting with alternate stores (e.g., Chroma), document the rationale in the PR and provide migration scripts back to pgvector before merge.
