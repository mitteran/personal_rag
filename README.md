# My personal RAG

Local retrieval-augmented generation stack built with Python, LangChain, pgvector, and OpenAI models.

## Features
- Ingest Markdown, text, PDF, and EPUB documents into a pgvector-backed index.
- Chunk documents with LangChain splitters tuned for long-form references.
- Query via CLI using OpenAI chat models with source attribution.

## Getting Started
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Provision PostgreSQL 15+ with the `pgvector` extension. Update `config/settings.yaml` with your connection details.
3. Export your OpenAI key: `echo "OPENAI_API_KEY=sk-..." >> .env` and `source .env` (or load via `python-dotenv`).

## Usage
- Ingest documents from `data/raw/`:
  ```bash
  python -m src.pipeline.cli ingest data/raw --reindex
  ```
- Ask a question against the index:
  ```bash
  python -m src.pipeline.cli ask "What does the design document say about evaluations?"
  ```

## Project Layout
- `src/ingestion/` document discovery and loaders.
- `src/vectorstore/` pgvector helpers and embeddings wiring.
- `src/pipeline/` CLI and LangChain orchestration.
- `config/settings.yaml` runtime configuration.
- `tests/` pytest suite covering utilities.
