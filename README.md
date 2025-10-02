# Personal RAG for PDF and EPUB files

Local retrieval-augmented generation stack built with Python, LangChain, pgvector, and OpenAI models.

## Features

- Ingest Markdown, text, PDF, and EPUB documents into a pgvector-backed index.
- Chunk documents with LangChain splitters tuned for long-form references.
- Automatic embedding cache trims duplicate OpenAI calls during ingest and query.
- Query via CLI or API with explicit source citations on every answer.
- Multi-turn chat backed by PostgreSQL memory, with an in-memory fallback for local runs.
- Lightweight FastAPI web interface for interactive chat on desktop browsers.

## Getting Started

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Provision PostgreSQL 15+ with the `pgvector` extension. Update `config/settings.yaml` with your connection details.
3. Export your OpenAI key: `echo "OPENAI_API_KEY=sk-..." >> .env` and `source .env`.

## Usage

- Ingest documents from `data/raw/`:
  ```bash
  python -m src.pipeline.cli ingest data/raw --reindex
  ```
- Ask a question against the index:
  ```bash
  python -m src.pipeline.cli ask "What is the main gist of the documents?"
  ```
- Start an interactive chat session with conversational memory:
  ```bash
  python -m src.pipeline.cli chat
  ```
- Launch the web chatbot (listens on `http://127.0.0.1:8000` by default):
  ```bash
 uvicorn src.web.app:app --reload
  ```

## Testing

Run the full suite (unit and integration tests) after installing dependencies:

```bash
pytest -q
```

Tests that touch the vector store or chat memory will automatically fall back to in-memory storage when a PostgreSQL instance is not available.

## Setting up pgvector

1. Ensure docker is installed and running (https://docs.docker.com/get-docker/)
2. Run the following command to start the postgres container:

```bash
docker run \
    --name pgvector-container \
    -e POSTGRES_USER=persrag \
    -e POSTGRES_PASSWORD=loremipsum \
    -e POSTGRES_DB=rag \
    -p 6024:5432 \
    -d pgvector/pgvector:pg16
```

If the Docker container already exists, start it with `docker start pgvector-container`.


## Project Layout

- `src/ingestion/` document discovery and loaders.
- `src/vectorstore/` pgvector helpers and embeddings wiring.
- `src/pipeline/` CLI and LangChain orchestration.
- `config/settings.yaml` runtime configuration.
- `tests/` pytest suite covering utilities.

Environment variables such as `DB_HOST`, `DB_PORT`, and `OPENAI_API_KEY` override values in `config/settings.yaml`, making it easy to point the stack at different infrastructure per environment.
