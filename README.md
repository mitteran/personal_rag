# Personal RAG for PDF and EPUB files

Local retrieval-augmented generation stack built with Python, LangChain, pgvector, and OpenAI models.

## Features

- **Enhanced Retrieval**: HyDE query transformation, hybrid search (vector + BM25), and cross-encoder reranking for improved accuracy
- Ingest Markdown, text, PDF, and EPUB documents into a pgvector-backed index.
- Multiple collections support: organize documents by topic in separate indexes.
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

### Working with Collections

The system supports multiple collections to organize documents by topic. Each folder in `data/raw/` automatically becomes a separate collection:

- Ingest documents into collections (collection name defaults to folder name):
  ```bash
  python -m src.pipeline.cli ingest data/raw/data_engineering
  python -m src.pipeline.cli ingest data/raw/history
  python -m src.pipeline.cli ingest data/raw/science
  ```
- Or specify a custom collection name:
  ```bash
  python -m src.pipeline.cli ingest data/raw/my_docs -c custom_name
  ```
- Ask a question against a specific collection:
  ```bash
  python -m src.pipeline.cli ask "What is Hadoop?" -c data_engineering
  python -m src.pipeline.cli ask "Who was Napoleon?" -c history
  ```
- Start an interactive chat session with a specific collection:
  ```bash
  python -m src.pipeline.cli chat -c science
  ```
- Launch the web chatbot (listens on `http://127.0.0.1:8000` by default):
  ```bash
  uvicorn src.web.app:app --reload
  ```

**Note:** If no collection is specified in queries, the default collection from `config/settings.yaml` is used.

### Enhanced Retrieval

The system uses three advanced retrieval techniques by default:

1. **HyDE (Hypothetical Document Embeddings)**: Generates a hypothetical answer to your query, then searches for documents similar to that answer
2. **Hybrid Search**: Combines vector similarity (semantic) with BM25 keyword search using reciprocal rank fusion
3. **Cross-Encoder Reranking**: Retrieves more candidates, then uses a cross-encoder model to rerank them for better relevance

Enhanced retrieval is **enabled by default**. To use standard retrieval:
```bash
python -m src.pipeline.cli ask "Your question" --standard
python -m src.pipeline.cli chat --standard
```

The BM25 index is automatically built during ingestion and stored in PostgreSQL alongside the vector index.

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
- `src/retrieval/` enhanced retrieval components (HyDE, hybrid search, reranking).
- `src/pipeline/` CLI and LangChain orchestration.
- `config/settings.yaml` runtime configuration.
- `tests/` pytest suite covering utilities.

Environment variables such as `DB_HOST`, `DB_PORT`, and `OPENAI_API_KEY` override values in `config/settings.yaml`, making it easy to point the stack at different infrastructure per environment.
