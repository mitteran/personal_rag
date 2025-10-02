"""BM25 keyword search implementation with PostgreSQL persistence."""

from __future__ import annotations

import json
import pickle
from typing import List

import psycopg
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.logging_config import get_logger
from src.settings import Settings

logger = get_logger(__name__)


class BM25Store:
    """BM25 keyword search index with PostgreSQL backing."""

    def __init__(self, settings: Settings, collection: str):
        self.settings = settings
        self.collection = collection
        self.bm25: BM25Okapi | None = None
        self.documents: List[Document] = []
        self._load_from_db()

    def _get_table_name(self) -> str:
        """Get sanitized table name for this collection's BM25 index."""
        # Simple sanitization - replace non-alphanumeric with underscore
        safe_name = "".join(c if c.isalnum() else "_" for c in self.collection)
        return f"bm25_index_{safe_name}"

    def _load_from_db(self) -> None:
        """Load BM25 index and documents from PostgreSQL."""
        try:
            with psycopg.connect(self.settings.database.psycopg_connection_string) as conn:
                with conn.cursor() as cur:
                    table_name = self._get_table_name()
                    cur.execute(
                        f"""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables
                            WHERE table_name = %s
                        )
                        """,
                        (table_name,),
                    )
                    if not cur.fetchone()[0]:
                        logger.debug(f"BM25 table {table_name} does not exist yet")
                        return

                    cur.execute(f"SELECT bm25_data, documents FROM {table_name} LIMIT 1")
                    row = cur.fetchone()
                    if row:
                        bm25_bytes, docs_data = row
                        self.bm25 = pickle.loads(bm25_bytes)
                        # PostgreSQL JSONB returns as Python list/dict, not JSON string
                        docs_list = docs_data if isinstance(docs_data, list) else json.loads(docs_data)
                        self.documents = [
                            Document(page_content=d["page_content"], metadata=d["metadata"])
                            for d in docs_list
                        ]
                        logger.info(f"Loaded BM25 index with {len(self.documents)} documents")
        except Exception as e:
            logger.warning(f"Could not load BM25 index: {e}")

    def build_index(self, documents: List[Document]) -> None:
        """Build BM25 index from documents and persist to PostgreSQL."""
        if not documents:
            logger.warning("No documents provided for BM25 indexing")
            return

        self.documents = documents
        tokenized_corpus = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Serialize and save to database
        bm25_bytes = pickle.dumps(self.bm25)
        docs_json = json.dumps(
            [{"page_content": d.page_content, "metadata": d.metadata} for d in documents]
        )

        try:
            with psycopg.connect(self.settings.database.psycopg_connection_string) as conn:
                with conn.cursor() as cur:
                    table_name = self._get_table_name()
                    # Create table if not exists
                    cur.execute(
                        f"""
                        CREATE TABLE IF NOT EXISTS {table_name} (
                            id SERIAL PRIMARY KEY,
                            bm25_data BYTEA NOT NULL,
                            documents JSONB NOT NULL,
                            updated_at TIMESTAMP DEFAULT NOW()
                        )
                        """
                    )
                    # Clear existing data and insert new
                    cur.execute(f"DELETE FROM {table_name}")
                    cur.execute(
                        f"INSERT INTO {table_name} (bm25_data, documents) VALUES (%s, %s)",
                        (bm25_bytes, docs_json),
                    )
                    conn.commit()
                    logger.info(f"Persisted BM25 index with {len(documents)} documents")
        except Exception as e:
            logger.error(f"Failed to persist BM25 index: {e}")

    def search(self, query: str, top_k: int = 10) -> List[Document]:
        """Search using BM25 keyword matching."""
        if not self.bm25 or not self.documents:
            logger.warning("BM25 index not available, returning empty results")
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = [self.documents[i] for i in top_indices if scores[i] > 0]
        logger.debug(f"BM25 search returned {len(results)} results for query: {query[:50]}")
        return results

    def clear(self) -> None:
        """Clear the BM25 index from database."""
        try:
            with psycopg.connect(self.settings.database.psycopg_connection_string) as conn:
                with conn.cursor() as cur:
                    table_name = self._get_table_name()
                    cur.execute(f"DROP TABLE IF EXISTS {table_name}")
                    conn.commit()
                    logger.info(f"Cleared BM25 index table {table_name}")
        except Exception as e:
            logger.error(f"Failed to clear BM25 index: {e}")
        self.bm25 = None
        self.documents = []
