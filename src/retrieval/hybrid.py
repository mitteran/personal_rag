"""Hybrid retrieval combining vector similarity and BM25 keyword search."""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_postgres import PGVector

from src.logging_config import get_logger
from src.retrieval.bm25_store import BM25Store
from src.settings import Settings

logger = get_logger(__name__)


class HybridRetriever:
    """Combines vector similarity search with BM25 keyword search."""

    def __init__(
        self,
        vector_store: PGVector,
        bm25_store: BM25Store,
        vector_weight: float = 0.5,
    ):
        """Initialize hybrid retriever.

        Parameters
        ----------
        vector_store : PGVector
            Vector similarity search store
        bm25_store : BM25Store
            BM25 keyword search store
        vector_weight : float
            Weight for vector search (0-1). BM25 weight is (1 - vector_weight).
            Default 0.5 means equal weighting.
        """
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.vector_weight = vector_weight
        self.bm25_weight = 1.0 - vector_weight

    def retrieve(self, query: str, top_k: int = 10) -> List[Document]:
        """Retrieve documents using hybrid search.

        Combines vector similarity and BM25 keyword search with reciprocal rank fusion.

        Parameters
        ----------
        query : str
            Search query
        top_k : int
            Number of results to return

        Returns
        -------
        List[Document]
            Ranked documents from hybrid search
        """
        # Retrieve more candidates from each method
        candidate_k = top_k * 3

        # Vector search
        vector_docs = self.vector_store.similarity_search(query, k=candidate_k)
        logger.debug(f"Vector search returned {len(vector_docs)} candidates")

        # BM25 search
        bm25_docs = self.bm25_store.search(query, top_k=candidate_k)
        logger.debug(f"BM25 search returned {len(bm25_docs)} candidates")

        # Reciprocal Rank Fusion (RRF)
        # RRF formula: score = sum(1 / (k + rank)) for each retrieval method
        # k=60 is a common default constant
        k_constant = 60
        doc_scores = {}

        # Score vector results
        for rank, doc in enumerate(vector_docs, start=1):
            doc_id = doc.page_content[:100]  # Use content prefix as ID
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"doc": doc, "score": 0.0}
            doc_scores[doc_id]["score"] += self.vector_weight * (1.0 / (k_constant + rank))

        # Score BM25 results
        for rank, doc in enumerate(bm25_docs, start=1):
            doc_id = doc.page_content[:100]
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"doc": doc, "score": 0.0}
            doc_scores[doc_id]["score"] += self.bm25_weight * (1.0 / (k_constant + rank))

        # Sort by combined score and return top-k
        ranked = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)[:top_k]
        results = [item["doc"] for item in ranked]

        logger.info(f"Hybrid retrieval returned {len(results)} documents")
        return results
