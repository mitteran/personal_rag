"""Cross-encoder reranking for retrieved documents."""

from __future__ import annotations

from typing import List, Tuple

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.logging_config import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    """Rerank documents using a cross-encoder model.

    Cross-encoders jointly encode query and document, providing more accurate
    relevance scores than bi-encoders (standard embeddings) but at higher cost.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize cross-encoder reranker.

        Parameters
        ----------
        model_name : str
            HuggingFace model name for cross-encoder.
            Default is a lightweight model trained on MS MARCO dataset.
        """
        logger.info(f"Loading cross-encoder model: {model_name}")
        self.model = CrossEncoder(model_name)
        logger.info("Cross-encoder model loaded successfully")

    def rerank(
        self, query: str, documents: List[Document], top_k: int | None = None
    ) -> List[Document]:
        """Rerank documents by relevance to query.

        Parameters
        ----------
        query : str
            Search query
        documents : List[Document]
            Documents to rerank
        top_k : int | None
            Number of top documents to return. If None, returns all documents reranked.

        Returns
        -------
        List[Document]
            Documents sorted by relevance score
        """
        if not documents:
            return []

        # Prepare query-document pairs
        pairs = [[query, doc.page_content] for doc in documents]

        # Score each pair
        scores = self.model.predict(pairs)

        # Combine documents with scores and sort
        doc_score_pairs: List[Tuple[Document, float]] = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # Return top-k or all
        if top_k is not None:
            doc_score_pairs = doc_score_pairs[:top_k]

        reranked_docs = [doc for doc, score in doc_score_pairs]
        logger.debug(
            f"Reranked {len(documents)} documents, returning top {len(reranked_docs)}"
        )

        return reranked_docs
