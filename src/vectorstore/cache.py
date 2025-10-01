"""Embedding cache to reduce redundant API calls."""

from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from typing import List

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class CachedEmbeddings(Embeddings):
    """Wrapper for embeddings with LRU caching.

    This reduces redundant API calls to OpenAI for identical text.
    """

    def __init__(self, embeddings: Embeddings, cache_size: int = 1000):
        """Initialize cached embeddings.

        Parameters
        ----------
        embeddings : Embeddings
            Underlying embeddings model
        cache_size : int
            Maximum number of embeddings to cache
        """
        self.embeddings = embeddings
        self.cache_size = cache_size
        self._cache_hits = 0
        self._cache_misses = 0

    @staticmethod
    def _hash_text(text: str) -> str:
        """Create hash of text for cache key."""
        return hashlib.sha256(text.encode()).hexdigest()

    @lru_cache(maxsize=1000)
    def _cached_embed_query(self, text_hash: str, text: str) -> List[float]:
        """Cache single embedding lookup."""
        self._cache_misses += 1
        logger.debug(f"Cache miss for query (total misses: {self._cache_misses})")
        return self.embeddings.embed_query(text)

    def embed_query(self, text: str) -> List[float]:
        """Embed query text with caching.

        Parameters
        ----------
        text : str
            Query text to embed

        Returns
        -------
        List[float]
            Embedding vector
        """
        text_hash = self._hash_text(text)

        # Check if in cache
        try:
            result = self._cached_embed_query.__wrapped__.__self__._cached_embed_query(
                self, text_hash, text
            )
            if result is not None:
                self._cache_hits += 1
                logger.debug(
                    f"Cache hit for query (hit rate: {self.cache_hit_rate:.1%})"
                )
            return result
        except AttributeError:
            # Cache not initialized yet
            pass

        return self._cached_embed_query(text_hash, text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents with caching.

        Parameters
        ----------
        texts : List[str]
            Documents to embed

        Returns
        -------
        List[List[float]]
            Embedding vectors
        """
        # For bulk operations, check cache individually
        embeddings = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            text_hash = self._hash_text(text)
            # Try to get from cache
            try:
                cached = self._cached_embed_query.__wrapped__.__self__._cached_embed_query(
                    self, text_hash, text
                )
                if cached is not None:
                    embeddings.append(cached)
                    self._cache_hits += 1
                    continue
            except (AttributeError, TypeError):
                pass

            # Not in cache
            uncached_texts.append(text)
            uncached_indices.append(i)
            embeddings.append(None)  # Placeholder

        # Batch embed uncached texts
        if uncached_texts:
            logger.debug(
                f"Embedding {len(uncached_texts)}/{len(texts)} uncached documents"
            )
            uncached_embeddings = self.embeddings.embed_documents(uncached_texts)
            self._cache_misses += len(uncached_texts)

            # Fill in uncached embeddings
            for idx, embedding in zip(uncached_indices, uncached_embeddings):
                embeddings[idx] = embedding

                # Cache the result
                text_hash = self._hash_text(texts[idx])
                self._cached_embed_query(text_hash, texts[idx])

        logger.info(
            f"Embedding batch complete (cache hit rate: {self.cache_hit_rate:.1%})"
        )
        return embeddings

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self._cache_hits + self._cache_misses
        if total == 0:
            return 0.0
        return self._cache_hits / total

    @property
    def cache_stats(self) -> dict[str, int | float]:
        """Get cache statistics."""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": self.cache_hit_rate,
        }

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cached_embed_query.cache_clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Cleared embedding cache")


__all__ = ["CachedEmbeddings"]
