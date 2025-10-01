"""Embedding cache to reduce redundant API calls."""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
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
        self._cache: "OrderedDict[str, tuple[float, ...]]" = OrderedDict()

    @staticmethod
    def _hash_text(text: str) -> str:
        """Create hash of text for cache key."""
        return hashlib.sha256(text.encode()).hexdigest()

    def _get_cached(self, key: str) -> tuple[float, ...] | None:
        value = self._cache.get(key)
        if value is not None:
            self._cache.move_to_end(key)
        return value

    def _set_cached(self, key: str, vector: List[float]) -> None:
        if self.cache_size == 0:
            return
        self._cache[key] = tuple(vector)
        self._cache.move_to_end(key)
        if len(self._cache) > self.cache_size:
            evicted_key, _ = self._cache.popitem(last=False)
            logger.debug("Evicted key from embedding cache: %s", evicted_key)

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
        if self.cache_size == 0:
            return self.embeddings.embed_query(text)

        key = self._hash_text(text)
        cached = self._get_cached(key)
        if cached is not None:
            self._cache_hits += 1
            logger.debug(
                "Cache hit for query (hit rate: %.1f%%)", self.cache_hit_rate * 100
            )
            return list(cached)

        self._cache_misses += 1
        logger.debug(
            "Cache miss for query (total misses: %s)", self._cache_misses
        )
        vector = self.embeddings.embed_query(text)
        self._set_cached(key, vector)
        return list(vector)

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
        if self.cache_size == 0:
            return self.embeddings.embed_documents(texts)

        results: List[List[float] | None] = [None] * len(texts)
        uncached: List[str] = []
        uncached_indices: List[int] = []

        for idx, text in enumerate(texts):
            key = self._hash_text(text)
            cached = self._get_cached(key)
            if cached is not None:
                self._cache_hits += 1
                results[idx] = list(cached)
            else:
                uncached.append(text)
                uncached_indices.append(idx)

        if uncached:
            logger.debug(
                "Embedding %s/%s uncached documents", len(uncached), len(texts)
            )
            embeddings = self.embeddings.embed_documents(uncached)
            for text, embedding in zip(uncached, embeddings):
                key = self._hash_text(text)
                self._cache_misses += 1
                self._set_cached(key, embedding)

            for idx, text in zip(uncached_indices, uncached):
                key = self._hash_text(text)
                cached = self._get_cached(key)
                if cached is not None:
                    results[idx] = list(cached)

        logger.info(
            "Embedding batch complete (cache hit rate: %.1f%%)",
            self.cache_hit_rate * 100,
        )

        final_results: List[List[float]] = []
        for vec in results:
            if vec is None:
                raise RuntimeError("Embedding cache returned an incomplete result.")
            final_results.append(vec)
        return final_results

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
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Cleared embedding cache")


__all__ = ["CachedEmbeddings"]
