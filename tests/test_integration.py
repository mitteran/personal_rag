"""Integration tests for the RAG pipeline."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.ingestion.loaders import load_documents
from src.settings import get_settings


class TestDocumentLoading:
    """Integration tests for document loading."""

    def test_load_text_documents(self):
        """Test loading text documents from a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("This is a test document.")

            # Load documents
            docs = load_documents(Path(tmpdir))

            assert len(docs) == 1
            assert "test document" in docs[0].page_content

    def test_load_nonexistent_directory(self):
        """Non-existent directories should yield no documents."""
        docs = load_documents(Path("/nonexistent/path"))
        assert docs == []

    def test_load_empty_directory(self):
        """Test loading from empty directory returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            docs = load_documents(Path(tmpdir))
            assert docs == []

    def test_unsupported_files_are_ignored(self):
        """Unsupported file types should be skipped without raising."""
        with tempfile.TemporaryDirectory() as tmpdir:
            supported = Path(tmpdir) / "valid.txt"
            supported.write_text("Valid content")

            unsupported = Path(tmpdir) / "note.docx"
            unsupported.write_text("Ignored content")

            docs = load_documents(Path(tmpdir))

            assert len(docs) == 1
            assert docs[0].metadata["source"].endswith("valid.txt")


class TestConfigurationValidation:
    """Integration tests for configuration validation."""

    def test_valid_configuration(self):
        """Test that valid configuration loads successfully."""
        config_path = Path("config/settings.yaml")
        if config_path.exists():
            get_settings.cache_clear()
            settings = get_settings(config_path)
            assert settings.database.host
            assert settings.database.port > 0
            assert settings.rag.chunk_size > 0

    def test_invalid_yaml_raises_error(self):
        """Invalid YAML should bubble up parser errors."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            invalid_config = Path(f.name)

        try:
            get_settings.cache_clear()
            with pytest.raises(yaml.YAMLError):
                get_settings(invalid_config)
        finally:
            invalid_config.unlink()

    def test_missing_config_file(self):
        """Test that missing config file raises FileNotFoundError."""
        get_settings.cache_clear()
        with pytest.raises(FileNotFoundError):
            get_settings(Path("/nonexistent/config.yaml"))


class TestErrorHandling:
    """Integration tests for error handling."""

    def test_missing_openai_key_raises_error(self):
        """Test that missing OPENAI_API_KEY is caught."""
        from src.vectorstore.pgvector import MissingOpenAIKeyError, _ensure_openai_key

        # Temporarily remove API key
        original_key = os.environ.get("OPENAI_API_KEY")
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

        try:
            with pytest.raises(MissingOpenAIKeyError):
                _ensure_openai_key()
        finally:
            # Restore API key
            if original_key:
                os.environ["OPENAI_API_KEY"] = original_key


class TestCaching:
    """Integration tests for caching functionality."""

    def test_embedding_cache_reduces_calls(self):
        """Test that embedding cache reduces API calls."""
        from src.vectorstore.cache import CachedEmbeddings

        # Mock embeddings class
        class MockEmbeddings:
            def __init__(self):
                self.call_count = 0

            def embed_query(self, text: str):
                self.call_count += 1
                return [0.1, 0.2, 0.3]

            def embed_documents(self, texts):
                self.call_count += len(texts)
                return [[0.1, 0.2, 0.3] for _ in texts]

        mock = MockEmbeddings()
        cached = CachedEmbeddings(mock, cache_size=10)

        # First call
        cached.embed_query("test")
        assert mock.call_count == 1

        # Second call (should hit cache)
        cached.embed_query("test")
        assert mock.call_count == 1

        stats = cached.cache_stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1


@pytest.mark.skipif(
    not os.environ.get("INTEGRATION_TESTS"),
    reason="Integration tests require database and API keys",
)
class TestFullPipeline:
    """Full integration tests requiring database and API access.

    These tests are skipped by default. Run with:
        INTEGRATION_TESTS=1 pytest tests/test_integration.py
    """

    def test_end_to_end_ingestion_and_query(self):
        """Test complete ingestion and query pipeline."""
        # This would require a running database and valid API keys
        pytest.skip("Requires database setup")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
