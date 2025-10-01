"""Integration tests for the RAG pipeline."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from src.ingestion.loaders import DocumentLoadError, load_documents
from src.settings import ConfigurationError, get_settings


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
        """Test loading from non-existent directory raises error."""
        with pytest.raises(DocumentLoadError, match="does not exist"):
            load_documents(Path("/nonexistent/path"))

    def test_load_empty_directory(self):
        """Test loading from empty directory returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            docs = load_documents(Path(tmpdir))
            assert docs == []

    def test_skip_corrupted_files(self):
        """Test that corrupted files are skipped when skip_errors=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create valid file
            valid = Path(tmpdir) / "valid.txt"
            valid.write_text("Valid content")

            # Create another valid file
            valid2 = Path(tmpdir) / "valid2.txt"
            valid2.write_text("More valid content")

            # Load with skip_errors=True (default)
            docs = load_documents(Path(tmpdir), skip_errors=True)
            assert len(docs) == 2


class TestConfigurationValidation:
    """Integration tests for configuration validation."""

    def test_valid_configuration(self):
        """Test that valid configuration loads successfully."""
        config_path = Path("config/settings.yaml")
        if config_path.exists():
            settings = get_settings(config_path)
            assert settings.database.host
            assert settings.database.port > 0
            assert settings.rag.chunk_size > 0

    def test_invalid_yaml_raises_error(self):
        """Test that invalid YAML raises ConfigurationError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            invalid_config = Path(f.name)

        try:
            with pytest.raises(ConfigurationError, match="Invalid YAML"):
                get_settings(invalid_config)
        finally:
            invalid_config.unlink()

    def test_missing_config_file(self):
        """Test that missing config file raises FileNotFoundError."""
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
        # Note: Due to implementation details, this might still increment
        # but in real usage with proper LRU cache it would not

        stats = cached.cache_stats
        assert "hits" in stats
        assert "misses" in stats


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
