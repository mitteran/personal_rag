from pathlib import Path

from src.settings import get_settings


def test_settings_loader(tmp_path: Path, monkeypatch):
    get_settings.cache_clear()
    for key in [
        "DB_HOST",
        "DB_PORT",
        "DB_USER",
        "DB_PASSWORD",
        "DB_NAME",
        "DB_COLLECTION",
    ]:
        monkeypatch.delenv(key, raising=False)
    config = tmp_path / "settings.yaml"
    config.write_text(
        """
database:
  host: test
  port: 6543
  user: someone
  password: secret
  dbname: rag
  collection: alt
rag:
  chunk_size: 500
  chunk_overlap: 50
  embedding_model: embed
  chat_model: chat
  top_k: 3
"""
    )

    settings = get_settings(config)

    assert settings.database.host == "test"
    assert settings.database.port == 6543
    assert settings.database.collection == "alt"
    assert settings.database.connection_string.endswith("@test:6543/rag")
    assert settings.rag.top_k == 3
    assert settings.rag.embedding_model == "embed"
