from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class DatabaseSettings:
    host: str
    port: int
    user: str
    password: str
    dbname: str
    collection: str

    @property
    def connection_string(self) -> str:
        """SQLAlchemy-style connection string for PGVector."""
        return (
            f"postgresql+psycopg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.dbname}"
        )

    @property
    def psycopg_connection_string(self) -> str:
        """Psycopg-style connection string for direct connections."""
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.dbname}"
        )


@dataclass
class RagSettings:
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    chat_model: str
    top_k: int


@dataclass
class Settings:
    database: DatabaseSettings
    rag: RagSettings


def _as_database_settings(raw: Dict[str, Any]) -> DatabaseSettings:
    return DatabaseSettings(
        host=os.getenv("DB_HOST", str(raw.get("host", "localhost"))),
        port=int(os.getenv("DB_PORT", raw.get("port", 5432))),
        user=os.getenv("DB_USER", str(raw.get("user", "postgres"))),
        password=os.getenv("DB_PASSWORD", str(raw.get("password", ""))),
        dbname=os.getenv("DB_NAME", str(raw.get("dbname", "postgres"))),
        collection=os.getenv("DB_COLLECTION", str(raw.get("collection", "rag_documents"))),
    )


def _as_rag_settings(raw: Dict[str, Any]) -> RagSettings:
    return RagSettings(
        chunk_size=int(raw.get("chunk_size", 1000)),
        chunk_overlap=int(raw.get("chunk_overlap", 200)),
        embedding_model=str(raw.get("embedding_model", "text-embedding-3-small")),
        chat_model=str(raw.get("chat_model", "gpt-4o-mini")),
        top_k=int(raw.get("top_k", 4)),
    )


@lru_cache(maxsize=1)
def get_settings(config_path: Path | None = None) -> Settings:
    path = config_path or Path("config/settings.yaml")
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {path.resolve()}."
        )

    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Configuration file is invalid; expected a top-level mapping.")

    database = _as_database_settings(dict(raw.get("database", {})))
    rag = _as_rag_settings(dict(raw.get("rag", {})))
    return Settings(database=database, rag=rag)
