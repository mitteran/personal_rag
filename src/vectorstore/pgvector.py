from __future__ import annotations

import os
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings

from src.settings import Settings


class MissingOpenAIKeyError(RuntimeError):
    pass


def _ensure_openai_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise MissingOpenAIKeyError(
            "OPENAI_API_KEY must be set in the environment to compute embeddings."
        )
    return key


def build_embeddings(settings: Settings) -> Embeddings:
    key = _ensure_openai_key()
    return OpenAIEmbeddings(model=settings.rag.embedding_model, openai_api_key=key)


def upsert_documents(documents, settings: Settings, *, reindex: bool = False) -> PGVector:
    embeddings = build_embeddings(settings)
    return PGVector.from_documents(
        documents=documents,
        embedding=embeddings,
        connection=settings.database.connection_string,
        collection_name=settings.database.collection,
        use_jsonb=True,
        pre_delete_collection=reindex,
    )


def get_store(settings: Settings) -> PGVector:
    embeddings = build_embeddings(settings)
    return PGVector(
        embeddings=embeddings,
        connection=settings.database.connection_string,
        collection_name=settings.database.collection,
        use_jsonb=True,
    )
