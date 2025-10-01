from __future__ import annotations

import os
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings

from src.logging_config import get_logger
from src.settings import Settings

logger = get_logger(__name__)


class MissingOpenAIKeyError(RuntimeError):
    pass


def _ensure_openai_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        logger.error("OPENAI_API_KEY not found in environment")
        raise MissingOpenAIKeyError(
            "OPENAI_API_KEY must be set in the environment to compute embeddings."
        )
    logger.debug("OpenAI API key found")
    return key


def build_embeddings(settings: Settings) -> Embeddings:
    key = _ensure_openai_key()
    logger.info(f"Initializing embeddings with model: {settings.rag.embedding_model}")
    return OpenAIEmbeddings(model=settings.rag.embedding_model, openai_api_key=key)


def upsert_documents(documents, settings: Settings, *, reindex: bool = False) -> PGVector:
    embeddings = build_embeddings(settings)
    collection = settings.database.collection

    if reindex:
        logger.warning(f"Reindexing: deleting existing collection '{collection}'")
    else:
        logger.info(f"Upserting {len(documents)} documents to collection '{collection}'")

    logger.debug(f"Connecting to database: {settings.database.host}:{settings.database.port}/{settings.database.dbname}")
    store = PGVector.from_documents(
        documents=documents,
        embedding=embeddings,
        connection=settings.database.connection_string,
        collection_name=collection,
        use_jsonb=True,
        pre_delete_collection=reindex,
    )
    logger.info(f"Successfully upserted {len(documents)} documents to vector store")
    return store


def get_store(settings: Settings) -> PGVector:
    embeddings = build_embeddings(settings)
    collection = settings.database.collection
    logger.debug(f"Connecting to vector store collection '{collection}'")
    return PGVector(
        embeddings=embeddings,
        connection=settings.database.connection_string,
        collection_name=collection,
        use_jsonb=True,
    )
