from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.loaders import load_documents
from src.settings import Settings, get_settings
from src.vectorstore.pgvector import MissingOpenAIKeyError, get_store, upsert_documents


def _split_documents(documents: Iterable[Document], settings: Settings) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.rag.chunk_size,
        chunk_overlap=settings.rag.chunk_overlap,
    )
    return splitter.split_documents(list(documents))


def ingest_corpus(content_path: Path, *, reindex: bool = False, config_path: Path | None = None) -> int:
    settings = get_settings(config_path)
    documents = load_documents(content_path)
    if not documents:
        raise FileNotFoundError(f"No supported documents found under {content_path}.")

    chunks = _split_documents(documents, settings)
    upsert_documents(chunks, settings, reindex=reindex)
    return len(chunks)


def _build_qa_chain(settings: Settings, *, top_k: int | None = None) -> RetrievalQA:
    llm = ChatOpenAI(model=settings.rag.chat_model, temperature=0.1)
    store = get_store(settings)
    retriever = store.as_retriever(search_kwargs={"k": top_k or settings.rag.top_k})
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)


def query(question: str, *, top_k: int | None = None, config_path: Path | None = None) -> Tuple[str, List[Document]]:
    settings = get_settings(config_path)
    chain = _build_qa_chain(settings, top_k=top_k)
    result = chain.invoke({"query": question})
    answer = result["result"]
    sources = result.get("source_documents", [])
    return answer, sources


__all__ = ["MissingOpenAIKeyError", "ingest_corpus", "query"]
