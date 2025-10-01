from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple
from uuid import uuid4

from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.loaders import load_documents
from src.logging_config import get_logger
from src.pipeline.memory import ChatMemoryStore, ChatMessage, serialize_history
from src.settings import Settings, get_settings
from src.vectorstore.pgvector import MissingOpenAIKeyError, get_store, upsert_documents

logger = get_logger(__name__)


def _split_documents(documents: Iterable[Document], settings: Settings) -> List[Document]:
    logger.info(f"Splitting documents with chunk_size={settings.rag.chunk_size}, overlap={settings.rag.chunk_overlap}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.rag.chunk_size,
        chunk_overlap=settings.rag.chunk_overlap,
    )
    docs_list = list(documents)
    chunks = splitter.split_documents(docs_list)
    logger.info(f"Split {len(docs_list)} documents into {len(chunks)} chunks")
    return chunks


def ingest_corpus(content_path: Path, *, reindex: bool = False, config_path: Path | None = None) -> int:
    logger.info(f"Starting corpus ingestion from {content_path} (reindex={reindex})")
    settings = get_settings(config_path)
    documents = load_documents(content_path)

    if not documents:
        logger.error(f"No supported documents found under {content_path}")
        raise FileNotFoundError(f"No supported documents found under {content_path}.")

    chunks = _split_documents(documents, settings)
    upsert_documents(chunks, settings, reindex=reindex)
    logger.info(f"Ingestion complete: {len(chunks)} chunks persisted")
    return len(chunks)


def _build_qa_chain(settings: Settings, *, top_k: int | None = None) -> RetrievalQA:
    k_value = top_k or settings.rag.top_k
    logger.debug(f"Building QA chain with model={settings.rag.chat_model}, top_k={k_value}")
    llm = ChatOpenAI(model=settings.rag.chat_model, temperature=0.1)
    store = get_store(settings)
    retriever = store.as_retriever(search_kwargs={"k": k_value})
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)


def query(question: str, *, top_k: int | None = None, config_path: Path | None = None) -> Tuple[str, List[Document]]:
    logger.info(f"Processing query: {question[:100]}...")
    settings = get_settings(config_path)
    chain = _build_qa_chain(settings, top_k=top_k)
    result = chain.invoke({"query": question})
    answer = result["result"]
    sources = result.get("source_documents", [])
    logger.info(f"Query completed with {len(sources)} source documents")
    return answer, sources


_memory_store = ChatMemoryStore()


@dataclass
class ChatResult:
    session_id: str
    answer: str
    sources: List[Document]
    history: List[ChatMessage]


def _build_conversational_chain(
    settings: Settings, *, top_k: int | None = None
) -> RunnableWithMessageHistory:
    llm = ChatOpenAI(model=settings.rag.chat_model, temperature=0.1)
    store = get_store(settings)
    retriever = store.as_retriever(search_kwargs={"k": top_k or settings.rag.top_k})

    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Given the following conversation and a follow up question, rewrite the"
                " follow up question to be a standalone query. If it already stands"
                " alone, return it unchanged.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Use the supplied context to answer the"
                " question. If the answer is not contained in the context, say you"
                " do not know.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "Context:\n{context}\n\nQuestion: {input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )
    doc_chain = create_stuff_documents_chain(llm, answer_prompt)
    retrieval_chain = create_retrieval_chain(history_aware_retriever, doc_chain)

    return RunnableWithMessageHistory(
        retrieval_chain,
        _memory_store.get_session,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


def start_chat_session(session_id: str | None = None) -> str:
    identifier = session_id or uuid4().hex
    _memory_store.get_session(identifier)
    logger.info(f"Started chat session: {identifier}")
    return identifier


def chat(
    session_id: str,
    message: str,
    *,
    top_k: int | None = None,
    config_path: Path | None = None,
) -> ChatResult:
    if not session_id:
        logger.error("Chat called without session_id")
        raise ValueError("session_id must be provided for chat interactions.")

    logger.info(f"Chat message received for session {session_id}: {message[:100]}...")
    settings = get_settings(config_path)
    chain = _build_conversational_chain(settings, top_k=top_k)
    result = chain.invoke(
        {"input": message},
        config={"configurable": {"session_id": session_id}},
    )
    sources = result.get("context", [])
    history = serialize_history(_memory_store.get_session(session_id))
    logger.info(f"Chat response generated with {len(sources)} sources, history length: {len(history)}")
    return ChatResult(
        session_id=session_id,
        answer=result["answer"],
        sources=sources,
        history=history,
    )


def reset_chat_session(session_id: str) -> None:
    logger.info(f"Resetting chat session: {session_id}")
    _memory_store.drop_session(session_id)


def get_chat_history(session_id: str) -> List[ChatMessage]:
    return serialize_history(_memory_store.get_session(session_id))


__all__ = [
    "MissingOpenAIKeyError",
    "ChatResult",
    "chat",
    "get_chat_history",
    "ingest_corpus",
    "query",
    "reset_chat_session",
    "start_chat_session",
]
