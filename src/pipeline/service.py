from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple
from uuid import uuid4

from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.loaders import load_documents
from src.logging_config import get_logger
from src.pipeline.memory import ChatMemoryStore, ChatMessage, serialize_history
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.hyde import HyDEQueryTransformer
from src.retrieval.reranker import CrossEncoderReranker
from src.settings import Settings, get_settings
from src.vectorstore.pgvector import MissingOpenAIKeyError, get_store, upsert_documents

logger = get_logger(__name__)

# Lazy-loaded reranker (expensive to initialize)
_reranker: CrossEncoderReranker | None = None


def _get_reranker() -> CrossEncoderReranker:
    """Get or initialize the reranker singleton."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker


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


def ingest_corpus(content_path: Path, *, reindex: bool = False, config_path: Path | None = None, collection: str | None = None) -> int:
    logger.info(f"Starting corpus ingestion from {content_path} (reindex={reindex})")
    settings = get_settings(config_path)

    # Use folder name as collection if not specified
    collection_name = collection or content_path.name
    logger.info(f"Using collection: {collection_name}")

    documents = load_documents(content_path)

    if not documents:
        logger.error(f"No supported documents found under {content_path}")
        raise FileNotFoundError(f"No supported documents found under {content_path}.")

    chunks = _split_documents(documents, settings)
    upsert_documents(chunks, settings, reindex=reindex, collection=collection_name)

    # Build BM25 index for hybrid search
    bm25_store = BM25Store(settings, collection_name)
    if reindex:
        bm25_store.clear()
    bm25_store.build_index(chunks)

    logger.info(f"Ingestion complete: {len(chunks)} chunks persisted")
    return len(chunks)


def _enhanced_retrieve(
    query: str, settings: Settings, *, top_k: int, collection: str | None = None, use_hyde: bool = True
) -> List[Document]:
    """Enhanced retrieval using HyDE, hybrid search, and reranking.

    Parameters
    ----------
    query : str
        User query
    settings : Settings
        Application settings
    top_k : int
        Number of final documents to return
    collection : str | None
        Collection name
    use_hyde : bool
        Whether to use HyDE query transformation

    Returns
    -------
    List[Document]
        Retrieved and reranked documents
    """
    # Step 1: HyDE query transformation (optional)
    search_query = query
    if use_hyde:
        hyde = HyDEQueryTransformer(settings)
        search_query = hyde.transform(query)

    # Step 2: Hybrid search (retrieve more candidates for reranking)
    collection_name = collection or settings.database.collection
    vector_store = get_store(settings, collection=collection)
    bm25_store = BM25Store(settings, collection_name)

    candidate_k = top_k * 3  # Retrieve 3x candidates for reranking
    hybrid_retriever = HybridRetriever(vector_store, bm25_store, vector_weight=0.5)
    candidates = hybrid_retriever.retrieve(search_query, top_k=candidate_k)

    # Step 3: Rerank with cross-encoder
    reranker = _get_reranker()
    reranked_docs = reranker.rerank(query, candidates, top_k=top_k)

    logger.info(f"Enhanced retrieval: {len(candidates)} candidates -> {len(reranked_docs)} final docs")
    return reranked_docs


def _build_qa_chain(settings: Settings, *, top_k: int | None = None, collection: str | None = None) -> RetrievalQA:
    k_value = top_k or settings.rag.top_k
    logger.debug(f"Building QA chain with model={settings.rag.chat_model}, top_k={k_value}")
    llm = ChatOpenAI(model=settings.rag.chat_model, temperature=0.1)
    store = get_store(settings, collection=collection)
    retriever = store.as_retriever(search_kwargs={"k": k_value})
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)


def query(
    question: str,
    *,
    top_k: int | None = None,
    config_path: Path | None = None,
    collection: str | None = None,
    use_enhanced: bool = True,
) -> Tuple[str, List[Document]]:
    """Query the RAG system.

    Parameters
    ----------
    question : str
        User question
    top_k : int | None
        Number of documents to retrieve
    config_path : Path | None
        Config file path
    collection : str | None
        Collection to search
    use_enhanced : bool
        Use enhanced retrieval (HyDE + hybrid + reranking). Default True.

    Returns
    -------
    Tuple[str, List[Document]]
        Answer and source documents
    """
    logger.info(f"Processing query: {question[:100]}...")
    settings = get_settings(config_path)
    k_value = top_k or settings.rag.top_k

    if use_enhanced:
        # Enhanced retrieval pipeline
        sources = _enhanced_retrieve(question, settings, top_k=k_value, collection=collection)
        llm = ChatOpenAI(model=settings.rag.chat_model, temperature=0.1)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant. Use the supplied context to answer the"
                    " question. If the answer is not contained in the context, say you"
                    " do not know.",
                ),
                ("human", "Context:\n{context}\n\nQuestion: {question}"),
            ]
        )
        context = "\n\n".join([doc.page_content for doc in sources])
        chain = prompt | llm
        response = chain.invoke({"context": context, "question": question})
        answer = response.content
    else:
        # Standard retrieval
        chain = _build_qa_chain(settings, top_k=k_value, collection=collection)
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


def _build_conversational_components(
    settings: Settings,
    *,
    top_k: int | None = None,
    collection: str | None = None,
    use_enhanced: bool = True,
):
    """Build chat components with optional enhanced retrieval."""
    llm = ChatOpenAI(model=settings.rag.chat_model, temperature=0.1)
    k_value = top_k or settings.rag.top_k

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

    if use_enhanced:
        # Return components for manual enhanced retrieval
        doc_chain = create_stuff_documents_chain(llm, answer_prompt)
        return doc_chain, None, settings, k_value, collection
    else:
        # Standard retriever-based approach
        store = get_store(settings, collection=collection)
        retriever = store.as_retriever(search_kwargs={"k": k_value})
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_prompt
        )
        doc_chain = create_stuff_documents_chain(llm, answer_prompt)
        return doc_chain, history_aware_retriever, None, None, None


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
    collection: str | None = None,
    use_enhanced: bool = True,
) -> ChatResult:
    """Chat with the RAG system.

    Parameters
    ----------
    session_id : str
        Chat session ID
    message : str
        User message
    top_k : int | None
        Number of documents to retrieve
    config_path : Path | None
        Config file path
    collection : str | None
        Collection to search
    use_enhanced : bool
        Use enhanced retrieval (HyDE + hybrid + reranking). Default True.

    Returns
    -------
    ChatResult
        Chat result with answer, sources, and history
    """
    if not session_id:
        logger.error("Chat called without session_id")
        raise ValueError("session_id must be provided for chat interactions.")

    logger.info(f"Chat message received for session {session_id}: {message[:100]}...")
    settings = get_settings(config_path)
    session_history = _memory_store.get_session(session_id)
    history_messages = session_history.messages

    if use_enhanced:
        # Enhanced retrieval: use enhanced retrieve directly
        doc_chain, _, settings_obj, k_value, coll = _build_conversational_components(
            settings, top_k=top_k, collection=collection, use_enhanced=True
        )
        documents = _enhanced_retrieve(
            message, settings_obj, top_k=k_value, collection=coll, use_hyde=False
        )
        answer, _ = doc_chain.combine_docs(
            documents, input=message, chat_history=history_messages
        )
    else:
        # Standard retrieval
        doc_chain, history_aware_retriever, _, _, _ = _build_conversational_components(
            settings, top_k=top_k, collection=collection, use_enhanced=False
        )
        retrieved = history_aware_retriever.invoke(
            {"input": message, "chat_history": history_messages}
        )
        documents = list(retrieved) if retrieved else []
        answer, _ = doc_chain.combine_docs(
            documents, input=message, chat_history=history_messages
        )

    session_history.add_user_message(message)
    session_history.add_ai_message(answer)

    serialized_history = serialize_history(session_history)
    sources: List[Document] = documents
    logger.info(
        "Chat response generated with %s sources, history length: %s",
        len(sources),
        len(serialized_history),
    )
    return ChatResult(
        session_id=session_id,
        answer=answer,
        sources=sources,
        history=serialized_history,
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
