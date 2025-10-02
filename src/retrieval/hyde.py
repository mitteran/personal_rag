"""HyDE (Hypothetical Document Embeddings) query transformation."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.logging_config import get_logger
from src.settings import Settings

logger = get_logger(__name__)


class HyDEQueryTransformer:
    """Transform queries using HyDE (Hypothetical Document Embeddings).

    HyDE generates a hypothetical answer to the query, then uses that answer
    for retrieval instead of the original query. This often improves retrieval
    quality by matching the style and content of actual documents.
    """

    def __init__(self, settings: Settings):
        self.llm = ChatOpenAI(model=settings.rag.chat_model, temperature=0.7)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert assistant. Given a question, write a detailed, "
                    "factual paragraph that would answer it. Write as if you are "
                    "extracting text directly from a reference document. "
                    "Do not use phrases like 'The answer is' or 'According to'. "
                    "Just write the content directly.",
                ),
                ("human", "{question}"),
            ]
        )

    def transform(self, query: str) -> str:
        """Transform query using HyDE.

        Parameters
        ----------
        query : str
            Original user query

        Returns
        -------
        str
            Hypothetical document (or original query if transformation fails)
        """
        try:
            chain = self.prompt | self.llm
            response = chain.invoke({"question": query})
            hypothetical_doc = response.content.strip()
            logger.debug(f"HyDE transformed query: {query[:50]} -> {hypothetical_doc[:50]}")
            return hypothetical_doc
        except Exception as e:
            logger.warning(f"HyDE transformation failed, using original query: {e}")
            return query
