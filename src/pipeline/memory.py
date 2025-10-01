from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Dict, List

import psycopg
from psycopg import Error as PsycopgError
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_postgres import PostgresChatMessageHistory

from src.logging_config import get_logger
from src.settings import Settings, get_settings

logger = get_logger(__name__)


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


class ChatMemoryStore:
    """Thread-safe registry for persistent chat session histories.

    Uses PostgreSQL to persist chat history across application restarts.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._store: Dict[str, BaseChatMessageHistory] = {}
        self._lock = Lock()
        self._table_name = "chat_message_history"
        self._connection = self._initialise_connection()
        if self._connection is not None:
            self._ensure_table()
            logger.info("ChatMemoryStore initialized with persistent PostgreSQL storage")
        else:
            logger.warning(
                "ChatMemoryStore falling back to in-memory storage (database unavailable)"
            )

    def _initialise_connection(self) -> psycopg.Connection | None:
        """Create a persistent database connection for chat history."""
        conn_str = self._settings.database.psycopg_connection_string
        logger.debug("Creating database connection for chat history")
        try:
            return psycopg.connect(conn_str, autocommit=True)
        except PsycopgError as exc:
            logger.error("Failed to connect to chat history database: %s", exc)
            return None

    def _ensure_table(self) -> None:
        """Create the chat message history table if it doesn't exist."""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self._table_name} (
            id SERIAL PRIMARY KEY,
            session_id UUID NOT NULL,
            message JSONB NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_{self._table_name}_session_id
            ON {self._table_name} (session_id);
        """
        with self._connection.cursor() as cursor:
            cursor.execute(create_table_sql)
        logger.info(f"Chat history table '{self._table_name}' ensured")

    def get_session(self, session_id: str) -> BaseChatMessageHistory:
        with self._lock:
            history = self._store.get(session_id)
            if history is None:
                if self._connection is not None:
                    logger.debug(f"Creating new persistent session: {session_id}")
                    history = PostgresChatMessageHistory(
                        self._table_name,
                        session_id,
                        sync_connection=self._connection,
                    )
                else:
                    logger.debug(f"Creating new in-memory session: {session_id}")
                    history = InMemoryChatMessageHistory()
                self._store[session_id] = history
            return history

    def drop_session(self, session_id: str) -> None:
        with self._lock:
            logger.info(f"Dropping session: {session_id}")
            history = self._store.pop(session_id, None)
            if (
                history
                and self._connection is not None
                and isinstance(history, PostgresChatMessageHistory)
            ):
                # Clear the messages from the database
                history.clear()

    def list_sessions(self) -> List[str]:
        with self._lock:
            return list(self._store.keys())

    def close(self) -> None:
        """Close the database connection."""
        if self._connection and not self._connection.closed:
            logger.debug("Closing database connection for chat history")
            self._connection.close()


def serialize_history(history: BaseChatMessageHistory) -> List[ChatMessage]:
    messages = []
    for message in history.messages:
        messages.append(ChatMessage(role=_resolve_role(message), content=message.content))
    return messages


def _resolve_role(message: BaseMessage) -> str:
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    return "system"


__all__ = ["ChatMessage", "ChatMemoryStore", "serialize_history"]
