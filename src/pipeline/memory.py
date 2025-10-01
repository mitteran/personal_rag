from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Dict, List

import psycopg
from langchain_core.chat_history import BaseChatMessageHistory
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
        self._connection = self._create_connection()
        self._ensure_table()
        logger.info("ChatMemoryStore initialized with persistent PostgreSQL storage")

    def _create_connection(self) -> psycopg.Connection:
        """Create a persistent database connection for chat history."""
        conn_str = self._settings.database.psycopg_connection_string
        logger.debug(f"Creating database connection for chat history")
        return psycopg.connect(conn_str, autocommit=True)

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
                logger.debug(f"Creating new persistent session: {session_id}")
                history = PostgresChatMessageHistory(
                    self._table_name,
                    session_id,
                    sync_connection=self._connection,
                )
                self._store[session_id] = history
            return history

    def drop_session(self, session_id: str) -> None:
        with self._lock:
            logger.info(f"Dropping session: {session_id}")
            history = self._store.pop(session_id, None)
            if history and isinstance(history, PostgresChatMessageHistory):
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
