from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Dict, List

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


class ChatMemoryStore:
    """Thread-safe registry for chat session histories."""

    def __init__(self) -> None:
        self._store: Dict[str, BaseChatMessageHistory] = {}
        self._lock = Lock()

    def get_session(self, session_id: str) -> BaseChatMessageHistory:
        with self._lock:
            history = self._store.get(session_id)
            if history is None:
                history = ChatMessageHistory()
                self._store[session_id] = history
            return history

    def drop_session(self, session_id: str) -> None:
        with self._lock:
            self._store.pop(session_id, None)

    def list_sessions(self) -> List[str]:
        with self._lock:
            return list(self._store.keys())


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
