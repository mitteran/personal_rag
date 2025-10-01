from src.pipeline.memory import ChatMemoryStore, serialize_history


def test_memory_store_persists_messages() -> None:
    store = ChatMemoryStore()
    session_id = "session-1"

    history = store.get_session(session_id)
    history.add_user_message("hi")
    history.add_ai_message("hello")

    same_history = store.get_session(session_id)
    serialized = serialize_history(same_history)

    assert [message.content for message in serialized] == ["hi", "hello"]
    assert [message.role for message in serialized] == ["user", "assistant"]


def test_memory_store_reset() -> None:
    store = ChatMemoryStore()
    session_id = "session-2"

    history = store.get_session(session_id)
    history.add_user_message("ping")
    store.drop_session(session_id)

    new_history = store.get_session(session_id)
    assert serialize_history(new_history) == []
