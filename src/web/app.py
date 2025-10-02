from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.logging_config import get_logger, setup_logging
from src.pipeline.service import (
    MissingOpenAIKeyError,
    chat as chat_with_memory,
    start_chat_session,
)

logger = get_logger(__name__)

DEFAULT_CONFIG = Path("config/settings.yaml")

app = FastAPI(title="RAG Chatbot", version="0.1.0")


@app.on_event("startup")
async def startup_event():
    """Initialize logging on application startup."""
    setup_logging("WARNING")
    logger.info("RAG Chatbot API starting up")


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    top_k: int | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[str]
    history: List[dict[str, str]]


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(
        """
        <!DOCTYPE html>
        <html lang=\"en\">
        <head>
            <meta charset=\"utf-8\">
            <title>RAG Chatbot</title>
            <style>
                body { font-family: Arial, sans-serif; background: #f4f4f7; margin: 0; padding: 0; }
                .container { max-width: 720px; margin: 0 auto; padding: 24px; }
                h1 { color: #333; }
                #chat { background: #fff; border-radius: 8px; padding: 16px; height: 480px; overflow-y: auto; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
                .message { margin-bottom: 16px; }
                .message.you { text-align: right; }
                .message.you .bubble { display: inline-block; background: #465dd0; color: #fff; padding: 12px 16px; border-radius: 16px 16px 0 16px; }
                .message.bot .bubble { display: inline-block; background: #e5e7ff; color: #333; padding: 12px 16px; border-radius: 16px 16px 16px 0; }
                .sources { font-size: 0.85rem; color: #555; margin-top: 4px; }
                form { display: flex; gap: 8px; margin-top: 16px; }
                input[type=text] { flex: 1; padding: 12px; border: 1px solid #ccc; border-radius: 8px; }
                button { padding: 12px 20px; border: none; border-radius: 8px; background: #465dd0; color: #fff; cursor: pointer; }
                button:hover { background: #3b4fb8; }
                #clear { margin-top: 12px; background: transparent; color: #465dd0; border: 1px solid #465dd0; }
                #clear:hover { background: #465dd0; color: white; }
                .error { color: #c0392b; margin-top: 12px; }
            </style>
        </head>
        <body>
            <div class=\"container\">
                <h1>RAG Chatbot</h1>
                <div id=\"chat\"></div>
                <form id=\"chat-form\">
                    <input type=\"text\" id=\"message\" placeholder=\"Ask anything...\" autocomplete=\"off\" required />
                    <button type=\"submit\">Send</button>
                </form>
                <button id=\"clear\">Start New Session</button>
                <div id=\"error\" class=\"error\"></div>
            </div>
            <script>
                const chat = document.getElementById('chat');
                const form = document.getElementById('chat-form');
                const input = document.getElementById('message');
                const clearButton = document.getElementById('clear');
                const errorBox = document.getElementById('error');
                let sessionId = null;

                function appendMessage(role, content, sources=[]) {
                    const wrapper = document.createElement('div');
                    wrapper.className = `message ${role === 'user' ? 'you' : 'bot'}`;

                    const bubble = document.createElement('div');
                    bubble.className = 'bubble';
                    bubble.textContent = content;
                    wrapper.appendChild(bubble);

                    if (role === 'assistant' && sources.length) {
                        const sourceList = document.createElement('div');
                        sourceList.className = 'sources';
                        sourceList.textContent = `Sources: ${sources.join(', ')}`;
                        wrapper.appendChild(sourceList);
                    }

                    chat.appendChild(wrapper);
                    chat.scrollTop = chat.scrollHeight;
                }

                async function sendMessage(message) {
                    const payload = { message };
                    if (sessionId) {
                        payload.session_id = sessionId;
                    }

                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });

                    if (!response.ok) {
                        const detail = await response.json().catch(() => ({ detail: 'Unknown error' }));
                        throw new Error(detail.detail || 'Request failed');
                    }

                    return response.json();
                }

                form.addEventListener('submit', async (event) => {
                    event.preventDefault();
                    errorBox.textContent = '';
                    const message = input.value.trim();
                    if (!message) {
                        return;
                    }

                    appendMessage('user', message);
                    input.value = '';
                    input.disabled = true;

                    try {
                        const data = await sendMessage(message);
                        sessionId = data.session_id;
                        appendMessage('assistant', data.answer, data.sources);
                    } catch (error) {
                        errorBox.textContent = error.message;
                    } finally {
                        input.disabled = false;
                        input.focus();
                    }
                });

                clearButton.addEventListener('click', () => {
                    sessionId = null;
                    chat.innerHTML = '';
                    errorBox.textContent = '';
                    input.focus();
                });
            </script>
        </body>
        </html>
        """,
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    session_id = payload.session_id or start_chat_session()
    logger.info(f"API chat request for session {session_id}")

    try:
        result = chat_with_memory(
            session_id=session_id,
            message=payload.message,
            top_k=payload.top_k,
            config_path=DEFAULT_CONFIG,
        )
    except MissingOpenAIKeyError as exc:  # pragma: no cover - configuration issue surfaced to client
        logger.error(f"Missing OpenAI API key for session {session_id}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Chat error for session {session_id}: {exc}")
        raise

    sources = [doc.metadata.get("source", "unknown") for doc in result.sources]
    history = [asdict(message) for message in result.history]
    logger.info(f"API chat response for session {session_id}: {len(sources)} sources")
    return ChatResponse(
        session_id=session_id,
        answer=result.answer,
        sources=sources,
        history=history,
    )


@app.get("/healthz")
def healthcheck() -> dict[str, str]:
    logger.debug("Health check requested")
    return {"status": "ok"}


__all__ = ["app", "DEFAULT_CONFIG"]
