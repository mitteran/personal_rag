from __future__ import annotations

from pathlib import Path

import typer

from src.logging_config import setup_logging
from src.pipeline.service import (
    MissingOpenAIKeyError,
    chat as chat_with_memory,
    ingest_corpus,
    query,
    start_chat_session,
)


app = typer.Typer(help="Utilities for the local RAG system.")


@app.command()
def ingest(
    content_dir: Path = typer.Argument(..., help="Directory with source documents."),
    reindex: bool = typer.Option(
        False, "--reindex", "-r", help="Drop and rebuild the vector collection."
    ),
    config: Path = typer.Option(
        Path("config/settings.yaml"),
        "--config",
        help="Path to the YAML configuration file.",
    ),
    log_level: str = typer.Option(
        "INFO", "--log-level", help="Logging level (DEBUG, INFO, WARNING, ERROR)"
    ),
):
    """Load documents, chunk them, and persist embeddings in the vector store.

    Parameters
    ----------
    content_dir:
        Directory containing the raw corpus files to ingest.
    reindex:
        When ``True``, drop any existing collection before rebuilding it.
    config:
        Path to the settings file with database and embedding configuration.
    log_level:
        Logging verbosity level.
    """
    setup_logging(log_level)
    try:
        chunk_count = ingest_corpus(content_dir, reindex=reindex, config_path=config)
    except MissingOpenAIKeyError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # pragma: no cover - surface unexpected errors
        typer.secho(f"Ingestion failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.secho(f"Ingested {chunk_count} chunks into pgvector.", fg=typer.colors.GREEN)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Natural language query."),
    top_k: int = typer.Option(
        None, "--top-k", help="Override number of retrieved chunks."
    ),
    config: Path = typer.Option(
        Path("config/settings.yaml"),
        "--config",
        help="Path to the YAML configuration file.",
    ),
    log_level: str = typer.Option(
        "INFO", "--log-level", help="Logging level (DEBUG, INFO, WARNING, ERROR)"
    ),
):
    """Run a single-shot query against the indexed corpus and print the answer.

    Parameters
    ----------
    question:
        Natural-language prompt passed to the retriever and LLM.
    top_k:
        Optional override for how many context chunks to retrieve.
    config:
        Path to the YAML settings file to override the default configuration.
    log_level:
        Logging verbosity level.
    """
    setup_logging(log_level)
    try:
        answer, sources = query(question, top_k=top_k, config_path=config)
    except MissingOpenAIKeyError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # pragma: no cover
        typer.secho(f"Query failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.echo(answer)
    if sources:
        typer.secho("\nSources:", fg=typer.colors.BLUE)
        for doc in sources:
            source = doc.metadata.get("source", "unknown")
            typer.echo(f"- {source}")


@app.command()
def chat(
    config: Path = typer.Option(
        Path("config/settings.yaml"),
        "--config",
        help="Path to the YAML configuration file.",
    ),
    top_k: int = typer.Option(
        None, "--top-k", help="Override number of retrieved chunks."
    ),
    session_id: str = typer.Option(
        None, "--session-id", help="Resume an existing chat session by ID."
    ),
    log_level: str = typer.Option(
        "INFO", "--log-level", help="Logging level (DEBUG, INFO, WARNING, ERROR)"
    ),
):
    """Start an interactive chat session backed by the local retrieval pipeline.

    Parameters
    ----------
    config:
        Path to the YAML configuration file to load pipeline settings.
    top_k:
        Optional override for how many retrieval chunks to supply per turn.
    session_id:
        Optional session ID to resume a previous conversation.
    log_level:
        Logging verbosity level.
    """
    setup_logging(log_level)
    try:
        if session_id is None:
            session_id = start_chat_session()
            typer.secho(f"Created new session: {session_id}", fg=typer.colors.GREEN)
        else:
            typer.secho(f"Resuming session: {session_id}", fg=typer.colors.GREEN)
    except Exception as exc:  # pragma: no cover - defensive guard
        typer.secho(f"Failed to initialise chat session: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.secho(
        "Starting interactive chat. Type 'exit' or press Ctrl+C to quit.",
        fg=typer.colors.BLUE,
    )

    while True:
        try:
            message = typer.prompt("You")
        except typer.Abort:
            typer.secho("\nExiting chat.", fg=typer.colors.BLUE)
            break

        if message.strip().lower() in {"exit", "quit"}:
            typer.secho("Goodbye!", fg=typer.colors.BLUE)
            break

        try:
            result = chat_with_memory(
                session_id=session_id,
                message=message,
                top_k=top_k,
                config_path=config,
            )
        except MissingOpenAIKeyError as exc:
            typer.secho(str(exc), fg=typer.colors.RED)
            raise typer.Exit(code=1) from exc
        except (
            Exception
        ) as exc:  # pragma: no cover - unexpected failure surfaced to user
            typer.secho(f"Chat failed: {exc}", fg=typer.colors.RED)
            continue

        typer.secho(f"Assistant: {result.answer}", fg=typer.colors.GREEN)
        if result.sources:
            typer.secho("Sources:", fg=typer.colors.BLUE)
            for doc in result.sources:
                source = doc.metadata.get("source", "unknown")
                typer.echo(f"- {source}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
