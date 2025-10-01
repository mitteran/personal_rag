from __future__ import annotations

from pathlib import Path

import typer

from src.pipeline.service import MissingOpenAIKeyError, ingest_corpus, query


app = typer.Typer(help="Utilities for the local RAG system.")


@app.command()
def ingest(
    content_dir: Path = typer.Argument(..., help="Directory with source documents."),
    reindex: bool = typer.Option(False, "--reindex", "-r", help="Drop and rebuild the vector collection."),
    config: Path = typer.Option(Path("config/settings.yaml"), "--config", help="Path to the YAML configuration file."),
):
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
    top_k: int = typer.Option(None, "--top-k", help="Override number of retrieved chunks."),
    config: Path = typer.Option(Path("config/settings.yaml"), "--config", help="Path to the YAML configuration file."),
):
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
