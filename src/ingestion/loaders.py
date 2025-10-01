from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Dict, List

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_community.document_loaders.epub import UnstructuredEPubLoader
from langchain_core.documents import Document

from src.logging_config import get_logger

logger = get_logger(__name__)


def _load_text(path: Path) -> List[Document]:
    logger.debug(f"Loading text file: {path}")
    loader = TextLoader(str(path), encoding="utf-8")
    docs = loader.load()
    logger.debug(f"Loaded {len(docs)} documents from {path}")
    return docs


def _load_pdf(path: Path) -> List[Document]:
    logger.debug(f"Loading PDF file: {path}")
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    logger.debug(f"Loaded {len(docs)} pages from {path}")
    return docs


def _load_epub(path: Path) -> List[Document]:
    logger.debug(f"Loading EPUB file: {path}")
    loader = UnstructuredEPubLoader(str(path))
    docs = loader.load()
    logger.debug(f"Loaded {len(docs)} documents from {path}")
    return docs


LOADERS: Dict[str, callable] = {
    ".txt": _load_text,
    ".md": _load_text,
    ".pdf": _load_pdf,
    ".epub": _load_epub,
}


def supported_files(root: Path) -> Iterable[Path]:
    for suffix in LOADERS:
        yield from root.rglob(f"*{suffix}")


def load_documents(root: Path) -> List[Document]:
    logger.info(f"Scanning for documents in {root}")
    documents: List[Document] = []
    files = sorted(set(supported_files(root)))
    logger.info(f"Found {len(files)} supported files")

    for path in files:
        loader = LOADERS.get(path.suffix.lower())
        if not loader:
            logger.warning(f"No loader for {path.suffix} files, skipping {path}")
            continue
        try:
            docs = loader(path)
            for doc in docs:
                doc.metadata.setdefault("source", str(path))
            documents.extend(docs)
        except Exception as exc:
            logger.error(f"Failed to load {path}: {exc}")
            raise

    logger.info(f"Successfully loaded {len(documents)} documents from {len(files)} files")
    return documents
