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


def _load_text(path: Path) -> List[Document]:
    loader = TextLoader(str(path), encoding="utf-8")
    return loader.load()


def _load_pdf(path: Path) -> List[Document]:
    loader = PyPDFLoader(str(path))
    return loader.load()


def _load_epub(path: Path) -> List[Document]:
    loader = UnstructuredEPubLoader(str(path))
    return loader.load()


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
    documents: List[Document] = []
    for path in sorted(set(supported_files(root))):
        loader = LOADERS.get(path.suffix.lower())
        if not loader:
            continue
        docs = loader(path)
        for doc in docs:
            doc.metadata.setdefault("source", str(path))
        documents.extend(docs)
    return documents
