"""
ChromaDB-based semantic index for the PDF RAG pipeline.

Stores all chunks with their metadata in a persistent ChromaDB collection.
Supports semantic (embedding-based) search and exact lookups.

ChromaDB handles embedding computation internally via sentence-transformers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Config
from .models import Chunk

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False
    logger.warning(
        "chromadb not installed. Run: pip install chromadb sentence-transformers"
    )

COLLECTION_NAME = "papers"

# Module-level cache to avoid re-creating client/collection on every call
_client_cache: Dict[str, Any] = {}


# ── Public API ────────────────────────────────────────────────────────────────

def is_available() -> bool:
    return _CHROMA_AVAILABLE


def add_to_chroma(
    chunks: List[Chunk],
    academic_meta: Optional[Any],   # AcademicMetadata | None
    config: Config,
) -> bool:
    """
    Add a document's chunks to the ChromaDB collection.
    Returns True on success.
    """
    if not _CHROMA_AVAILABLE:
        logger.warning("ChromaDB unavailable; skipping semantic index.")
        return False
    if not chunks:
        return True

    collection = _get_collection(config)
    if collection is None:
        return False

    ids, documents, metadatas = [], [], []

    for chunk in chunks:
        if not chunk.text.strip():
            continue

        meta = _build_chroma_metadata(chunk, academic_meta)
        ids.append(chunk.chunk_id)
        documents.append(chunk.text)
        metadatas.append(meta)

    if not ids:
        return True

    try:
        # upsert handles re-processing the same document gracefully
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        logger.info("ChromaDB: upserted %d chunks", len(ids))
        return True
    except Exception as exc:
        logger.error("ChromaDB upsert failed: %s", exc)
        return False


def semantic_search(
    query: str,
    config: Config,
    n_results: int = 5,
    where: Optional[Dict] = None,
) -> List[Dict[str, Any]]:
    """
    Run a semantic (embedding-based) query against the collection.

    Returns a list of result dicts, each containing:
      text, distance, and all stored metadata fields.
    """
    if not _CHROMA_AVAILABLE:
        return []

    collection = _get_collection(config)
    if collection is None:
        return []

    try:
        kwargs: Dict[str, Any] = {
            "query_texts": [query],
            "n_results": min(n_results, max(collection.count(), 1)),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = collection.query(**kwargs)
    except Exception as exc:
        logger.error("ChromaDB query failed: %s", exc)
        return []

    hits = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, dists):
        hit = {"text": doc, "distance": round(dist, 4)}
        hit.update(meta)
        hits.append(hit)

    return hits


def document_exists(document_id: str, config: Config) -> bool:
    """Return True if any chunk for this document_id is already indexed."""
    if not _CHROMA_AVAILABLE:
        return False
    collection = _get_collection(config)
    if collection is None:
        return False
    try:
        result = collection.get(where={"document_id": document_id}, limit=1)
        return len(result.get("ids", [])) > 0
    except Exception:
        return False


def delete_document(document_id: str, config: Config) -> int:
    """Remove all chunks for a document. Returns number of chunks removed."""
    if not _CHROMA_AVAILABLE:
        return 0

    collection = _get_collection(config)
    if collection is None:
        return 0

    try:
        existing = collection.get(where={"document_id": document_id})
        ids_to_delete = existing.get("ids", [])
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)
    except Exception as exc:
        logger.error("ChromaDB delete failed: %s", exc)
        return 0


def get_collection_stats(config: Config) -> Dict[str, Any]:
    """Return basic statistics about the ChromaDB collection."""
    if not _CHROMA_AVAILABLE:
        return {"available": False, "total_chunks": 0}

    collection = _get_collection(config)
    if collection is None:
        return {"available": True, "total_chunks": 0, "error": "could not open collection"}

    try:
        count = collection.count()
        # Sample a few records to find unique documents
        sample = collection.peek(min(count, 100))
        doc_ids = set(
            m.get("document_id", "")
            for m in (sample.get("metadatas") or [])
        )
        return {
            "available": True,
            "total_chunks": count,
            "document_count": len(doc_ids),
            "embedding_model": config.embeddings_model,
        }
    except Exception as exc:
        return {"available": True, "error": str(exc)}


def list_documents(config: Config) -> List[Dict[str, Any]]:
    """
    Return one metadata record per unique document in the collection.
    Useful for listing the knowledge base contents.
    """
    if not _CHROMA_AVAILABLE:
        return []

    collection = _get_collection(config)
    if collection is None:
        return []

    try:
        count = collection.count()
        if count == 0:
            return []

        # Fetch all metadata (no embeddings/documents to keep it light)
        all_meta = collection.get(include=["metadatas"])["metadatas"] or []

        # Deduplicate by document_id
        seen: set = set()
        docs = []
        for meta in all_meta:
            doc_id = meta.get("document_id", "")
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                docs.append({
                    "document_id": doc_id,
                    "source_file": meta.get("source_file", ""),
                    "title": meta.get("title", ""),
                    "authors": meta.get("authors", ""),
                    "year": meta.get("year", ""),
                    "doi": meta.get("doi", ""),
                    "citation_key": meta.get("citation_key", ""),
                })
        return sorted(docs, key=lambda d: d.get("title") or d.get("source_file") or "")
    except Exception as exc:
        logger.error("ChromaDB list_documents failed: %s", exc)
        return []


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_collection(config: Config):
    """Return (or create) the ChromaDB collection, with caching."""
    cache_key = str(config.chroma_dir)

    if cache_key in _client_cache:
        return _client_cache[cache_key]

    try:
        config.chroma_dir.mkdir(parents=True, exist_ok=True)

        ef = SentenceTransformerEmbeddingFunction(
            model_name=config.embeddings_model,
            device="cpu",
        )
        client = chromadb.PersistentClient(path=str(config.chroma_dir))
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        _client_cache[cache_key] = collection
        logger.debug(
            "ChromaDB collection '%s' opened at %s (%d chunks)",
            COLLECTION_NAME, config.chroma_dir, collection.count(),
        )
        return collection
    except Exception as exc:
        logger.error("Cannot open ChromaDB collection: %s", exc)
        return None


def _build_chroma_metadata(chunk: Chunk, academic_meta: Optional[Any]) -> Dict[str, str]:
    """
    Build a flat metadata dict for ChromaDB.
    ChromaDB only accepts str/int/float values (no None, no lists).
    """
    meta: Dict[str, Any] = {
        "document_id":       chunk.document_id,
        "source_file":       chunk.source_file,
        "chunk_index":       chunk.chunk_index,
        "page_start":        chunk.page_start,
        "page_end":          chunk.page_end,
        "section_title":     chunk.section_title or "",
        "extraction_method": chunk.extraction_method,
        "quality_score":     round(chunk.quality_score, 3),
        "created_at":        chunk.created_at,
    }

    if academic_meta:
        meta["title"]        = academic_meta.title or ""
        meta["authors"]      = academic_meta.authors or ""
        meta["year"]         = academic_meta.year or ""
        meta["doi"]          = academic_meta.doi or ""
        meta["citation_key"] = academic_meta.citation_key or ""
        meta["keywords"]     = academic_meta.keywords or ""

    return meta
