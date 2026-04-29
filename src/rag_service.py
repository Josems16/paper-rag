"""
Reusable RAG service layer.

Shared by query.py (CLI) and mcp_server.py (MCP server).
All public functions return structured data or raise RuntimeError with a
human-readable message — they never call sys.exit().
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .chromadb_index import get_collection_stats, is_available, semantic_search
from .config import Config, load_config
from .context_builder import build_context
from .llm_adapter import query_with_rag

_PREVIEW_LEN = 220


# ── Public API ────────────────────────────────────────────────────────────────

def search(
    question: str,
    top_k: int = 5,
    config: Optional[Config] = None,
) -> List[Dict[str, Any]]:
    """
    Semantic search over the ChromaDB index.

    Returns a list of hit dicts (text, distance, metadata).

    Raises:
        RuntimeError: ChromaDB unavailable or index empty.
    """
    cfg = config or load_config()

    if not is_available():
        raise RuntimeError(
            "ChromaDB no está disponible. "
            "Instala las dependencias con: pip install chromadb sentence-transformers"
        )

    stats = get_collection_stats(cfg)
    if stats.get("total_chunks", 0) == 0:
        raise RuntimeError(
            "El índice está vacío. Procesa primero los papers con:\n"
            "  python cli.py process-folder <carpeta_con_pdfs>"
        )

    return semantic_search(query=question, config=cfg, n_results=top_k)


def answer(
    question: str,
    top_k: int = 5,
    config: Optional[Config] = None,
) -> Dict[str, Any]:
    """
    Full RAG pipeline: search → build context → generate LLM answer.

    Always returns a dict — never raises.  Errors appear in 'warning'.

    Returns:
        {
          "answer":  str,              # LLM-generated text (empty on error)
          "sources": List[dict],       # formatted source list
          "warning": str | None,       # error message, or None on success
        }
    """
    cfg = config or load_config()

    try:
        hits = search(question, top_k=top_k, config=cfg)
    except RuntimeError as exc:
        return {"answer": "", "sources": [], "warning": str(exc)}

    if not hits:
        return {
            "answer": "",
            "sources": [],
            "warning": "No se encontraron fragmentos relevantes para esta pregunta.",
        }

    context = build_context(hits)

    try:
        text = query_with_rag(question=question, retrieved_context=context)
    except RuntimeError as exc:
        # Return sources even when the LLM call fails
        return {
            "answer": "",
            "sources": _hits_to_sources(hits),
            "warning": str(exc),
        }

    return {
        "answer": text,
        "sources": _hits_to_sources(hits),
        "warning": None,
    }


def status(config: Optional[Config] = None) -> Dict[str, Any]:
    """
    Return index and configuration status.

    Never raises.
    """
    cfg = config or load_config()

    result: Dict[str, Any] = {
        "chroma_available": is_available(),
        "chroma_dir": str(cfg.chroma_dir),
        "embeddings_model": cfg.embeddings_model,
        "anthropic_api_key_set": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "total_chunks": 0,
        "document_count": 0,
        "warnings": [],
    }

    if is_available():
        stats = get_collection_stats(cfg)
        result["total_chunks"] = stats.get("total_chunks", 0)
        result["document_count"] = stats.get("document_count", 0)
    else:
        result["warnings"].append(
            "ChromaDB no disponible: pip install chromadb sentence-transformers"
        )

    if not result["anthropic_api_key_set"]:
        result["warnings"].append(
            "ANTHROPIC_API_KEY no está definida — rag_answer no funcionará"
        )

    if result["chroma_available"] and result["total_chunks"] == 0:
        result["warnings"].append(
            "El índice está vacío — ingesta papers con: python cli.py process-folder <carpeta>"
        )

    return result


# ── Internal helpers ──────────────────────────────────────────────────────────

def _hits_to_sources(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sources = []
    for rank, hit in enumerate(hits, 1):
        text = hit.get("text", "")
        preview = text[:_PREVIEW_LEN] + ("..." if len(text) > _PREVIEW_LEN else "")
        sources.append({
            "rank":          rank,
            "chunk_id":      hit.get("chunk_id", ""),
            "paper_id":      hit.get("document_id", ""),
            "citation_key":  hit.get("citation_key", ""),
            "title":         hit.get("title", ""),
            "authors":       hit.get("authors", ""),
            "year":          hit.get("year", ""),
            "page":          hit.get("page_start"),
            "section":       hit.get("section_title") or None,
            "distance":      hit.get("distance"),
            "text_preview":  preview,
        })
    return sources
