#!/usr/bin/env python3
"""
MCP server for the PDF RAG knowledge base.

Exposes these tools to Claude:
  - search_papers     : semantic search over ingested papers
  - get_paper_info    : full metadata for a specific document
  - list_papers       : list all papers in the knowledge base
  - get_paper_chunks  : retrieve all chunks from a paper (for deep reading)

Run this server by registering it in Claude Code settings (see README).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure the project root is on the Python path
sys.path.insert(0, str(Path(__file__).parent))

from mcp.server.fastmcp import FastMCP

from src.chromadb_index import (
    get_collection_stats,
    is_available,
    list_documents,
    semantic_search,
)
from src.config import load_config
from src.storage import load_chunks, load_report
import src.rag_service as _rag

# ── Server setup ──────────────────────────────────────────────────────────────

mcp = FastMCP(
    "pdf-knowledge-base",
    instructions=(
        "Knowledge base of scientific papers and books. "
        "Use search_papers to find relevant passages before writing. "
        "Always cite the source_file, authors, year, and doi when referencing content. "
        "Never paraphrase beyond what the retrieved text actually says."
    ),
)
_cfg = load_config()


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def search_papers(query: str, n_results: int = 5) -> str:
    """
    Semantic search over the scientific paper knowledge base.

    Use this tool BEFORE writing any claim that should be supported by
    literature.  Returns the most relevant text passages with full citation
    information so you can reference them accurately.

    Args:
        query:     Natural-language search query (e.g. "attention mechanism
                   transformer self-attention").
        n_results: Number of results to return (default 5, max 20).

    Returns:
        Formatted string with ranked results, each containing:
        - citation key and full citation
        - document section
        - the actual text passage
        - relevance distance (lower = more similar)
    """
    if not is_available():
        return (
            "ChromaDB is not available. "
            "Run: pip install chromadb sentence-transformers\n"
            "Then re-ingest your documents."
        )

    n_results = max(1, min(n_results, 20))
    hits = semantic_search(query, _cfg, n_results=n_results)

    if not hits:
        return (
            f"No results found for query: '{query}'\n"
            "Make sure documents have been ingested with: "
            "python cli.py process-folder ./input"
        )

    lines = [f"Search results for: \"{query}\"\n{'─'*60}"]

    for i, hit in enumerate(hits, 1):
        citation_key = hit.get("citation_key") or ""
        title        = hit.get("title") or hit.get("source_file", "Unknown")
        authors      = hit.get("authors", "")
        year         = hit.get("year", "")
        doi          = hit.get("doi", "")
        section      = hit.get("section_title", "")
        page_start   = hit.get("page_start", "?")
        page_end     = hit.get("page_end", "?")
        source_file  = hit.get("source_file", "")
        distance     = hit.get("distance", 0)
        text         = hit.get("text", "").strip()

        # Build citation line
        citation_parts = []
        if authors:
            citation_parts.append(authors)
        if year:
            citation_parts.append(f"({year})")
        if title:
            citation_parts.append(f'"{title}"')
        if doi:
            citation_parts.append(f"DOI: {doi}")
        citation = ". ".join(citation_parts) or source_file

        lines.append(f"\n[{i}] {citation_key or '?'} — relevance: {1 - distance:.2f}")
        lines.append(f"    Citation : {citation}")
        if section:
            lines.append(f"    Section  : {section}")
        lines.append(f"    Pages    : {page_start}–{page_end} ({source_file})")
        lines.append(f"    Text     :\n{_indent(text, 4)}")
        lines.append("─" * 60)

    return "\n".join(lines)


@mcp.tool()
def get_paper_info(document_id: str) -> str:
    """
    Return full metadata and processing report for a specific document.

    Args:
        document_id: The UUID assigned during ingestion (from list_papers or
                     search_papers results).

    Returns:
        JSON-formatted metadata including title, authors, year, DOI,
        abstract, quality score, and chunk count.
    """
    report = load_report(document_id, _cfg)
    if not report:
        return f"No report found for document_id: {document_id}"

    # Also pull from ChromaDB for academic metadata
    docs = list_documents(_cfg)
    chroma_meta = next((d for d in docs if d["document_id"] == document_id), {})

    combined = {**report, **{k: v for k, v in chroma_meta.items() if v}}
    return json.dumps(combined, indent=2, ensure_ascii=False)


@mcp.tool()
def list_papers() -> str:
    """
    List all scientific papers and books currently in the knowledge base.

    Returns a formatted table with:
    - Citation key
    - Authors and year
    - Title
    - DOI
    - Source file name

    Use this to discover what literature is available before writing.
    """
    if not is_available():
        return "ChromaDB not available. Install chromadb and re-ingest documents."

    docs = list_documents(_cfg)
    if not docs:
        return (
            "Knowledge base is empty.\n"
            "Add papers with: python cli.py process-folder ./input"
        )

    lines = [f"Knowledge base — {len(docs)} document(s)\n{'─'*70}"]
    for doc in docs:
        key    = doc.get("citation_key") or "?"
        title  = doc.get("title") or doc.get("source_file", "?")
        authors = doc.get("authors", "")
        year   = doc.get("year", "")
        doi    = doc.get("doi", "")
        fid    = doc.get("document_id", "")

        lines.append(f"\n[{key}]")
        lines.append(f"  Title   : {title}")
        if authors:
            lines.append(f"  Authors : {authors}")
        if year:
            lines.append(f"  Year    : {year}")
        if doi:
            lines.append(f"  DOI     : https://doi.org/{doi}")
        lines.append(f"  ID      : {fid}")

    return "\n".join(lines)


@mcp.tool()
def get_paper_chunks(document_id: str, max_chunks: int = 20) -> str:
    """
    Retrieve the text chunks from a specific paper in order.

    Use this for deep reading of a paper — for example, to get the full
    Methods section, or to read the paper from beginning to end.

    Args:
        document_id: The document UUID (from list_papers or search_papers).
        max_chunks:  Maximum chunks to return (default 20).

    Returns:
        Ordered chunks with section titles and page numbers.
    """
    chunks = load_chunks(document_id, _cfg)
    if not chunks:
        return f"No chunks found for document_id: {document_id}"

    chunks_sorted = sorted(chunks, key=lambda c: c.get("chunk_index", 0))
    chunks_sorted = chunks_sorted[:max_chunks]

    source = chunks_sorted[0].get("source_file", "?") if chunks_sorted else "?"
    lines = [f"Chunks from: {source}  ({len(chunks_sorted)} of {len(chunks)} total)\n{'─'*60}"]

    for c in chunks_sorted:
        idx     = c.get("chunk_index", "?")
        section = c.get("section_title", "")
        p_start = c.get("page_start", "?")
        p_end   = c.get("page_end", "?")
        text    = c.get("text", "").strip()

        header = f"\n[Chunk {idx}]"
        if section:
            header += f"  § {section}"
        header += f"  (pp. {p_start}–{p_end})"
        lines.append(header)
        lines.append(_indent(text, 2))

    if len(chunks) > max_chunks:
        lines.append(f"\n... {len(chunks) - max_chunks} more chunks. Increase max_chunks to see them.")

    return "\n".join(lines)


@mcp.tool()
def knowledge_base_stats() -> str:
    """
    Show statistics about the knowledge base: number of documents,
    chunks, and whether embeddings are active.
    """
    stats = get_collection_stats(_cfg)
    lines = ["Knowledge base statistics", "─" * 40]
    for k, v in stats.items():
        lines.append(f"  {k:<22}: {v}")
    return "\n".join(lines)


# ── RAG tools (search + LLM answer, built on rag_service) ────────────────────

@mcp.tool()
def rag_search(question: str, top_k: int = 5) -> str:
    """
    Semantic search over indexed papers. Returns ranked chunks with metadata.

    Does NOT call the LLM — pure retrieval only. Use this to inspect evidence
    before deciding whether to call rag_answer.

    Args:
        question: Natural-language question or search query.
        top_k:    Number of chunks to retrieve (default 5, max 20).

    Returns:
        Ranked list of chunks with paper, page, section, distance, and a
        text preview for each result.
    """
    top_k = max(1, min(top_k, 20))

    try:
        hits = _rag.search(question, top_k=top_k, config=_cfg)
    except RuntimeError as exc:
        return f"[rag_search] Error: {exc}"

    if not hits:
        return (
            f"[rag_search] No se encontraron resultados para: {question!r}\n"
            "Comprueba que los papers han sido indexados con: "
            "python cli.py process-folder <carpeta>"
        )

    sources = _rag._hits_to_sources(hits)
    lines = [f'[rag_search] "{question}"  (top-{top_k})', "=" * 64]

    for s in sources:
        key = s["citation_key"] or s["paper_id"] or "?"
        title = s["title"] or s["paper_id"] or "?"
        dist = s["distance"]
        relevance = f"{1 - dist:.3f}" if dist is not None else "?"
        lines.append(f"\n[{s['rank']}] {key}  —  relevancia: {relevance}")
        if title and title != key:
            lines.append(f"    Titulo   : {title}")
        if s["authors"]:
            year_str = f"  ({s['year']})" if s["year"] else ""
            lines.append(f"    Autores  : {s['authors']}{year_str}")
        page_str = str(s["page"]) if s["page"] is not None else "?"
        lines.append(f"    Pagina   : {page_str}")
        if s["section"]:
            lines.append(f"    Seccion  : {s['section']}")
        lines.append(f"    Preview  : {s['text_preview']}")
        lines.append("-" * 64)

    return "\n".join(lines)


@mcp.tool()
def rag_answer(question: str, top_k: int = 5) -> str:
    """
    [OPCIONAL — requiere Anthropic API billing, no Claude Pro]

    Full RAG pipeline: retrieve relevant chunks then generate a grounded answer
    by calling the Anthropic API directly. This incurs token costs independent
    of any Claude Pro subscription.

    For most use cases, prefer rag_sources to retrieve evidence and then answer
    in the conversation using your existing Claude session — no API costs.

    Args:
        question: The question to answer based on indexed papers.
        top_k:    Number of chunks to retrieve as context (default 5).

    Returns:
        LLM-generated answer with in-text citations, followed by a source list.
        Returns a clear error if ANTHROPIC_API_KEY is missing or the index is empty.
    """
    top_k = max(1, min(top_k, 20))
    result = _rag.answer(question, top_k=top_k, config=_cfg)

    lines = [f'[rag_answer] "{question}"', "=" * 64]

    if result["warning"] and not result["answer"]:
        lines.append(f"\nAdvertencia: {result['warning']}")
        return "\n".join(lines)

    if result["warning"]:
        lines.append(f"Advertencia: {result['warning']}\n")

    lines.append(result["answer"])

    if result["sources"]:
        lines.append("\n" + "=" * 64)
        lines.append(f"Fuentes utilizadas ({len(result['sources'])} fragmentos):")
        for s in result["sources"]:
            key = s["citation_key"] or s["paper_id"] or "?"
            page_str = f"  p. {s['page']}" if s["page"] is not None else ""
            sec_str = f"  §  {s['section']}" if s["section"] else ""
            lines.append(f"  [{s['rank']}] {key}{page_str}{sec_str}")

    return "\n".join(lines)


@mcp.tool()
def rag_sources(question: str, top_k: int = 5) -> str:
    """
    Retrieve and display the evidence sources for a question — without calling
    the LLM. Use this to inspect what evidence is available before answering.

    Args:
        question: The question or topic to find sources for.
        top_k:    Number of sources to retrieve (default 5).

    Returns:
        Detailed source list: paper, authors, year, page, section, chunk preview.
    """
    top_k = max(1, min(top_k, 20))

    try:
        hits = _rag.search(question, top_k=top_k, config=_cfg)
    except RuntimeError as exc:
        return f"[rag_sources] Error: {exc}"

    if not hits:
        return (
            f"[rag_sources] No se encontraron fuentes para: {question!r}\n"
            "Asegurate de que los papers han sido indexados."
        )

    sources = _rag._hits_to_sources(hits)
    lines = [f'[rag_sources] Evidencia para: "{question}"', "=" * 64]

    for s in sources:
        key = s["citation_key"] or s["paper_id"] or "?"
        lines.append(f"\n[{s['rank']}] {key}")
        if s["title"]:
            lines.append(f"    Titulo   : {s['title']}")
        if s["authors"]:
            lines.append(f"    Autores  : {s['authors']}")
        if s["year"]:
            lines.append(f"    Anio     : {s['year']}")
        page_str = str(s["page"]) if s["page"] is not None else "?"
        lines.append(f"    Pagina   : {page_str}")
        if s["section"]:
            lines.append(f"    Seccion  : {s['section']}")
        dist = s["distance"]
        if dist is not None:
            lines.append(f"    Distancia: {dist:.4f}  (relevancia: {1-dist:.3f})")
        if s["chunk_id"]:
            lines.append(f"    Chunk ID : {s['chunk_id']}")
        lines.append(f"    Preview  :\n      {s['text_preview']}")
        lines.append("-" * 64)

    return "\n".join(lines)


@mcp.tool()
def rag_status() -> str:
    """
    Show the current status of the RAG system: index size, embeddings model,
    ChromaDB path, and whether ANTHROPIC_API_KEY is configured.

    Use this first to verify the system is ready before calling other RAG tools.
    """
    st = _rag.status(config=_cfg)

    yes_no = lambda v: "Si" if v else "No"  # noqa: E731

    lines = ["[rag_status]", "=" * 64]
    lines.append(f"  ChromaDB disponible : {yes_no(st['chroma_available'])}")
    lines.append(f"  Documentos          : {st['document_count']}")
    lines.append(f"  Chunks indexados    : {st['total_chunks']}")
    lines.append(f"  Modelo embeddings   : {st['embeddings_model']}")
    lines.append(f"  ANTHROPIC_API_KEY   : {'Configurada' if st['anthropic_api_key_set'] else 'NO configurada'}")
    lines.append(f"  Ruta ChromaDB       : {st['chroma_dir']}")
    lines.append("=" * 64)

    if st["warnings"]:
        lines.append("Advertencias:")
        for w in st["warnings"]:
            lines.append(f"  - {w}")
    else:
        lines.append("Sin advertencias. Sistema listo.")

    return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _indent(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines())


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
