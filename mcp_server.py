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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _indent(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines())


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
