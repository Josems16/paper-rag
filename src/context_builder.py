"""
Convert semantic_search results into structured context blocks for the LLM prompt.
"""

from __future__ import annotations

from typing import Any, Dict, List


def build_context(results: List[Dict[str, Any]]) -> str:
    """
    Format a list of semantic_search hits into numbered context blocks.

    Each block includes whichever metadata fields are present in the hit dict:
    citation_key, document_id, title, authors, year, page_start/end, section_title.
    Missing fields are silently omitted.
    """
    if not results:
        return ""

    blocks: List[str] = []

    for i, hit in enumerate(results, start=1):
        lines: List[str] = [f"[Fragmento {i}]"]

        # Identifier: prefer citation_key, fall back to document_id
        if hit.get("citation_key"):
            lines.append(f"Clave: {hit['citation_key']}")
        elif hit.get("document_id"):
            lines.append(f"Documento: {hit['document_id']}")

        if hit.get("title"):
            lines.append(f"Título: {hit['title']}")
        if hit.get("authors"):
            lines.append(f"Autores: {hit['authors']}")
        if hit.get("year"):
            lines.append(f"Año: {hit['year']}")

        page_start = hit.get("page_start")
        page_end = hit.get("page_end")
        if page_start is not None:
            if page_end and int(page_end) != int(page_start):
                lines.append(f"Páginas: {page_start}-{page_end}")
            else:
                lines.append(f"Página: {page_start}")

        if hit.get("section_title"):
            lines.append(f"Sección: {hit['section_title']}")

        text = hit.get("text", "").strip()
        lines.append(f"Texto:\n{text}")

        blocks.append("\n".join(lines))

    return "\n\n---\n\n".join(blocks)
