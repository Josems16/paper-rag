"""
Cross-reference linker: connects text chunks to figures, tables, and equations.

After chunking and extraction:
  1. Scans each chunk's text for in-text mentions (Fig. 1, Table 2, Eq. (3)).
  2. Writes normalised label strings into chunk.linked_figures /
     linked_tables / linked_equations.
  3. Updates figure/table/equation dicts with mentioned_in_chunks lists.

All mutations are in-place. No new files are created here; storage handles
persisting the updated data.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from .models import Chunk

# ── Mention patterns ──────────────────────────────────────────────────────────

# Figures: Fig. 1, Fig 1, Figure 1, Figura 1, Fig. 1.2
_FIG_MENTION = re.compile(
    r'\b(fig(?:ure|ura?)?\.?\s*\d+(?:\.\d+)?)',
    re.IGNORECASE,
)
# Tables: Table 1, Tabla 1, Tab. 2, Tab 3, Table II (Roman)
_TABLE_MENTION = re.compile(
    r'\b(tab(?:le|la?)?\.?\s*(?:[IVX]+|\d+))',
    re.IGNORECASE,
)
# Equations: Eq. (1), Eqs. (1), Equation (2), Ecuación (3), bare (1) only
# when preceded by context words
_EQ_MENTION = re.compile(
    r'\b(eq(?:uation)?s?\.?\s*\(\s*\d{1,3}[a-z]?\s*\)'
    r'|ecuaci[oó]n\s*\(\s*\d{1,3}[a-z]?\s*\))',
    re.IGNORECASE,
)


# ── Public API ────────────────────────────────────────────────────────────────

def link_chunks_to_elements(
    chunks: List[Chunk],
    figures: List[Dict[str, Any]],
    tables: List[Dict[str, Any]],
    equations: List[Dict[str, Any]],
) -> None:
    """
    Mutate chunks and element dicts in-place to add cross-reference fields.

    After this call:
      - chunk.linked_figures  contains normalised label strings
      - chunk.linked_tables   contains normalised label strings
      - chunk.linked_equations contains normalised label strings
      - figure["mentioned_in_chunks"]  contains chunk_ids
      - table["mentioned_in_chunks"]   contains chunk_ids
      - equation["mentioned_in_chunks"] contains chunk_ids
    """
    # Index elements by normalised label for reverse lookup
    fig_ids_by_label  = _index_by_label(figures,   "label", "figure_id")
    tab_ids_by_label  = _index_by_label(tables,    "label", "table_id")
    eq_ids_by_label   = _index_by_label(equations, "label", "equation_id")

    # Forward pass: enrich chunks
    for chunk in chunks:
        text = chunk.text

        fig_labels = {_norm(m) for m in _FIG_MENTION.findall(text)}
        tab_labels = {_norm(m) for m in _TABLE_MENTION.findall(text)}
        eq_labels  = {_norm(m) for m in _EQ_MENTION.findall(text)}

        chunk.linked_figures   = sorted(fig_labels)
        chunk.linked_tables    = sorted(tab_labels)
        chunk.linked_equations = sorted(eq_labels)

    # Reverse pass: populate mentioned_in_chunks on each element
    _fill_mentioned(chunks, figures,   "linked_figures",   "figure_id",   "mentioned_in_chunks")
    _fill_mentioned(chunks, tables,    "linked_tables",    "table_id",    "mentioned_in_chunks")
    _fill_mentioned(chunks, equations, "linked_equations", "equation_id", "mentioned_in_chunks")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _norm(label: str) -> str:
    """Normalise a label string for consistent comparison."""
    return re.sub(r'\s+', ' ', label).strip().lower()


def _index_by_label(
    elements: List[Dict[str, Any]],
    label_key: str,
    id_key: str,
) -> Dict[str, str]:
    """Build {normalised_label: element_id} dict."""
    idx: Dict[str, str] = {}
    for elem in elements:
        lbl = elem.get(label_key)
        eid = elem.get(id_key)
        if lbl and eid:
            idx[_norm(str(lbl))] = str(eid)
    return idx


def _fill_mentioned(
    chunks: List[Chunk],
    elements: List[Dict[str, Any]],
    chunk_link_attr: str,         # e.g. "linked_figures"
    elem_id_key: str,             # e.g. "figure_id"
    elem_mention_key: str,        # e.g. "mentioned_in_chunks"
) -> None:
    """
    For each element, find which chunks reference its label and record chunk_ids.
    """
    # Build element id → list of mentioning chunk_ids
    mention_map: Dict[str, List[str]] = {}

    for chunk in chunks:
        linked_labels: List[str] = getattr(chunk, chunk_link_attr, [])
        for lbl in linked_labels:
            # We don't have a direct label→id map here, so we tag by label
            mention_map.setdefault(lbl, []).append(chunk.chunk_id)

    # Write back to element dicts — match by normalised label
    for elem in elements:
        lbl = elem.get("label")
        if not lbl:
            continue
        norm_lbl = _norm(str(lbl))
        chunk_ids = mention_map.get(norm_lbl, [])
        elem[elem_mention_key] = chunk_ids
