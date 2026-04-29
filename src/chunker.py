"""
Automatic text chunking for RAG.

Strategy (in order of preference):
1. Structure-aware: detect headings and split at section boundaries.
2. Paragraph-aware: group paragraphs until target size is reached.
3. Hard split: when no structure is detectable, split by character count.

All strategies apply overlap between consecutive chunks.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .config import Config
from .models import Chunk, ExtractionMethod

# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------

# Matches lines that look like section headings.
# Covers: numbered sections (1., 1.1., I., A.), ALL-CAPS short lines,
# and common academic headings.
_HEADING_PATTERNS = [
    re.compile(r"^(\d+\.)+\s+\S"),                  # 1. / 1.1. / 1.1.1.
    re.compile(r"^[IVX]+\.\s+\S"),                  # Roman numerals
    re.compile(r"^[A-Z][A-Z\s\-:]{4,60}$"),         # ALL CAPS line
    re.compile(r"^(Abstract|Introduction|"
               r"Methods?|Results?|Discussion|"
               r"Conclusion|References?|"
               r"Acknowledgem\w+|Appendix)\b",
               re.IGNORECASE),
]

_PAGE_BREAK = re.compile(r"\[PAGE_BREAK\]")


def chunk_text(
    normalized_text: str,
    normalized_pages: List[str],
    document_id: str,
    source_file: str,
    extraction_method: str,
    quality_score: float,
    config: Optional[Config] = None,
) -> List[Chunk]:
    """
    Produce a list of Chunk objects from normalized document text.
    Tries structure-aware first; falls back to paragraph/hard-split.
    """
    cfg = config or Config()
    now = datetime.now(timezone.utc).isoformat()

    # Build page-start mapping (offset in normalised full_text → page number)
    page_offsets = _build_page_offsets(normalized_pages, normalized_text)

    # Attempt structure-aware chunking first
    sections = _split_by_headings(normalized_text)
    if len(sections) >= 3:
        raw_chunks = _sections_to_raw_chunks(sections, cfg)
    else:
        raw_chunks = _paragraph_chunks(normalized_text, cfg)

    # Apply overlap
    raw_chunks_with_overlap = _apply_overlap(raw_chunks, cfg)

    # Build Chunk objects
    chunks: List[Chunk] = []
    for idx, (text, heading) in enumerate(raw_chunks_with_overlap):
        text = text.strip()
        if len(text) < cfg.min_chunk_size:
            continue

        # Determine page range
        page_start, page_end = _estimate_page_range(text, normalized_text, page_offsets)

        chunk = Chunk(
            chunk_id=str(uuid.uuid4()),
            document_id=document_id,
            source_file=source_file,
            text=text,
            page_start=page_start,
            page_end=page_end,
            chunk_index=idx,
            section_title=heading,
            extraction_method=extraction_method,
            quality_score=quality_score,
            processing_status="chunked",
            created_at=now,
        )
        chunks.append(chunk)

    # Re-index after filtering
    for i, chunk in enumerate(chunks):
        chunk.chunk_index = i

    return chunks


# ---------------------------------------------------------------------------
# Structure-aware splitting
# ---------------------------------------------------------------------------

def _split_by_headings(text: str) -> List[Tuple[str, Optional[str]]]:
    """
    Return list of (section_text, heading_title) pairs.
    If no headings found, returns [(full_text, None)].
    """
    lines = text.splitlines(keepends=True)
    sections: List[Tuple[str, Optional[str]]] = []
    current_heading: Optional[str] = None
    current_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped and _is_heading(stripped):
            if current_lines:
                sections.append(("".join(current_lines), current_heading))
            current_heading = stripped[:200]  # cap heading length
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append(("".join(current_lines), current_heading))

    return sections if sections else [(text, None)]


def _is_heading(line: str) -> bool:
    if len(line) > 150 or len(line) < 3:
        return False
    return any(p.match(line) for p in _HEADING_PATTERNS)


# ---------------------------------------------------------------------------
# Section → raw chunk conversion
# ---------------------------------------------------------------------------

def _sections_to_raw_chunks(
    sections: List[Tuple[str, Optional[str]]],
    cfg: Config,
) -> List[Tuple[str, Optional[str]]]:
    """
    Convert sections (which may be very long or very short) into
    appropriately-sized raw chunks.  Each raw chunk keeps its heading.
    """
    raw: List[Tuple[str, Optional[str]]] = []

    for section_text, heading in sections:
        if len(section_text) <= cfg.max_chunk_size:
            raw.append((section_text, heading))
        else:
            # Section is too big: split it by paragraphs
            sub = _paragraph_chunks(section_text, cfg)
            for i, (text, _) in enumerate(sub):
                raw.append((text, heading if i == 0 else None))

    return raw


# ---------------------------------------------------------------------------
# Paragraph-based chunking
# ---------------------------------------------------------------------------

def _paragraph_chunks(
    text: str,
    cfg: Config,
) -> List[Tuple[str, Optional[str]]]:
    """
    Group consecutive paragraphs until target chunk size is reached.
    """
    # Split on double newlines, page breaks, or page-break markers
    paragraphs = [
        p.strip()
        for p in re.split(r"\n{2,}|\[PAGE_BREAK\]", text)
        if p.strip()
    ]

    chunks: List[Tuple[str, Optional[str]]] = []
    current_parts: List[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)

        # Single paragraph is already over max → hard-split it
        if para_len > cfg.max_chunk_size:
            if current_parts:
                chunks.append(("\n\n".join(current_parts), None))
                current_parts, current_len = [], 0
            for sub in _hard_split(para, cfg.chunk_size):
                chunks.append((sub, None))
            continue

        if current_len + para_len > cfg.chunk_size and current_parts:
            chunks.append(("\n\n".join(current_parts), None))
            current_parts, current_len = [], 0

        current_parts.append(para)
        current_len += para_len

    if current_parts:
        chunks.append(("\n\n".join(current_parts), None))

    return chunks or [(text, None)]


def _hard_split(text: str, size: int) -> List[str]:
    """Split text into chunks of `size` chars at word boundaries."""
    words = text.split()
    parts, current = [], []
    current_len = 0

    for word in words:
        wl = len(word) + 1
        if current_len + wl > size and current:
            parts.append(" ".join(current))
            current, current_len = [], 0
        current.append(word)
        current_len += wl

    if current:
        parts.append(" ".join(current))

    return parts


# ---------------------------------------------------------------------------
# Overlap application
# ---------------------------------------------------------------------------

def _apply_overlap(
    chunks: List[Tuple[str, Optional[str]]],
    cfg: Config,
) -> List[Tuple[str, Optional[str]]]:
    """
    Add a suffix from the previous chunk as a prefix to the current chunk.
    This creates context overlap without duplicating full chunks.
    """
    if cfg.chunk_overlap <= 0 or len(chunks) < 2:
        return chunks

    result: List[Tuple[str, Optional[str]]] = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_text = chunks[i - 1][0]
        current_text, heading = chunks[i]

        # Take last `overlap` chars from previous chunk (at word boundary)
        overlap_text = _tail_words(prev_text, cfg.chunk_overlap)
        if overlap_text:
            current_text = overlap_text + "\n" + current_text

        result.append((current_text, heading))

    return result


def _tail_words(text: str, max_chars: int) -> str:
    """Return the last `max_chars` characters of `text`, trimmed to a word boundary."""
    if len(text) <= max_chars:
        return text
    tail = text[-max_chars:]
    # Trim to first space to avoid cutting mid-word
    space_idx = tail.find(" ")
    if space_idx > 0:
        tail = tail[space_idx + 1:]
    return tail


# ---------------------------------------------------------------------------
# Page range estimation
# ---------------------------------------------------------------------------

def _build_page_offsets(normalized_pages: List[str], full_text: str = "") -> List[int]:
    """
    Return start offsets for each page in the whitespace-normalised full_text.
    Uses [PAGE_BREAK] markers so offsets are exact regardless of _final_cleanup.
    """
    if not full_text:
        return []
    full_norm = _normalize_ws(full_text)
    marker = _normalize_ws("[PAGE_BREAK]")
    offsets = [0]
    pos = 0
    while True:
        idx = full_norm.find(marker, pos)
        if idx == -1:
            break
        offsets.append(idx + len(marker))
        pos = idx + 1
    return offsets


def _normalize_ws(text: str) -> str:
    """Collapse all whitespace runs to a single space for fuzzy matching."""
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Specialised chunk builders (tables, figures, equations)
# ---------------------------------------------------------------------------

def build_specialized_chunks(
    tables: List[Dict[str, Any]],
    figures: List[Dict[str, Any]],
    equations: List[Dict[str, Any]],
    document_id: str,
    source_file: str,
    config: Optional[Config] = None,
) -> List[Chunk]:
    """
    Build one indexable Chunk per table, figure, and equation.

    These chunks complement regular text chunks in ChromaDB so that
    semantic search can surface structured content alongside prose.
    """
    cfg = config or Config()
    now = datetime.now(timezone.utc).isoformat()
    chunks: List[Chunk] = []
    idx = 0

    for table in tables:
        text = _format_table_chunk(table, source_file)
        if len(text) < cfg.min_chunk_size:
            continue
        chunks.append(Chunk(
            chunk_id=str(uuid.uuid4()),
            document_id=document_id,
            source_file=source_file,
            text=text,
            page_start=table.get("page", 1),
            page_end=table.get("page", 1),
            chunk_index=idx,
            section_title=table.get("label"),
            extraction_method="table_extractor",
            quality_score=1.0,
            processing_status="chunked",
            created_at=now,
            chunk_type="table",
            source_id=table.get("table_id"),
        ))
        idx += 1

    for fig in figures:
        text = _format_figure_chunk(fig, source_file)
        if len(text) < cfg.min_chunk_size:
            continue
        chunks.append(Chunk(
            chunk_id=str(uuid.uuid4()),
            document_id=document_id,
            source_file=source_file,
            text=text,
            page_start=fig.get("page", 1),
            page_end=fig.get("page", 1),
            chunk_index=idx,
            section_title=fig.get("label"),
            extraction_method="image_extractor",
            quality_score=1.0,
            processing_status="chunked",
            created_at=now,
            chunk_type="figure",
            source_id=fig.get("figure_id"),
        ))
        idx += 1

    for eq in equations:
        text = _format_equation_chunk(eq, source_file)
        if len(text) < cfg.min_chunk_size:
            continue
        chunks.append(Chunk(
            chunk_id=str(uuid.uuid4()),
            document_id=document_id,
            source_file=source_file,
            text=text,
            page_start=eq.get("page", 1),
            page_end=eq.get("page", 1),
            chunk_index=idx,
            section_title=eq.get("label"),
            extraction_method="equation_extractor",
            quality_score=0.9,
            processing_status="chunked",
            created_at=now,
            chunk_type="equation",
            source_id=eq.get("equation_id"),
        ))
        idx += 1

    return chunks


def _format_table_chunk(table: Dict[str, Any], source_file: str) -> str:
    label   = table.get("label") or "Table"
    page    = table.get("page", "?")
    caption = table.get("caption") or ""
    md      = table.get("markdown") or ""
    lines = [f"[{label}]", f"Paper: {source_file}", f"Page: {page}"]
    if caption and caption != label:
        lines.append(f"Caption: {caption}")
    if md:
        lines.append("Markdown:")
        lines.append(md)
    return "\n".join(lines)


def _format_figure_chunk(fig: Dict[str, Any], source_file: str) -> str:
    label   = fig.get("label") or "Figure"
    page    = fig.get("page", "?")
    caption = fig.get("caption") or ""
    surr    = fig.get("surrounding_text") or ""
    lines = [f"[{label}]", f"Paper: {source_file}", f"Page: {page}"]
    if caption:
        lines.append(f"Caption: {caption}")
    if surr:
        lines.append("Related text:")
        lines.append(surr)
    return "\n".join(lines)


def _format_equation_chunk(eq: Dict[str, Any], source_file: str) -> str:
    label = eq.get("label") or "Equation"
    page  = eq.get("page", "?")
    text  = eq.get("text") or ""
    surr  = eq.get("surrounding_text") or ""
    lines = [f"[{label}]", f"Paper: {source_file}", f"Page: {page}"]
    if text:
        lines.append("Equation:")
        lines.append(text)
    if surr:
        lines.append("Related text:")
        lines.append(surr)
    return "\n".join(lines)


def _estimate_page_range(
    chunk_text: str,
    full_text: str,
    page_offsets: List[int],
) -> Tuple[int, int]:
    """
    Find the page range for a chunk by locating it in the full text.
    Searches on whitespace-normalised strings so newline/space divergence
    introduced by the paragraph joiner does not cause misses.
    page_offsets must be positions in the *normalised* full_text.
    Falls back to (1, 1) if not found.
    """
    if not page_offsets:
        return 1, 1

    full_norm = _normalize_ws(full_text)

    # Try probes at increasing offsets to skip over the overlap prefix
    pos = -1
    for start in range(0, min(len(chunk_text), 400), 50):
        probe = _normalize_ws(chunk_text[start:start + 120])
        if probe:
            pos = full_norm.find(probe)
            if pos != -1:
                break

    if pos == -1:
        return 1, 1

    chunk_end = pos + len(_normalize_ws(chunk_text))
    n_pages = len(page_offsets)

    page_start = 1
    page_end = n_pages

    for i, offset in enumerate(page_offsets):
        if offset <= pos:
            page_start = i + 1
        if offset <= chunk_end:
            page_end = i + 1

    return page_start, page_end
