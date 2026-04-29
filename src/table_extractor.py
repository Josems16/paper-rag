"""
Table extraction from PDFs using pdfplumber.

For each page:
  1. Detect tables with pdfplumber's extract_tables() (lattice + stream).
  2. Convert each table to Markdown, CSV-ready lists, and JSON-serialisable form.
  3. Detect caption by searching text blocks near the table bbox.
  4. Return TableRecord objects ready for storage.

Soft dependency: pdfplumber. If unavailable, extract_tables() returns [].
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import pdfplumber as _pdfplumber
    _PDFPLUMBER_AVAILABLE = True
except ImportError:
    _PDFPLUMBER_AVAILABLE = False
    logger.debug("pdfplumber not available; table extraction disabled")

# Caption patterns — Table 1, Tabla 1, TABLE I, Tab. 2, Tab 3, etc.
_TABLE_CAPTION_RE = re.compile(
    r"(?:^|\n)\s*(tab(?:le|la?)?\.?\s*(?:[IVX]+|\d+)(?:[.:]\s*[^\n]{0,200})?)",
    re.IGNORECASE,
)
_TABLE_LABEL_RE = re.compile(
    r"^\s*(tab(?:le|la?)?\.?\s*(?:[IVX]+|\d+))",
    re.IGNORECASE,
)

CAPTION_GAP = 60   # points above/below table to search for caption
SURROUND_GAP = 80  # additional context window


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class TableRecord:
    table_id: str
    paper_id: str
    page: int
    label: Optional[str]            # "Table 1", "Tabla 2"
    caption: Optional[str]          # full caption text including label
    headers: List[str]              # first row treated as headers
    rows: List[List[Optional[str]]] # remaining rows
    markdown: str                   # Markdown representation
    bbox: Optional[Tuple[float, float, float, float]]  # (x0,y0,x1,y1) in pts
    surrounding_text: Optional[str]
    csv_path: Optional[str] = None   # set after storage
    json_path: Optional[str] = None  # set after storage

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table_id": self.table_id,
            "paper_id": self.paper_id,
            "page": self.page,
            "label": self.label,
            "caption": self.caption,
            "headers": self.headers,
            "rows": self.rows,
            "markdown": self.markdown,
            "bbox": list(self.bbox) if self.bbox else None,
            "surrounding_text": self.surrounding_text,
            "csv_path": self.csv_path,
            "json_path": self.json_path,
            "mentioned_in_chunks": [],   # filled by linker.py
        }


# ── Public API ────────────────────────────────────────────────────────────────

def extract_tables(pdf_path: str | Path, paper_id: str) -> List[TableRecord]:
    """
    Extract tables from all pages of a PDF.

    Returns a list of TableRecord objects; empty list if pdfplumber is
    unavailable or no tables are found.
    """
    if not _PDFPLUMBER_AVAILABLE:
        logger.warning("pdfplumber not available; table extraction skipped")
        return []

    path = Path(pdf_path)
    records: List[TableRecord] = []

    try:
        with _pdfplumber.open(str(path)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_records = _extract_page_tables(page, page_num, paper_id)
                records.extend(page_records)
    except Exception as exc:
        logger.warning("Table extraction failed for %s: %s", path.name, exc)

    logger.info(
        "Extracted %d table(s) from %s", len(records), path.name
    )
    return records


# ── Per-page extraction ───────────────────────────────────────────────────────

def _extract_page_tables(
    page: Any,   # pdfplumber.Page
    page_num: int,
    paper_id: str,
) -> List[TableRecord]:
    """Extract all tables from a single page."""
    records: List[TableRecord] = []

    # pdfplumber returns raw tables as List[List[Optional[str]]]
    try:
        raw_tables = page.extract_tables() or []
    except Exception as exc:
        logger.debug("extract_tables() error on page %d: %s", page_num, exc)
        return []

    # Get bboxes for each detected table
    try:
        table_objects = page.find_tables() or []
    except Exception:
        table_objects = []

    # Also grab all text blocks for caption search
    try:
        words = page.extract_words() or []
        page_text = page.extract_text() or ""
    except Exception:
        words = []
        page_text = ""

    for i, raw_table in enumerate(raw_tables):
        if not raw_table or len(raw_table) < 2:
            # Ignore single-row "tables" (likely just a line)
            continue

        # Get corresponding bbox if available
        bbox_tuple: Optional[Tuple[float, float, float, float]] = None
        if i < len(table_objects):
            try:
                tb = table_objects[i].bbox   # (x0, top, x1, bottom)
                bbox_tuple = (tb[0], tb[1], tb[2], tb[3])
            except Exception:
                pass

        # Clean cell values
        cleaned = _clean_table(raw_table)
        if not cleaned:
            continue

        headers = cleaned[0]
        rows = cleaned[1:]

        # Skip if all cells are empty
        if not any(any(c for c in row) for row in cleaned):
            continue

        markdown = _to_markdown(headers, rows)
        caption, label = _find_caption(bbox_tuple, page_text, page)
        surrounding = _get_surrounding_text(bbox_tuple, page_text, caption, page)

        records.append(TableRecord(
            table_id=str(uuid.uuid4()),
            paper_id=paper_id,
            page=page_num,
            label=label,
            caption=caption,
            headers=headers,
            rows=rows,
            markdown=markdown,
            bbox=bbox_tuple,
            surrounding_text=surrounding,
        ))

    return records


# ── Caption detection ─────────────────────────────────────────────────────────

def _find_caption(
    bbox: Optional[Tuple],
    page_text: str,
    page: Any,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Find a caption for a table.

    Strategy:
      1. If bbox is known, search the page text for 'Table N' or 'Tabla N'
         in vertical proximity to the table.
      2. Otherwise fall back to a plain regex search in the full page text.

    Returns (full_caption, label).
    """
    # Strategy 1: proximity-based (requires pdfplumber words with bboxes)
    if bbox is not None:
        table_top, table_bottom = bbox[1], bbox[3]
        try:
            # Check text blocks above and below the table
            words = page.extract_words(x_tolerance=3, y_tolerance=3) or []
            # Group words into lines by top-y proximity
            lines = _words_to_lines(words)
            for line_y, line_text in lines:
                # Look in a zone around the table
                if (table_top - CAPTION_GAP <= line_y <= table_top + 5 or
                        table_bottom - 5 <= line_y <= table_bottom + CAPTION_GAP):
                    m = _TABLE_CAPTION_RE.match(line_text.strip())
                    if m:
                        cap = re.sub(r"\s+", " ", m.group(1)).strip()
                        lbl_m = _TABLE_LABEL_RE.match(cap)
                        lbl = lbl_m.group(1).strip() if lbl_m else None
                        return cap, lbl
        except Exception:
            pass

    # Strategy 2: plain regex over full page text
    m = _TABLE_CAPTION_RE.search(page_text)
    if m:
        cap = re.sub(r"\s+", " ", m.group(1)).strip()
        lbl_m = _TABLE_LABEL_RE.match(cap)
        lbl = lbl_m.group(1).strip() if lbl_m else None
        return cap, lbl

    return None, None


def _get_surrounding_text(
    bbox: Optional[Tuple],
    page_text: str,
    caption: Optional[str],
    page: Any,
) -> Optional[str]:
    """
    Return up to ~400 chars of text near the table (excluding the caption).
    """
    if bbox is None:
        return None

    table_top, table_bottom = bbox[1], bbox[3]
    try:
        words = page.extract_words(x_tolerance=3, y_tolerance=3) or []
        lines = _words_to_lines(words)
        context_lines: List[str] = []
        for line_y, line_text in lines:
            if (table_top - SURROUND_GAP <= line_y <= table_top or
                    table_bottom <= line_y <= table_bottom + SURROUND_GAP):
                txt = line_text.strip()
                if not txt or len(txt) < 15:
                    continue
                cap_norm = re.sub(r"\s+", " ", caption or "").strip()
                if cap_norm and txt.startswith(cap_norm[:30]):
                    continue
                context_lines.append(txt)
        if context_lines:
            combined = " … ".join(context_lines)
            return combined[:400] if len(combined) > 400 else combined
    except Exception:
        pass
    return None


# ── Markdown conversion ───────────────────────────────────────────────────────

def _to_markdown(headers: List[str], rows: List[List[Optional[str]]]) -> str:
    """Convert table headers + rows to a GitHub-flavoured Markdown table."""
    def _cell(v: Optional[str]) -> str:
        return (v or "").replace("|", "\\|").replace("\n", " ").strip()

    if not headers:
        return ""

    header_row = "| " + " | ".join(_cell(h) for h in headers) + " |"
    sep_row    = "| " + " | ".join("---" for _ in headers) + " |"
    data_rows  = [
        "| " + " | ".join(_cell(c) for c in (row + [""] * max(0, len(headers) - len(row)))) + " |"
        for row in rows
    ]
    return "\n".join([header_row, sep_row] + data_rows)


# ── Table cleaning ────────────────────────────────────────────────────────────

def _clean_table(
    raw: List[List[Optional[str]]],
) -> List[List[Optional[str]]]:
    """
    Normalise cell whitespace and remove fully-empty rows.
    """
    cleaned: List[List[Optional[str]]] = []
    for row in raw:
        cleaned_row = [
            re.sub(r"\s+", " ", cell).strip() if cell else None
            for cell in row
        ]
        # Skip rows where every cell is None or empty
        if any(c for c in cleaned_row):
            cleaned.append(cleaned_row)
    return cleaned


# ── Line grouping helper ──────────────────────────────────────────────────────

def _words_to_lines(words: List[Dict]) -> List[Tuple[float, str]]:
    """
    Group pdfplumber word dicts into (top_y, line_text) tuples.
    Words are grouped when their top-y values are within 3 pts.
    """
    if not words:
        return []

    lines: List[Tuple[float, str]] = []
    current_y: Optional[float] = None
    current_words: List[str] = []

    for w in sorted(words, key=lambda x: (x.get("top", 0), x.get("x0", 0))):
        top = w.get("top", 0)
        text = w.get("text", "")
        if current_y is None or abs(top - current_y) > 3:
            if current_words:
                lines.append((current_y, " ".join(current_words)))
            current_y = top
            current_words = [text]
        else:
            current_words.append(text)

    if current_words and current_y is not None:
        lines.append((current_y, " ".join(current_words)))

    return lines
