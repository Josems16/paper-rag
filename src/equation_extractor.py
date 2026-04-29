"""
Equation detection from PDF text.

Uses text-based heuristics on the raw extracted text (by page):
  1. Lines with equation labels: (1), (2), Eq. (1), Equation (1), Ecuación (1)
  2. Lines with high density of Unicode math symbols.
  3. Lines with dimensionless-number / thermal keywords (COP, NTU, Re, Nu, …).

Does NOT call any external API or LLM.
Detection is best-effort; undetected equations are normal for complex PDFs.

Image crops are NOT generated in this implementation — image_path is always None.
If line-level bboxes become available (via a PyMuPDF dict pass), this can be
extended to produce crops.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# ── Detection patterns ────────────────────────────────────────────────────────

# "(1)", "(12)", "(3a)" at end of line — canonical equation numbering
_EQ_NUMBER_END = re.compile(r'\(\s*\d{1,3}[a-z]?\s*\)\s*$')

# Explicit labels: Eq. (1), Eqs. (1)–(3), Equation (2), Ecuación (3)
_EQ_EXPLICIT = re.compile(
    r'\b(eq(?:uation)?s?\.?\s*\(\s*\d{1,3}[a-z]?\s*\)'
    r'|ecuaci[oó]n\s*\(\s*\d{1,3}[a-z]?\s*\))',
    re.IGNORECASE,
)

# Dimensionless numbers and thermal/fluid keywords common in HT papers
_THERMAL_KEYWORDS = re.compile(
    r'\b(COP|NTU|Re|Nu|Pr|Gr|Bi|Fo|Ra|We|Pe|Le|Sh|EER|DIEC|EEI)\b'
)

# High-density math: Unicode math symbols (Greek, operators, special)
_MATH_CHARS = frozenset(
    'αβγδεζηθικλμνξοπρστυφχψω'   # lowercase Greek
    'ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ'   # uppercase Greek
    '∑∫∂∇±×÷√∞≈≠≤≥∝∈∉⊂⊃∪∩'       # math operators
    '⁰¹²³⁴⁵⁶⁷⁸⁹⁻'               # superscripts
)
MIN_MATH_DENSITY = 0.10   # fraction of non-space chars that are math symbols

# Context window (lines before/after) for surrounding_text
CONTEXT_LINES = 2

# Minimum line length to consider as equation content
MIN_LINE_LEN = 5


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class EquationRecord:
    equation_id: str
    paper_id: str
    page: int                        # 1-based
    label: Optional[str]             # "(1)" / "Eq. (2)" / None
    text: str                        # the equation line(s)
    image_path: Optional[str]        # always None in this implementation
    surrounding_text: Optional[str]  # few lines of context

    def to_dict(self) -> Dict[str, Any]:
        return {
            "equation_id": self.equation_id,
            "paper_id": self.paper_id,
            "page": self.page,
            "label": self.label,
            "text": self.text,
            "image_path": self.image_path,
            "surrounding_text": self.surrounding_text,
            "mentioned_in_chunks": [],   # filled by linker.py
        }


# ── Public API ────────────────────────────────────────────────────────────────

def extract_equations(
    text_by_page: Sequence[str],
    paper_id: str,
) -> List[EquationRecord]:
    """
    Detect equations from the per-page raw text.

    Args:
        text_by_page: list of page texts (index 0 = page 1).
        paper_id:     document_id from pipeline.

    Returns:
        List of EquationRecord; empty list if nothing detected.
    """
    records: List[EquationRecord] = []

    for page_idx, page_text in enumerate(text_by_page):
        page_num = page_idx + 1
        page_records = _detect_page_equations(page_text, page_num, paper_id)
        records.extend(page_records)

    # Deduplicate by (page, label) — keep first occurrence
    seen: set = set()
    unique: List[EquationRecord] = []
    for rec in records:
        key = (rec.page, rec.label or rec.text[:40])
        if key not in seen:
            seen.add(key)
            unique.append(rec)

    logger.info("Detected %d equation(s) across %d page(s)", len(unique), len(text_by_page))
    return unique


# ── Per-page detection ────────────────────────────────────────────────────────

def _detect_page_equations(
    page_text: str,
    page_num: int,
    paper_id: str,
) -> List[EquationRecord]:
    lines = page_text.splitlines()
    records: List[EquationRecord] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if len(stripped) < MIN_LINE_LEN:
            continue

        label = _extract_label(stripped)
        is_eq = (
            label is not None
            or _has_math_density(stripped)
            or _has_thermal_keyword(stripped)
        )
        if not is_eq:
            continue

        surrounding = _get_surrounding(lines, i, CONTEXT_LINES, stripped)

        records.append(EquationRecord(
            equation_id=str(uuid.uuid4()),
            paper_id=paper_id,
            page=page_num,
            label=label,
            text=stripped,
            image_path=None,
            surrounding_text=surrounding or None,
        ))

    return records


# ── Detection helpers ─────────────────────────────────────────────────────────

def _extract_label(line: str) -> Optional[str]:
    """Return the equation label token if found, else None."""
    # Check for explicit "Eq. (1)" style
    m = _EQ_EXPLICIT.search(line)
    if m:
        return m.group(1).strip()
    # Check for trailing "(1)" numbering
    m2 = _EQ_NUMBER_END.search(line)
    if m2:
        return m2.group(0).strip()
    return None


def _has_math_density(line: str) -> bool:
    """True if a meaningful fraction of chars are Unicode math symbols."""
    non_space = [c for c in line if not c.isspace()]
    if len(non_space) < 8:
        return False
    math_count = sum(1 for c in non_space if c in _MATH_CHARS)
    return math_count / len(non_space) >= MIN_MATH_DENSITY


def _has_thermal_keyword(line: str) -> bool:
    """True if the line contains common thermal/dimensionless-number symbols."""
    return bool(_THERMAL_KEYWORDS.search(line))


def _get_surrounding(
    lines: List[str],
    idx: int,
    window: int,
    current_line: str,
) -> str:
    """Return the CONTEXT_LINES lines before and after idx, excluding current."""
    start = max(0, idx - window)
    end   = min(len(lines), idx + window + 1)
    ctx = [
        lines[j].strip()
        for j in range(start, end)
        if j != idx and lines[j].strip() and lines[j].strip() != current_line
    ]
    return " | ".join(ctx)[:300] if ctx else ""
