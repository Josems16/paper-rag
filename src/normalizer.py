"""
Text normalization: clean extracted text while preserving useful structure.

Goals:
- Remove control chars and artefacts from PDF extraction
- Reconstruct hyphenated words split across lines
- Collapse excessive whitespace without destroying paragraph breaks
- Detect and strip repetitive headers/footers
- Preserve page boundaries and section structure
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_text(
    raw_text_by_page: List[str],
    strip_headers_footers: bool = True,
) -> Tuple[str, List[str]]:
    """
    Normalize a list of per-page raw strings.

    Returns:
        (full_normalized_text, normalized_pages_list)
    """
    # 1. Per-page cleaning
    cleaned_pages = [_clean_page(page_text) for page_text in raw_text_by_page]

    # 2. Strip repetitive headers/footers across pages
    if strip_headers_footers and len(cleaned_pages) > 3:
        cleaned_pages = _strip_repeated_lines(cleaned_pages)

    # 3. Join pages with a clear separator
    full_text = "\n\n[PAGE_BREAK]\n\n".join(
        p for p in cleaned_pages if p.strip()
    )

    # 4. Final global cleanup
    full_text = _final_cleanup(full_text)

    return full_text, cleaned_pages


def normalize_single_page(page_text: str) -> str:
    """Convenience wrapper for normalizing a single page."""
    normalized, _ = normalize_text([page_text], strip_headers_footers=False)
    return normalized


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_CONTROL_CHARS = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ufffd\ufffe\uffff]"
)
_FORM_FEED = re.compile(r"\f")
_HYPHENATED_LINE_BREAK = re.compile(r"-\n([a-záéíóúñüa-z])", re.MULTILINE)
_MULTIPLE_BLANK_LINES = re.compile(r"\n{3,}")
_TRAILING_SPACE = re.compile(r"[ \t]+\n")
_LEADING_SPACE_LINE = re.compile(r"\n[ \t]+")


def _clean_page(text: str) -> str:
    if not text:
        return ""

    # Remove form feeds (PyMuPDF sometimes embeds them)
    text = _FORM_FEED.sub(" ", text)

    # Strip control characters
    text = _CONTROL_CHARS.sub("", text)

    # NFKC: decomposes compatibility chars including ligatures (ﬁ→fi, ﬃ→ffi, etc.)
    text = unicodedata.normalize("NFKC", text)

    # Reconstruct hyphenated words broken at line boundaries
    # e.g. "pro-\ncessing" → "processing"
    text = _HYPHENATED_LINE_BREAK.sub(r"\1", text)

    # Collapse trailing spaces before newlines
    text = _TRAILING_SPACE.sub("\n", text)

    # Remove leading whitespace at start of lines (preserves para breaks)
    text = _LEADING_SPACE_LINE.sub("\n", text)

    # Collapse 3+ blank lines to 2
    text = _MULTIPLE_BLANK_LINES.sub("\n\n", text)

    return text.strip()


def _strip_repeated_lines(pages: List[str]) -> List[str]:
    """
    Identify lines that appear on >60% of pages (headers/footers) and remove them.
    Uses normalised line keys so page numbers don't prevent matching.
    """
    total = len(pages)
    threshold = 0.6

    # Count how often each normalised line appears
    line_counter: Counter = Counter()
    for page_text in pages:
        seen = set()
        for line in page_text.splitlines():
            stripped = line.strip()
            if not stripped or len(stripped) > 120:
                continue
            key = _normalise_key(stripped)
            if key not in seen:
                line_counter[key] += 1
                seen.add(key)

    boilerplate: set = {
        key for key, cnt in line_counter.items()
        if cnt / total >= threshold
    }

    if not boilerplate:
        return pages

    cleaned = []
    for page_text in pages:
        new_lines = []
        for line in page_text.splitlines():
            stripped = line.strip()
            if stripped and _normalise_key(stripped) in boilerplate:
                continue
            new_lines.append(line)
        cleaned.append("\n".join(new_lines))

    return cleaned


def _normalise_key(line: str) -> str:
    """Normalise a line for duplicate detection (collapse numbers, lowercase)."""
    return re.sub(r"\d+", "#", line.lower()).strip()


def _final_cleanup(text: str) -> str:
    # Collapse runs of spaces (but not newlines)
    text = re.sub(r" {2,}", " ", text)
    # Ensure consistent line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3+ newlines again after all processing
    text = _MULTIPLE_BLANK_LINES.sub("\n\n", text)
    return text.strip()
