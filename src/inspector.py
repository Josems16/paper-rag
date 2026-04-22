"""PDF inspection: detect type, structure and potential problems before extraction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import fitz  # PyMuPDF

from .models import InspectionResult, LayoutHint

logger = logging.getLogger(__name__)

# Pages with fewer printable chars than this are considered "effectively empty"
EMPTY_PAGE_THRESHOLD = 10
# Pages with fewer chars than this are treated as scanned (no embedded text)
SCANNED_PAGE_THRESHOLD = 50
# Ratio of scanned pages above which the whole doc is flagged as scanned
SCANNED_DOC_RATIO = 0.6


def inspect_pdf(pdf_path: str | Path) -> InspectionResult:
    """
    Inspect a PDF and return metadata used to choose the extraction strategy.
    Does NOT extract full text — only samples enough to detect problems.
    """
    path = Path(pdf_path)

    result = InspectionResult(
        filename=path.name,
        file_size_bytes=0,
        page_count=0,
        can_open=False,
        has_embedded_text=False,
        appears_scanned=False,
        is_encrypted=False,
        is_corrupted=False,
    )

    if not path.exists():
        result.errors.append(f"File not found: {path}")
        return result

    result.file_size_bytes = path.stat().st_size

    if result.file_size_bytes == 0:
        result.errors.append("File is empty (0 bytes)")
        result.is_corrupted = True
        return result

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        result.is_corrupted = True
        result.errors.append(f"Cannot open PDF: {exc}")
        return result

    result.can_open = True
    result.page_count = len(doc)

    if result.page_count == 0:
        result.warnings.append("PDF has 0 pages")
        doc.close()
        return result

    result.is_encrypted = doc.is_encrypted

    if result.is_encrypted:
        result.warnings.append("PDF is encrypted / password-protected; text extraction may fail")
        doc.close()
        return result

    # --- Per-page analysis ---
    total_chars = 0
    pages_with_good_text = 0
    chars_per_page: List[int] = []

    for page_num in range(result.page_count):
        try:
            page = doc[page_num]
            text = page.get_text("text")
            char_count = len(text.strip())
        except Exception as exc:
            logger.warning("Error reading page %d: %s", page_num + 1, exc)
            result.warnings.append(f"Page {page_num + 1}: read error ({exc})")
            char_count = 0

        chars_per_page.append(char_count)
        total_chars += char_count

        if char_count < EMPTY_PAGE_THRESHOLD:
            result.empty_pages.append(page_num + 1)
        elif char_count >= SCANNED_PAGE_THRESHOLD:
            pages_with_good_text += 1

    result.avg_chars_per_page = total_chars / result.page_count
    result.text_page_ratio = pages_with_good_text / result.page_count
    result.has_embedded_text = result.text_page_ratio > 0.1
    result.appears_scanned = result.text_page_ratio < (1 - SCANNED_DOC_RATIO)

    # --- Warnings ---
    if result.appears_scanned:
        result.warnings.append(
            f"Document appears scanned: only {result.text_page_ratio:.0%} of pages have embedded text"
        )

    empty_ratio = len(result.empty_pages) / result.page_count
    if empty_ratio > 0.15:
        result.warnings.append(
            f"{len(result.empty_pages)} empty pages ({empty_ratio:.0%} of total)"
        )

    if result.avg_chars_per_page < SCANNED_PAGE_THRESHOLD and not result.appears_scanned:
        result.warnings.append(
            f"Low average text density: {result.avg_chars_per_page:.0f} chars/page"
        )

    # --- Layout heuristic ---
    result.layout_hint = _detect_layout(doc)

    doc.close()
    return result


def _detect_layout(doc: fitz.Document) -> LayoutHint:
    """
    Heuristically detect single vs. two-column layout by sampling block x-positions
    on the first few pages.
    """
    sample_count = min(5, len(doc))
    two_col_votes = 0

    for i in range(sample_count):
        try:
            page = doc[i]
            page_width = page.rect.width
            if page_width == 0:
                continue
            blocks = page.get_text("blocks")
            if not blocks:
                continue

            # Count text blocks in left vs right half (with a small dead zone in center)
            left_blocks = sum(1 for b in blocks if b[4].strip() and b[0] < page_width * 0.45)
            right_blocks = sum(1 for b in blocks if b[4].strip() and b[0] > page_width * 0.55)

            if left_blocks >= 2 and right_blocks >= 2:
                two_col_votes += 1
        except Exception:
            continue

    if sample_count == 0:
        return LayoutHint.UNKNOWN
    if two_col_votes >= max(1, sample_count // 2):
        return LayoutHint.TWO_COL
    return LayoutHint.SINGLE_COL
