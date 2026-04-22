"""
Quality validation of extracted text.

Calculates objective metrics and assigns an overall quality status.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import List

from .config import Config
from .models import ExtractionResult, QualityStatus, ValidationResult

# Regex for characters that strongly suggest garbage extraction
_GARBAGE_PATTERN = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f"  # control chars
    r"\ufffd"                                   # replacement char
    r"\ufffe\uffff"                             # non-chars
    r"]"
)

# Lines that are suspiciously short and repetitive (headers/footers)
_MIN_BODY_LINE_LEN = 15


def validate_extraction(
    extraction: ExtractionResult,
    config: Config | None = None,
) -> ValidationResult:
    """Assess quality of an extraction result and return a ValidationResult."""
    cfg = config or Config()

    pages = extraction.pages
    texts_by_page = extraction.raw_text_by_page

    total_pages = max(len(pages), 1)
    pages_with_text = sum(1 for p in pages if p.has_text)
    pages_empty = sum(1 for p in pages if p.is_empty)

    # --- Char counts ---
    all_chars = sum(p.char_count for p in pages)
    avg_chars = all_chars / total_pages

    # --- Garbage char ratio ---
    full_text = "\n".join(texts_by_page)
    garbage_ratio = _garbage_ratio(full_text)

    # --- Duplicate header/footer ratio ---
    dup_ratio = _duplicate_line_ratio(texts_by_page)

    warnings: List[str] = []
    errors: List[str] = []

    # --- Build score ---
    score = 1.0

    if pages_with_text == 0:
        score = 0.0
        errors.append("No pages with usable text were extracted")
    else:
        text_coverage = pages_with_text / total_pages
        if text_coverage < 0.5:
            score -= 0.3
            warnings.append(f"Only {text_coverage:.0%} of pages have extracted text")

        if garbage_ratio > cfg.garbage_char_threshold:
            penalty = min(0.3, garbage_ratio * 3)
            score -= penalty
            warnings.append(
                f"High garbage character ratio: {garbage_ratio:.2%} "
                f"(threshold: {cfg.garbage_char_threshold:.2%})"
            )

        if dup_ratio > 0.15:
            score -= 0.1
            warnings.append(
                f"Suspiciously high repeated-line ratio ({dup_ratio:.0%}); "
                "possible header/footer contamination"
            )

        if avg_chars < 100 and pages_with_text > 0:
            score -= 0.1
            warnings.append(
                f"Low average character density: {avg_chars:.0f} chars/page"
            )

    score = max(0.0, min(1.0, score))

    # --- Assign status ---
    if score == 0.0 or not extraction.success:
        status = QualityStatus.FAILED
    elif score < cfg.critical_quality_score:
        status = QualityStatus.FAILED
        errors.append(
            f"Quality score {score:.2f} is below critical threshold {cfg.critical_quality_score}"
        )
    elif score < cfg.min_quality_score:
        status = QualityStatus.REVIEW_RECOMMENDED
        warnings.append(
            f"Quality score {score:.2f} is below minimum threshold {cfg.min_quality_score}"
        )
    elif warnings:
        status = QualityStatus.OK_WITH_WARNINGS
    else:
        status = QualityStatus.OK

    return ValidationResult(
        quality_status=status,
        quality_score=score,
        pages_with_text=pages_with_text,
        pages_empty=pages_empty,
        avg_chars_per_page=avg_chars,
        garbage_char_ratio=garbage_ratio,
        duplicate_header_footer_ratio=dup_ratio,
        warnings=warnings,
        errors=errors,
    )


def _garbage_ratio(text: str) -> float:
    if not text:
        return 0.0
    garbage_count = len(_GARBAGE_PATTERN.findall(text))
    return garbage_count / max(len(text), 1)


def _duplicate_line_ratio(pages: List[str]) -> float:
    """
    Detect repetitive short lines (headers/footers) that appear on many pages.
    Returns fraction of pages that share at least one suspicious repeated line.
    """
    if len(pages) < 3:
        return 0.0

    line_page_counts: Counter = Counter()
    for page_text in pages:
        seen_on_page = set()
        for line in page_text.splitlines():
            stripped = line.strip()
            if stripped and len(stripped) <= 80:
                # Normalize to catch near-duplicates (e.g. page numbers)
                normalized = re.sub(r"\d+", "#", stripped)
                if normalized not in seen_on_page:
                    line_page_counts[normalized] += 1
                    seen_on_page.add(normalized)

    total_pages = len(pages)
    # A line appearing on >60% of pages is suspicious
    suspicious_lines = {
        line for line, cnt in line_page_counts.items()
        if cnt / total_pages > 0.6
    }

    if not suspicious_lines:
        return 0.0

    # Count how many pages contain ≥1 suspicious line
    affected = sum(
        1 for page_text in pages
        if any(
            re.sub(r"\d+", "#", line.strip()) in suspicious_lines
            for line in page_text.splitlines()
            if line.strip()
        )
    )
    return affected / total_pages
