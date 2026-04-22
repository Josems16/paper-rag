"""Tests for the validator module."""

from __future__ import annotations

import pytest

from src.models import ExtractionMethod, ExtractionResult, PageInfo, QualityStatus


def _make_extraction(page_texts: list[str], method=ExtractionMethod.PYMUPDF) -> ExtractionResult:
    pages = [
        PageInfo(
            page_num=i + 1,
            has_text=len(t.strip()) >= 50,
            char_count=len(t.strip()),
            is_scanned=False,
            is_empty=len(t.strip()) < 10,
            quality_score=min(1.0, len(t.strip()) / 500),
        )
        for i, t in enumerate(page_texts)
    ]
    return ExtractionResult(
        success=True,
        method=method,
        pages=pages,
        raw_text_by_page=page_texts,
        raw_text="\n".join(page_texts),
    )


def test_validate_good_text():
    from src.validator import validate_extraction
    good_text = "This is a well-extracted paragraph with many words. " * 20
    extraction = _make_extraction([good_text] * 5)
    result = validate_extraction(extraction)

    assert result.quality_status in (QualityStatus.OK, QualityStatus.OK_WITH_WARNINGS)
    assert result.quality_score > 0.7
    assert result.pages_with_text == 5


def test_validate_empty_extraction():
    from src.validator import validate_extraction
    extraction = _make_extraction([""] * 4)
    result = validate_extraction(extraction)

    assert result.quality_status == QualityStatus.FAILED
    assert result.quality_score == 0.0


def test_validate_garbage_text():
    from src.validator import validate_extraction
    garbage = "\x00\x01\x02\x03\x04\x05" * 100
    extraction = _make_extraction([garbage] * 3)
    result = validate_extraction(extraction)

    assert result.garbage_char_ratio > 0
    assert len(result.warnings) > 0


def test_validate_mixed_quality():
    from src.validator import validate_extraction
    good = "Normal readable text paragraph. " * 30
    bad = ""
    extraction = _make_extraction([good, bad, good, bad])
    result = validate_extraction(extraction)

    assert result.pages_with_text == 2
    assert result.pages_empty == 2


def test_validate_duplicate_headers():
    from src.validator import validate_extraction
    header = "My Document Title — Page #\n"
    body = "Some content here that is different for each page. " * 10
    pages = [header + body + f"\nUnique content {i}" for i in range(10)]
    extraction = _make_extraction(pages)
    result = validate_extraction(extraction)

    # Duplicate header ratio should be detected
    assert result.duplicate_header_footer_ratio > 0


def test_validate_returns_dict():
    from src.validator import validate_extraction
    good = "Normal text. " * 50
    extraction = _make_extraction([good] * 3)
    result = validate_extraction(extraction)
    d = result.to_dict()

    assert "quality_status" in d
    assert "quality_score" in d
    assert "garbage_char_ratio" in d
