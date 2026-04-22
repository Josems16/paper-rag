"""Tests for the extractor module."""

from __future__ import annotations

import pytest


def test_extract_valid_pdf(sample_pdf_path, sample_config):
    from src.extractor import extract_text
    result = extract_text(sample_pdf_path, config=sample_config)

    assert result.success is True
    assert len(result.raw_text_by_page) == 4
    assert len(result.raw_text) > 0
    assert any(p.char_count > 0 for p in result.pages)


def test_extract_nonexistent(nonexistent_path, sample_config):
    from src.extractor import extract_text
    result = extract_text(nonexistent_path, config=sample_config)

    assert result.success is False
    assert len(result.errors) > 0


def test_extract_pages_match(sample_pdf_path, sample_config):
    from src.extractor import extract_text
    result = extract_text(sample_pdf_path, config=sample_config)

    assert len(result.pages) == len(result.raw_text_by_page)


def test_extract_page_metadata(sample_pdf_path, sample_config):
    from src.extractor import extract_text
    result = extract_text(sample_pdf_path, config=sample_config)

    for page in result.pages:
        assert page.page_num >= 1
        assert page.char_count >= 0
        assert 0.0 <= page.quality_score <= 1.0


def test_extract_raw_text_content(sample_pdf_path, sample_config):
    """Extracted text should contain expected content."""
    from src.extractor import extract_text
    result = extract_text(sample_pdf_path, config=sample_config)

    combined = result.raw_text.lower()
    # Our sample PDF contains these words
    assert "introduction" in combined or "methods" in combined or "results" in combined


def test_extraction_method_set(sample_pdf_path, sample_config):
    from src.extractor import extract_text
    from src.models import ExtractionMethod
    result = extract_text(sample_pdf_path, config=sample_config)

    assert result.method in list(ExtractionMethod)
    assert result.method != ExtractionMethod.FAILED
