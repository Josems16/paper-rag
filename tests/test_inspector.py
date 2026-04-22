"""Tests for the inspector module."""

from __future__ import annotations

import pytest


def test_inspect_valid_pdf(sample_pdf_path):
    from src.inspector import inspect_pdf
    result = inspect_pdf(sample_pdf_path)

    assert result.can_open is True
    assert result.is_corrupted is False
    assert result.is_encrypted is False
    assert result.page_count == 4
    assert result.file_size_bytes > 0
    assert not result.errors


def test_inspect_nonexistent_file(nonexistent_path):
    from src.inspector import inspect_pdf
    result = inspect_pdf(nonexistent_path)

    assert result.can_open is False
    assert len(result.errors) > 0


def test_inspect_has_embedded_text(sample_pdf_path):
    from src.inspector import inspect_pdf
    result = inspect_pdf(sample_pdf_path)

    assert result.has_embedded_text is True
    assert result.avg_chars_per_page > 0


def test_inspect_empty_pdf(empty_pdf_path):
    from src.inspector import inspect_pdf
    result = inspect_pdf(empty_pdf_path)

    assert result.can_open is True
    assert result.page_count == 1
    # Empty page should be reported
    assert result.avg_chars_per_page < 50


def test_inspect_layout_hint(sample_pdf_path):
    from src.inspector import inspect_pdf
    from src.models import LayoutHint
    result = inspect_pdf(sample_pdf_path)

    # Should return a valid layout hint
    assert result.layout_hint in list(LayoutHint)


def test_inspect_to_dict(sample_pdf_path):
    from src.inspector import inspect_pdf
    result = inspect_pdf(sample_pdf_path)
    d = result.to_dict()

    required_keys = {
        "filename", "file_size_bytes", "page_count", "can_open",
        "has_embedded_text", "appears_scanned", "is_encrypted",
        "is_corrupted", "layout_hint",
    }
    assert required_keys.issubset(d.keys())
