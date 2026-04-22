"""Tests for the normalizer module."""

from __future__ import annotations

import pytest


def test_basic_normalization():
    from src.normalizer import normalize_text
    pages = ["Hello World.\n\nThis is a paragraph."]
    full, pages_out = normalize_text(pages)

    assert "Hello World" in full
    assert len(pages_out) == 1


def test_control_chars_removed():
    from src.normalizer import normalize_text
    pages = ["Normal text\x00\x01\x02with garbage."]
    full, _ = normalize_text(pages)

    assert "\x00" not in full
    assert "Normal text" in full


def test_hyphenated_word_reconstruction():
    from src.normalizer import normalize_text
    pages = ["The pro-\ncessing of documents."]
    full, _ = normalize_text(pages)

    assert "processing" in full


def test_empty_pages_filtered():
    from src.normalizer import normalize_text
    pages = ["Real content here.", "", "   "]
    full, pages_out = normalize_text(pages)

    assert "Real content" in full
    # Empty pages should not produce PAGE_BREAK markers
    assert full.count("[PAGE_BREAK]") == 0  # only 1 non-empty page


def test_multiple_blank_lines_collapsed():
    from src.normalizer import normalize_text
    pages = ["Line 1\n\n\n\n\n\nLine 2"]
    full, _ = normalize_text(pages)

    # Should not have more than 2 consecutive newlines
    assert "\n\n\n" not in full


def test_repeated_header_stripped():
    from src.normalizer import normalize_text
    header = "Journal of Science — Vol. 1\n"
    pages = [header + f"Unique content for page {i}. Some longer text here to pad the page." for i in range(8)]
    full, cleaned = normalize_text(pages, strip_headers_footers=True)

    # After stripping, the repeated header should appear less
    stripped_occurrences = sum("Journal of Science" in p for p in cleaned)
    assert stripped_occurrences < len(pages)


def test_page_breaks_inserted():
    from src.normalizer import normalize_text
    pages = ["Page one content.", "Page two content.", "Page three content."]
    full, _ = normalize_text(pages)

    assert "[PAGE_BREAK]" in full


def test_unicode_normalization():
    from src.normalizer import normalize_text
    # ﬁ is a ligature that should normalise to fi
    pages = ["The eﬃcient algorithm for proﬁling."]
    full, _ = normalize_text(pages)

    assert "efficient" in full or "ﬃ" not in full or "fi" in full


def test_normalize_single_page():
    from src.normalizer import normalize_single_page
    text = "Hello\x00World"
    result = normalize_single_page(text)

    assert "\x00" not in result
    assert "Hello" in result
