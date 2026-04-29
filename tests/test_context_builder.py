"""Tests for src/context_builder.py"""

from __future__ import annotations

import pytest
from src.context_builder import build_context


def _hit(**kwargs) -> dict:
    base = {"text": "Sample text.", "distance": 0.12}
    base.update(kwargs)
    return base


class TestBuildContext:
    def test_empty_returns_empty_string(self):
        assert build_context([]) == ""

    def test_text_appears_in_output(self):
        ctx = build_context([_hit(text="Sintering at 1200°C.")])
        assert "Sintering at 1200°C." in ctx

    def test_fragment_index_starts_at_one(self):
        hits = [_hit(text="A"), _hit(text="B"), _hit(text="C")]
        ctx = build_context(hits)
        assert "[Fragmento 1]" in ctx
        assert "[Fragmento 2]" in ctx
        assert "[Fragmento 3]" in ctx

    def test_multiple_fragments_separated_by_divider(self):
        hits = [_hit(text="First."), _hit(text="Second.")]
        ctx = build_context(hits)
        assert "---" in ctx

    def test_citation_key_preferred_over_document_id(self):
        ctx = build_context([_hit(citation_key="Smith2023", document_id="doc_abc")])
        assert "Smith2023" in ctx
        assert "doc_abc" not in ctx

    def test_document_id_fallback_when_no_citation_key(self):
        ctx = build_context([_hit(document_id="doc_xyz")])
        assert "doc_xyz" in ctx

    def test_title_authors_year_included(self):
        ctx = build_context([_hit(
            title="Novel DLP Ceramic Process",
            authors="García, J.; Martínez, A.",
            year="2024",
        )])
        assert "Novel DLP Ceramic Process" in ctx
        assert "García" in ctx
        assert "2024" in ctx

    def test_page_range_when_start_and_end_differ(self):
        ctx = build_context([_hit(page_start=4, page_end=6)])
        assert "4-6" in ctx

    def test_single_page_label_when_start_equals_end(self):
        ctx = build_context([_hit(page_start=5, page_end=5)])
        assert "Página: 5" in ctx
        assert "5-5" not in ctx

    def test_section_title_included(self):
        ctx = build_context([_hit(section_title="3.2 Experimental Setup")])
        assert "3.2 Experimental Setup" in ctx

    def test_missing_metadata_does_not_raise(self):
        hit = {"text": "Bare hit with no metadata.", "distance": 0.5}
        ctx = build_context([hit])
        assert "Bare hit" in ctx

    def test_empty_string_metadata_not_shown(self):
        ctx = build_context([_hit(section_title="", title="", year="")])
        assert "Sección:" not in ctx
        assert "Título:" not in ctx
        assert "Año:" not in ctx
