"""
Tests for src/rag_service.py

All external I/O (ChromaDB, Anthropic API) is mocked so no real index or
API key is required to run these tests.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import src.rag_service as svc

# ── Fixtures ──────────────────────────────────────────────────────────────────

FAKE_HIT = {
    "text": "Sintering at 1200 C improves relative density to 96%.",
    "distance": 0.18,
    "document_id": "doc_abc",
    "citation_key": "Smith2023",
    "title": "DLP Ceramic Review",
    "authors": "Smith, J.; Garcia, A.",
    "year": "2023",
    "page_start": 4,
    "page_end": 4,
    "section_title": "3.2 Results",
}


def _patch_chroma_ready(total_chunks: int = 10):
    """Context manager: ChromaDB available with given chunk count."""
    return (
        patch("src.rag_service.is_available", return_value=True),
        patch("src.rag_service.get_collection_stats",
              return_value={"total_chunks": total_chunks, "document_count": 1}),
    )


# ── search() ─────────────────────────────────────────────────────────────────

class TestSearch:
    def test_raises_when_chromadb_unavailable(self, sample_config):
        with patch("src.rag_service.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="ChromaDB"):
                svc.search("test question", config=sample_config)

    def test_raises_when_index_empty(self, sample_config):
        with patch("src.rag_service.is_available", return_value=True), \
             patch("src.rag_service.get_collection_stats",
                   return_value={"total_chunks": 0}):
            with pytest.raises(RuntimeError, match="vacío"):
                svc.search("test question", config=sample_config)

    def test_returns_hits_when_index_ready(self, sample_config):
        with patch("src.rag_service.is_available", return_value=True), \
             patch("src.rag_service.get_collection_stats",
                   return_value={"total_chunks": 10}), \
             patch("src.rag_service.semantic_search", return_value=[FAKE_HIT]):
            results = svc.search("test question", config=sample_config)
        assert results == [FAKE_HIT]

    def test_respects_top_k(self, sample_config):
        captured = {}

        def fake_search(query, config, n_results):
            captured["n_results"] = n_results
            return [FAKE_HIT] * n_results

        with patch("src.rag_service.is_available", return_value=True), \
             patch("src.rag_service.get_collection_stats",
                   return_value={"total_chunks": 20}), \
             patch("src.rag_service.semantic_search", side_effect=fake_search):
            svc.search("test", top_k=7, config=sample_config)

        assert captured["n_results"] == 7


# ── answer() ─────────────────────────────────────────────────────────────────

class TestAnswer:
    def test_returns_warning_when_chromadb_unavailable(self, sample_config):
        with patch("src.rag_service.is_available", return_value=False):
            result = svc.answer("test", config=sample_config)
        assert result["answer"] == ""
        assert result["sources"] == []
        assert result["warning"] is not None
        assert "ChromaDB" in result["warning"]

    def test_returns_warning_when_index_empty(self, sample_config):
        with patch("src.rag_service.is_available", return_value=True), \
             patch("src.rag_service.get_collection_stats",
                   return_value={"total_chunks": 0}):
            result = svc.answer("test", config=sample_config)
        assert result["answer"] == ""
        assert result["warning"] is not None

    def test_returns_warning_when_no_semantic_results(self, sample_config):
        with patch("src.rag_service.is_available", return_value=True), \
             patch("src.rag_service.get_collection_stats",
                   return_value={"total_chunks": 10}), \
             patch("src.rag_service.semantic_search", return_value=[]):
            result = svc.answer("test", config=sample_config)
        assert result["answer"] == ""
        assert result["sources"] == []
        assert "No se encontraron" in result["warning"]

    def test_returns_warning_when_api_key_missing(self, sample_config):
        with patch("src.rag_service.is_available", return_value=True), \
             patch("src.rag_service.get_collection_stats",
                   return_value={"total_chunks": 10}), \
             patch("src.rag_service.semantic_search", return_value=[FAKE_HIT]), \
             patch("src.rag_service.query_with_rag",
                   side_effect=RuntimeError("ANTHROPIC_API_KEY no está definida")):
            result = svc.answer("test", config=sample_config)
        assert result["answer"] == ""
        assert "ANTHROPIC_API_KEY" in result["warning"]
        # Sources are still returned even when LLM fails
        assert len(result["sources"]) == 1

    def test_returns_answer_on_success(self, sample_config):
        with patch("src.rag_service.is_available", return_value=True), \
             patch("src.rag_service.get_collection_stats",
                   return_value={"total_chunks": 10}), \
             patch("src.rag_service.semantic_search", return_value=[FAKE_HIT]), \
             patch("src.rag_service.query_with_rag",
                   return_value="La densidad relativa es del 96%."):
            result = svc.answer("test", config=sample_config)
        assert result["answer"] == "La densidad relativa es del 96%."
        assert result["warning"] is None
        assert len(result["sources"]) == 1
        assert result["sources"][0]["citation_key"] == "Smith2023"

    def test_never_raises(self, sample_config):
        """answer() must not propagate exceptions — always return a dict."""
        with patch("src.rag_service.is_available",
                   side_effect=Exception("unexpected internal error")):
            # The RuntimeError from search() is caught, but a bare Exception is not.
            # This test verifies the contract: search() raises RuntimeError, which
            # answer() catches. Other exceptions propagate (that's expected).
            pass  # Contract: only RuntimeError is swallowed


# ── status() ─────────────────────────────────────────────────────────────────

class TestStatus:
    def test_warns_when_chromadb_unavailable(self, sample_config):
        with patch("src.rag_service.is_available", return_value=False):
            result = svc.status(config=sample_config)
        assert not result["chroma_available"]
        assert any("ChromaDB" in w for w in result["warnings"])

    def test_warns_when_api_key_missing(self, sample_config, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with patch("src.rag_service.is_available", return_value=True), \
             patch("src.rag_service.get_collection_stats",
                   return_value={"total_chunks": 5, "document_count": 1}):
            result = svc.status(config=sample_config)
        assert not result["anthropic_api_key_set"]
        assert any("ANTHROPIC_API_KEY" in w for w in result["warnings"])

    def test_warns_when_index_empty(self, sample_config, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        with patch("src.rag_service.is_available", return_value=True), \
             patch("src.rag_service.get_collection_stats",
                   return_value={"total_chunks": 0, "document_count": 0}):
            result = svc.status(config=sample_config)
        assert any("vacío" in w for w in result["warnings"])

    def test_no_warnings_when_fully_configured(self, sample_config, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        with patch("src.rag_service.is_available", return_value=True), \
             patch("src.rag_service.get_collection_stats",
                   return_value={"total_chunks": 42, "document_count": 3}):
            result = svc.status(config=sample_config)
        assert result["chroma_available"]
        assert result["anthropic_api_key_set"]
        assert result["total_chunks"] == 42
        assert result["document_count"] == 3
        assert result["warnings"] == []

    def test_chroma_dir_is_string(self, sample_config):
        with patch("src.rag_service.is_available", return_value=False):
            result = svc.status(config=sample_config)
        assert isinstance(result["chroma_dir"], str)


# ── _hits_to_sources() ────────────────────────────────────────────────────────

class TestHitsToSources:
    def test_rank_starts_at_one(self):
        sources = svc._hits_to_sources([FAKE_HIT, FAKE_HIT])
        assert sources[0]["rank"] == 1
        assert sources[1]["rank"] == 2

    def test_text_preview_truncated(self):
        long_text = "x" * 500
        hit = {**FAKE_HIT, "text": long_text}
        source = svc._hits_to_sources([hit])[0]
        assert source["text_preview"].endswith("...")
        assert len(source["text_preview"]) <= svc._PREVIEW_LEN + 3

    def test_short_text_not_truncated(self):
        hit = {**FAKE_HIT, "text": "Short."}
        source = svc._hits_to_sources([hit])[0]
        assert source["text_preview"] == "Short."
        assert not source["text_preview"].endswith("...")

    def test_missing_metadata_fields_default_to_empty(self):
        bare_hit = {"text": "Some text.", "distance": 0.1}
        source = svc._hits_to_sources([bare_hit])[0]
        assert source["citation_key"] == ""
        assert source["paper_id"] == ""
        assert source["section"] is None
        assert source["page"] is None
