"""
Unit tests for the v2 structured extraction modules.

Covers:
  - DOI detection (academic_meta)
  - Volume and page-range extraction (academic_meta)
  - Figure caption detection patterns (image_extractor)
  - Table caption detection patterns (table_extractor)
  - Equation label detection (equation_extractor)
  - In-text mention detection (linker)
  - Markdown table conversion (table_extractor)
  - Specialised chunk formatting (chunker)
  - Linking: linked_figures / linked_tables / linked_equations

All tests are pure-Python — no PDF files required.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

# ── Imports under test ─────────────────────────────────────────────────────────
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.academic_meta import (
    _extract_doi,
    _extract_volume,
    _extract_page_range,
    _extract_keywords,
)
from src.image_extractor import _FIG_RE, _caption_slug
from src.table_extractor import (
    _TABLE_CAPTION_RE,
    _TABLE_LABEL_RE,
    _to_markdown,
    _clean_table,
)
from src.equation_extractor import (
    _extract_label,
    _has_math_density,
    _has_thermal_keyword,
    _EQ_NUMBER_END,
    _EQ_EXPLICIT,
)
from src.linker import _norm, link_chunks_to_elements
from src.chunker import (
    _format_table_chunk,
    _format_figure_chunk,
    _format_equation_chunk,
    build_specialized_chunks,
)
from src.models import Chunk


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_chunk(text: str, chunk_id: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id or str(uuid.uuid4()),
        document_id="doc1",
        source_file="paper.pdf",
        text=text,
        page_start=1,
        page_end=1,
        chunk_index=0,
        extraction_method="pymupdf",
        quality_score=0.9,
        processing_status="chunked",
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _make_table(label: str = "Table 1", page: int = 2) -> dict:
    return {
        "table_id": str(uuid.uuid4()),
        "paper_id": "doc1",
        "page": page,
        "label": label,
        "caption": f"{label}: Summary of results",
        "headers": ["A", "B", "C"],
        "rows": [["1", "2", "3"], ["4", "5", "6"]],
        "markdown": "| A | B | C |\n|---|---|---|\n| 1 | 2 | 3 |",
        "bbox": None,
        "surrounding_text": None,
        "mentioned_in_chunks": [],
    }


def _make_figure(label: str = "Fig. 1", page: int = 3) -> dict:
    return {
        "figure_id": str(uuid.uuid4()),
        "image_index": 0,
        "page": page,
        "filename": "p03_img01_Fig1.png",
        "label": label,
        "caption": f"{label}. Schematic of the system.",
        "bbox": None,
        "surrounding_text": "The system is shown in Fig. 1.",
        "mentioned_in_chunks": [],
    }


def _make_equation(label: str = "(1)", page: int = 4) -> dict:
    return {
        "equation_id": str(uuid.uuid4()),
        "paper_id": "doc1",
        "page": page,
        "label": label,
        "text": "COP = Q_c / W_net  (1)",
        "image_path": None,
        "surrounding_text": "The coefficient of performance is defined as",
        "mentioned_in_chunks": [],
    }


# ── A. DOI detection ───────────────────────────────────────────────────────────

class TestDoiExtraction:
    def test_standard_doi(self):
        text = "doi: 10.1016/j.enconman.2023.117500"
        assert _extract_doi(text) == "10.1016/j.enconman.2023.117500"

    def test_doi_in_sentence(self):
        text = "Available at https://doi.org/10.1007/s00231-020-02985-4 (open access)."
        doi = _extract_doi(text)
        assert doi is not None
        assert doi.startswith("10.")

    def test_no_doi(self):
        assert _extract_doi("No DOI in this text") is None

    def test_doi_trailing_period_stripped(self):
        text = "See 10.1016/j.energy.2022.123456."
        doi = _extract_doi(text)
        assert doi is not None
        assert not doi.endswith(".")


# ── A. Volume & pages ─────────────────────────────────────────────────────────

class TestVolumeAndPages:
    def test_volume_found(self):
        assert _extract_volume("Applied Energy, Vol. 12, pp. 123-145") == "12"

    def test_volume_long_form(self):
        assert _extract_volume("Volume 305, 2023") == "305"

    def test_volume_not_found(self):
        assert _extract_volume("No volume info here") is None

    def test_pages_hyphen(self):
        result = _extract_page_range("pp. 123-145, 2022")
        assert result == "123-145"

    def test_pages_en_dash(self):
        result = _extract_page_range("pp. 23–89")
        assert result == "23-89"

    def test_pages_not_found(self):
        assert _extract_page_range("No page numbers") is None


# ── B. Figure caption detection ───────────────────────────────────────────────

class TestFigureCaptionPattern:
    @pytest.mark.parametrize("text", [
        "Fig. 1. System overview.",
        "Fig 1 System overview",
        "Figure 1. Schematic diagram.",
        "FIGURE 1: Results",
        "Figura 1. Esquema del sistema.",
        "figura 2 detalle",
        "Fig. 1.2 Sub-figure",
    ])
    def test_fig_re_matches(self, text):
        assert _FIG_RE.match(text), f"Pattern did not match: {text!r}"

    @pytest.mark.parametrize("text", [
        "This figure shows the result",
        "Configuration of the system",
        "Table 1. Summary",
    ])
    def test_fig_re_no_match(self, text):
        assert not _FIG_RE.match(text), f"Pattern wrongly matched: {text!r}"

    def test_caption_slug_basic(self):
        assert _caption_slug("Fig. 3") == "_Fig3"

    def test_caption_slug_none(self):
        assert _caption_slug(None) == ""

    def test_caption_slug_figura(self):
        assert _caption_slug("Figura 2") == "_Figura2"


# ── C. Table caption detection ────────────────────────────────────────────────

class TestTableCaptionPattern:
    @pytest.mark.parametrize("text", [
        "Table 1. Summary of properties",
        "TABLE 2: Experimental conditions",
        "Tabla 1. Propiedades del material",
        "Tab. 3 Results",
        "Table IV. Comparison",
    ])
    def test_table_caption_matches(self, text):
        m = _TABLE_CAPTION_RE.search(text)
        assert m is not None, f"Caption pattern did not match: {text!r}"

    @pytest.mark.parametrize("text", [
        "Fig. 1. System",
        "In this paper we present",
    ])
    def test_table_caption_no_match(self, text):
        assert not _TABLE_CAPTION_RE.search(text)

    def test_table_label_extraction(self):
        m = _TABLE_LABEL_RE.match("Table 1. Full caption here")
        assert m is not None
        assert "Table 1" in m.group(1)


# ── D. Equation label detection ───────────────────────────────────────────────

class TestEquationDetection:
    @pytest.mark.parametrize("line,expected_label_prefix", [
        ("Q = m * cp * ΔT                (1)", "(1)"),
        ("COP = Q_c / W_net  (2)", "(2)"),
        ("η_sys = Q_u / Q_in           (12)", "(12)"),
    ])
    def test_trailing_number_detected(self, line, expected_label_prefix):
        label = _extract_label(line)
        assert label is not None
        assert expected_label_prefix in label

    @pytest.mark.parametrize("line", [
        "Eq. (1) defines the COP",
        "see Equation (3) for details",
        "Ecuación (2) muestra el balance",
    ])
    def test_explicit_label_detected(self, line):
        label = _extract_label(line)
        assert label is not None

    def test_no_label_in_plain_text(self):
        assert _extract_label("The system operates at steady state") is None

    def test_thermal_keyword_detected(self):
        assert _has_thermal_keyword("The COP of the system was 0.72")
        assert _has_thermal_keyword("Re = 4500 turbulent regime")
        assert _has_thermal_keyword("NTU method for heat exchangers")

    def test_math_density(self):
        # Line with Greek letters: triggers math density
        assert _has_math_density("η = ΔT / T_hot   α·β·γ = 0.95")

    def test_plain_text_no_math(self):
        assert not _has_math_density("The results show good agreement with literature")


# ── E. In-text mention detection (linker) ─────────────────────────────────────

class TestMentionDetection:
    @pytest.mark.parametrize("text,expected_fig", [
        ("As shown in Fig. 1,", "fig. 1"),
        ("See Figure 2 for details", "figure 2"),
        ("Figura 3 ilustra el sistema", "figura 3"),
        ("In Fig 4 we present", "fig 4"),
    ])
    def test_figure_mentions(self, text, expected_fig):
        from src.linker import _FIG_MENTION
        matches = [_norm(m) for m in _FIG_MENTION.findall(text)]
        assert any(expected_fig in m for m in matches), f"Expected '{expected_fig}' in {matches}"

    @pytest.mark.parametrize("text,expected_tab", [
        ("Table 1 shows the results", "table 1"),
        ("See Tabla 2 for details", "tabla 2"),
        ("Tab. 3 presents", "tab. 3"),
    ])
    def test_table_mentions(self, text, expected_tab):
        from src.linker import _TABLE_MENTION
        matches = [_norm(m) for m in _TABLE_MENTION.findall(text)]
        assert any(expected_tab in m for m in matches), f"Expected '{expected_tab}' in {matches}"

    @pytest.mark.parametrize("text,expected_eq", [
        ("Using Eq. (1) we compute", "eq. (1)"),
        ("from Equation (3)", "equation (3)"),
    ])
    def test_equation_mentions(self, text, expected_eq):
        from src.linker import _EQ_MENTION
        matches = [_norm(m) for m in _EQ_MENTION.findall(text)]
        assert any(expected_eq in m for m in matches), f"Expected '{expected_eq}' in {matches}"


# ── F. Markdown table conversion ──────────────────────────────────────────────

class TestMarkdownConversion:
    def test_basic_table(self):
        headers = ["A", "B", "C"]
        rows = [["1", "2", "3"], ["4", "5", "6"]]
        md = _to_markdown(headers, rows)
        assert "| A | B | C |" in md
        assert "| --- |" in md
        assert "| 1 | 2 | 3 |" in md

    def test_empty_cells(self):
        headers = ["X", "Y"]
        rows = [[None, "value"], ["a", None]]
        md = _to_markdown(headers, rows)
        assert "| X | Y |" in md

    def test_pipe_in_cell_escaped(self):
        headers = ["Col|A", "B"]
        rows = [["1|2", "3"]]
        md = _to_markdown(headers, rows)
        assert "\\|" in md

    def test_empty_headers_returns_empty(self):
        assert _to_markdown([], []) == ""

    def test_table_cleaning_strips_whitespace(self):
        raw = [["  hello  ", "  world  "], ["  a  ", None]]
        cleaned = _clean_table(raw)
        assert cleaned[0][0] == "hello"
        assert cleaned[0][1] == "world"

    def test_clean_table_removes_all_empty_rows(self):
        raw = [[None, None], ["a", "b"]]
        cleaned = _clean_table(raw)
        assert len(cleaned) == 1
        assert cleaned[0] == ["a", "b"]


# ── G. Specialised chunks ─────────────────────────────────────────────────────

class TestSpecialisedChunks:
    def test_table_chunk_format(self):
        table = _make_table("Table 1")
        text = _format_table_chunk(table, "paper.pdf")
        assert "[Table 1]" in text
        assert "Paper: paper.pdf" in text
        assert "Page: 2" in text
        assert "Markdown:" in text

    def test_figure_chunk_format(self):
        fig = _make_figure("Fig. 3")
        text = _format_figure_chunk(fig, "paper.pdf")
        assert "[Fig. 3]" in text
        assert "Caption:" in text

    def test_equation_chunk_format(self):
        eq = _make_equation("(1)")
        text = _format_equation_chunk(eq, "paper.pdf")
        assert "[(1)]" in text
        assert "Equation:" in text
        assert "COP" in text

    def test_build_specialized_chunks_returns_chunks(self):
        tables = [_make_table("Table 1")]
        figures = [_make_figure("Fig. 1")]
        equations = [_make_equation("(1)")]
        chunks = build_specialized_chunks(
            tables=tables,
            figures=figures,
            equations=equations,
            document_id="doc1",
            source_file="paper.pdf",
        )
        types = {c.chunk_type for c in chunks}
        assert "table" in types
        assert "figure" in types
        assert "equation" in types

    def test_chunk_source_id_set(self):
        table = _make_table()
        chunks = build_specialized_chunks(
            tables=[table], figures=[], equations=[],
            document_id="doc1", source_file="paper.pdf",
        )
        assert chunks[0].source_id == table["table_id"]


# ── H. Linking ────────────────────────────────────────────────────────────────

class TestLinking:
    def test_linked_figures_populated(self):
        chunk = _make_chunk("As shown in Fig. 1 and Figure 2, the system…")
        fig1 = _make_figure("Fig. 1")
        fig2 = _make_figure("Figure 2")
        link_chunks_to_elements([chunk], [fig1, fig2], [], [])
        assert any("fig" in lf for lf in chunk.linked_figures)

    def test_linked_tables_populated(self):
        chunk = _make_chunk("Table 1 and Tabla 2 show the results")
        t1 = _make_table("Table 1")
        t2 = _make_table("Tabla 2")
        link_chunks_to_elements([chunk], [], [t1, t2], [])
        assert len(chunk.linked_tables) >= 1

    def test_linked_equations_populated(self):
        chunk = _make_chunk("Using Eq. (1) and Equation (2) we derive…")
        e1 = _make_equation("Eq. (1)")
        e2 = _make_equation("Equation (2)")
        link_chunks_to_elements([chunk], [], [], [e1, e2])
        assert len(chunk.linked_equations) >= 1

    def test_no_links_when_no_mentions(self):
        chunk = _make_chunk("The system operates at steady state.")
        link_chunks_to_elements([chunk], [_make_figure()], [_make_table()], [])
        assert chunk.linked_figures == []
        assert chunk.linked_tables == []

    def test_mentioned_in_chunks_back_filled(self):
        chunk = _make_chunk("See Fig. 1 for details.", chunk_id="cid-001")
        fig = _make_figure("Fig. 1")
        link_chunks_to_elements([chunk], [fig], [], [])
        assert "cid-001" in fig["mentioned_in_chunks"]
