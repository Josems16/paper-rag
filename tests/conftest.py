"""Shared test fixtures and helpers."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Generator

import pytest

# ---------------------------------------------------------------------------
# Minimal synthetic PDF factory (no external test assets required)
# ---------------------------------------------------------------------------

def _make_pdf_bytes(pages: list[str]) -> bytes:
    """
    Build a minimal but valid single-stream PDF with one text page per entry in `pages`.
    Suitable for unit tests without needing real PDF files.
    """
    try:
        import fitz
        doc = fitz.open()
        for page_text in pages:
            page = doc.new_page(width=595, height=842)  # A4
            page.insert_text((50, 100), page_text, fontsize=12)
        return doc.tobytes()
    except Exception:
        # Fallback: return a trivial hand-crafted PDF
        return _minimal_pdf()


def _minimal_pdf() -> bytes:
    """Return a byte string that is a syntactically valid one-page PDF."""
    content = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R "
        b"/MediaBox [0 0 612 792] /Contents 4 0 R /Resources "
        b"<< /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length 44 >>\nstream\n"
        b"BT /F1 12 Tf 100 700 Td (Hello World) Tj ET\n"
        b"endstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000274 00000 n \n"
        b"0000000369 00000 n \n"
        b"trailer\n<< /Size 6 /Root 1 0 R >>\n"
        b"startxref\n441\n%%EOF\n"
    )
    return content


@pytest.fixture()
def sample_pdf_path(tmp_path: Path) -> Path:
    """Write a synthetic multi-page PDF and return its path."""
    pages = [
        "Introduction\n\nThis paper presents a novel approach to machine learning.\n"
        "We demonstrate state-of-the-art results on several benchmarks.\n" * 3,
        "Methods\n\nWe use a transformer architecture with self-attention.\n"
        "The model was trained on a large corpus of scientific text.\n" * 3,
        "Results\n\nTable 1 shows the performance on benchmark datasets.\n"
        "Our method outperforms all baselines by a significant margin.\n" * 3,
        "Conclusion\n\nWe have presented a novel framework for document understanding.\n"
        "Future work will explore multi-modal extensions.\n" * 3,
    ]
    pdf_bytes = _make_pdf_bytes(pages)
    pdf_file = tmp_path / "sample_paper.pdf"
    pdf_file.write_bytes(pdf_bytes)
    return pdf_file


@pytest.fixture()
def empty_pdf_path(tmp_path: Path) -> Path:
    """PDF with no text content."""
    try:
        import fitz
        doc = fitz.open()
        doc.new_page(width=595, height=842)
        pdf_bytes = doc.tobytes()
    except Exception:
        pdf_bytes = _minimal_pdf()
    pdf_file = tmp_path / "empty.pdf"
    pdf_file.write_bytes(pdf_bytes)
    return pdf_file


@pytest.fixture()
def nonexistent_path(tmp_path: Path) -> Path:
    return tmp_path / "does_not_exist.pdf"


@pytest.fixture()
def sample_config(tmp_path: Path):
    from src.config import Config
    cfg = Config(
        input_dir=tmp_path / "input",
        raw_dir=tmp_path / "data" / "raw",
        processed_dir=tmp_path / "data" / "processed",
        reports_dir=tmp_path / "data" / "reports",
        index_dir=tmp_path / "data" / "index",
        ocr_enabled=False,  # no tesseract in test environment
        embeddings_enabled=False,
        min_chunk_size=20,
        chunk_size=200,
        chunk_overlap=30,
    )
    cfg.ensure_dirs()
    return cfg
