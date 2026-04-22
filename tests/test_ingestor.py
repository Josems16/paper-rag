"""Integration tests for the full ingestion pipeline."""

from __future__ import annotations

import pytest


def test_ingest_valid_pdf(sample_pdf_path, sample_config):
    from src.ingestor import ingest_pdf
    from src.models import ProcessingStatus, QualityStatus
    report = ingest_pdf(sample_pdf_path, config=sample_config, verbose=False)

    assert report.document_id
    assert report.source_file == sample_pdf_path.name
    assert report.page_count == 4
    assert report.processing_status != ProcessingStatus.FAILED
    assert report.quality_status != QualityStatus.FAILED
    assert report.chunks_created > 0
    assert report.stored_successfully is True


def test_ingest_produces_artifacts(sample_pdf_path, sample_config):
    from src.ingestor import ingest_pdf
    report = ingest_pdf(sample_pdf_path, config=sample_config, verbose=False)

    doc_id = report.document_id
    raw_dir = sample_config.raw_dir / doc_id
    proc_dir = sample_config.processed_dir / doc_id

    assert (raw_dir / "original.pdf").exists()
    assert (raw_dir / "raw_text.txt").exists()
    assert (proc_dir / "normalized_text.txt").exists()
    assert (proc_dir / "chunks.jsonl").exists()
    assert (proc_dir / "metadata.json").exists()
    assert (sample_config.reports_dir / f"{doc_id}.json").exists()


def test_ingest_nonexistent_file(nonexistent_path, sample_config):
    from src.ingestor import ingest_pdf
    from src.models import ProcessingStatus
    report = ingest_pdf(nonexistent_path, config=sample_config, verbose=False)

    assert report.processing_status == ProcessingStatus.FAILED
    assert len(report.errors) > 0


def test_ingest_report_structure(sample_pdf_path, sample_config):
    from src.ingestor import ingest_pdf
    report = ingest_pdf(sample_pdf_path, config=sample_config, verbose=False)
    d = report.to_dict()

    required_keys = {
        "document_id", "source_file", "processing_status",
        "quality_status", "extraction_method", "page_count",
        "chunks_created", "stored_successfully", "ready_for_rag",
        "warnings", "errors", "processing_time_seconds",
    }
    assert required_keys.issubset(d.keys())


def test_ingest_index_updated(sample_pdf_path, sample_config):
    from src.ingestor import ingest_pdf
    from src.indexer import get_index_stats
    report = ingest_pdf(sample_pdf_path, config=sample_config, verbose=False)

    stats = get_index_stats(sample_config)
    assert stats["total_chunks"] > 0
    assert report.document_id in stats["documents"]


def test_ingest_folder(tmp_path, sample_config):
    from src.ingestor import ingest_folder, ingest_pdf
    from tests.conftest import _make_pdf_bytes

    # Create two PDFs in a temp folder
    folder = tmp_path / "batch_input"
    folder.mkdir()
    for i in range(2):
        pages = [f"Page content {i} — " + "text " * 50] * 2
        (folder / f"doc_{i}.pdf").write_bytes(_make_pdf_bytes(pages))

    reports = ingest_folder(folder, config=sample_config, verbose=False)

    assert len(reports) == 2
    for r in reports:
        assert r.chunks_created > 0
