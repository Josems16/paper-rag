"""
Main orchestration pipeline.

Drives the full ingestion flow for a single PDF:
  received → inspecting → extracting → validating → normalizing
  → chunking → storing → indexing → completed (or failed)

All errors are caught at each stage; the pipeline never raises — it
always returns a ProcessingReport describing what happened.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Optional

from .academic_meta import extract_academic_metadata, AcademicMetadata
from .chromadb_index import add_to_chroma, document_exists
from .config import Config, load_config
from .extractor import extract_text
from .indexer import index_chunks
from .inspector import inspect_pdf
from .models import (
    ErrorType,
    ExtractionMethod,
    ExtractionResult,
    InspectionResult,
    ProcessingReport,
    ProcessingStatus,
    QualityStatus,
    ValidationResult,
    now_iso,
)
from .normalizer import normalize_text
from .chunker import chunk_text
from .image_extractor import extract_images
from .reporter import build_report, print_report
from .storage import StorageError, save_artifacts, save_report
from .validator import validate_extraction

logger = logging.getLogger(__name__)


def ingest_pdf(
    pdf_path: str | Path,
    config: Optional[Config] = None,
    verbose: bool = True,
) -> ProcessingReport:
    """
    Full ingestion pipeline for a single PDF.

    Args:
        pdf_path: Path to the PDF file.
        config:   Pipeline configuration (uses defaults if None).
        verbose:  Whether to print the report to console.

    Returns:
        ProcessingReport with full details of the ingestion.
    """
    cfg = config or load_config()
    cfg.ensure_dirs()

    path = Path(pdf_path)
    # Deterministic ID: same filename always gets the same ID so ChromaDB upsert
    # de-duplicates correctly when the watcher re-runs on an already-indexed PDF.
    document_id = hashlib.sha256(path.name.encode()).hexdigest()[:32]
    start_time = time.monotonic()

    logger.info("=== Ingesting %s (id=%s) ===", path.name, document_id)

    # Skip if this document is already in the semantic index
    if document_exists(document_id, cfg):
        logger.info("Skipping %s — already indexed", path.name)
        return _finalize(
            document_id, path.name, ProcessingStatus.COMPLETED,
            None, None, None, 0, True, True,
            ["Already indexed — skipped"], [], start_time, cfg, verbose,
        )

    # Accumulate state across stages
    inspection: Optional[InspectionResult] = None
    extraction: Optional[ExtractionResult] = None
    validation: Optional[ValidationResult] = None
    academic_meta: Optional[AcademicMetadata] = None
    normalized_text: str = ""
    normalized_pages: list = []
    chunks: list = []
    stored_ok = False
    indexed_ok = False
    extra_warnings: list = []
    extra_errors: list = []
    final_status = ProcessingStatus.RECEIVED

    # -----------------------------------------------------------------------
    # Stage 1: INSPECTING
    # -----------------------------------------------------------------------
    final_status = ProcessingStatus.INSPECTING
    logger.info("[%s] Inspecting…", path.name)
    try:
        inspection = inspect_pdf(path)
        if inspection.is_corrupted:
            extra_errors.append(f"[{ErrorType.ACCESS_ERROR.value}] File is corrupted or unreadable")
            final_status = ProcessingStatus.FAILED
            return _finalize(
                document_id, path.name, final_status,
                inspection, None, None, 0, False, False,
                extra_warnings, extra_errors, start_time, cfg, verbose,
            )
        if inspection.is_encrypted:
            extra_errors.append(
                f"[{ErrorType.ACCESS_ERROR.value}] PDF is encrypted; "
                "provide the password or decrypt the file first"
            )
            final_status = ProcessingStatus.FAILED
            return _finalize(
                document_id, path.name, final_status,
                inspection, None, None, 0, False, False,
                extra_warnings, extra_errors, start_time, cfg, verbose,
            )
    except Exception as exc:
        extra_errors.append(f"[{ErrorType.ACCESS_ERROR.value}] Inspection failed: {exc}")
        logger.exception("Inspection error")
        final_status = ProcessingStatus.FAILED
        return _finalize(
            document_id, path.name, final_status,
            None, None, None, 0, False, False,
            extra_warnings, extra_errors, start_time, cfg, verbose,
        )

    # -----------------------------------------------------------------------
    # Stage 2: EXTRACTING
    # -----------------------------------------------------------------------
    final_status = ProcessingStatus.EXTRACTING
    logger.info("[%s] Extracting text…", path.name)
    try:
        extraction = extract_text(path, inspection=inspection, config=cfg)
        if not extraction.success and not extraction.raw_text_by_page:
            extra_errors.append(
                f"[{ErrorType.EXTRACTION_ERROR.value}] All extraction attempts failed"
            )
            final_status = ProcessingStatus.FAILED
            return _finalize(
                document_id, path.name, final_status,
                inspection, extraction, None, 0, False, False,
                extra_warnings, extra_errors, start_time, cfg, verbose,
            )
        if extraction.fallback_used:
            extra_warnings.append("Primary extractor had low quality; pdfplumber fallback was used")
        if extraction.ocr_used:
            extra_warnings.append(
                f"OCR was used on {len(extraction.ocr_pages)} page(s): {extraction.ocr_pages}"
            )
    except Exception as exc:
        extra_errors.append(f"[{ErrorType.EXTRACTION_ERROR.value}] Extraction error: {exc}")
        logger.exception("Extraction error")
        final_status = ProcessingStatus.FAILED
        return _finalize(
            document_id, path.name, final_status,
            inspection, None, None, 0, False, False,
            extra_warnings, extra_errors, start_time, cfg, verbose,
        )

    # -----------------------------------------------------------------------
    # Stage 3: VALIDATING
    # -----------------------------------------------------------------------
    final_status = ProcessingStatus.VALIDATING
    logger.info("[%s] Validating quality…", path.name)
    try:
        validation = validate_extraction(extraction, cfg)
        if validation.quality_status == QualityStatus.FAILED:
            extra_errors.append(
                f"[{ErrorType.LOW_QUALITY.value}] Extraction quality too low to proceed "
                f"(score={validation.quality_score:.2f})"
            )
            final_status = ProcessingStatus.FAILED
            return _finalize(
                document_id, path.name, final_status,
                inspection, extraction, validation, 0, False, False,
                extra_warnings, extra_errors, start_time, cfg, verbose,
            )
        if validation.quality_status == QualityStatus.REVIEW_RECOMMENDED:
            extra_warnings.append(
                f"[{ErrorType.LOW_QUALITY.value}] Quality score {validation.quality_score:.2f} "
                "is below recommended threshold — manual review advised"
            )
    except Exception as exc:
        extra_errors.append(f"Validation error: {exc}")
        logger.exception("Validation error")
        # Non-fatal: continue with defaults
        validation = ValidationResult(
            quality_status=QualityStatus.OK_WITH_WARNINGS,
            quality_score=0.5,
            pages_with_text=len(extraction.pages),
            pages_empty=0,
            avg_chars_per_page=0,
            garbage_char_ratio=0,
            duplicate_header_footer_ratio=0,
        )

    # -----------------------------------------------------------------------
    # Stage 4: NORMALIZING
    # -----------------------------------------------------------------------
    final_status = ProcessingStatus.NORMALIZING
    logger.info("[%s] Normalizing…", path.name)
    try:
        normalized_text, normalized_pages = normalize_text(
            extraction.raw_text_by_page,
            strip_headers_footers=True,
        )
        if not normalized_text.strip():
            extra_warnings.append(
                f"[{ErrorType.NORMALIZATION_ERROR.value}] Normalization produced empty text"
            )
    except Exception as exc:
        extra_errors.append(f"[{ErrorType.NORMALIZATION_ERROR.value}] Normalization error: {exc}")
        logger.exception("Normalization error")
        normalized_text = extraction.raw_text
        normalized_pages = extraction.raw_text_by_page

    # -----------------------------------------------------------------------
    # Stage 5: CHUNKING
    # -----------------------------------------------------------------------
    final_status = ProcessingStatus.CHUNKING
    logger.info("[%s] Chunking…", path.name)
    try:
        chunks = chunk_text(
            normalized_text=normalized_text,
            normalized_pages=normalized_pages,
            document_id=document_id,
            source_file=path.name,
            extraction_method=extraction.method.value,
            quality_score=validation.quality_score,
            config=cfg,
        )
        if not chunks:
            extra_warnings.append(
                f"[{ErrorType.CHUNKING_ERROR.value}] No chunks were produced"
            )
        else:
            empty_chunks = sum(1 for c in chunks if not c.text.strip())
            if empty_chunks:
                extra_warnings.append(f"{empty_chunks} empty chunks were discarded")
            validation.empty_chunks_ratio = empty_chunks / max(len(chunks), 1)
    except Exception as exc:
        extra_errors.append(f"[{ErrorType.CHUNKING_ERROR.value}] Chunking error: {exc}")
        logger.exception("Chunking error")
        chunks = []

    # -----------------------------------------------------------------------
    # Stage 5b: ACADEMIC METADATA
    # -----------------------------------------------------------------------
    if cfg.extract_academic_metadata:
        logger.info("[%s] Extracting academic metadata…", path.name)
        try:
            academic_meta = extract_academic_metadata(path)
            if academic_meta.title:
                logger.info(
                    "[%s] Title detected: %s", path.name, academic_meta.title[:80]
                )
            if academic_meta.doi:
                logger.info("[%s] DOI: %s", path.name, academic_meta.doi)
        except Exception as exc:
            logger.warning("Academic metadata extraction failed: %s", exc)
            academic_meta = None

    # -----------------------------------------------------------------------
    # Stage 5c: IMAGE EXTRACTION
    # -----------------------------------------------------------------------
    images = []
    logger.info("[%s] Extracting images…", path.name)
    try:
        images = extract_images(path)
        if images:
            captioned = sum(1 for img in images if img.caption)
            logger.info(
                "[%s] %d image(s) found (%d with caption)",
                path.name, len(images), captioned,
            )
    except Exception as exc:
        logger.warning("Image extraction failed (non-critical): %s", exc)

    # -----------------------------------------------------------------------
    # Stage 6: STORING
    # -----------------------------------------------------------------------
    final_status = ProcessingStatus.STORING
    logger.info("[%s] Storing artifacts…", path.name)
    try:
        save_artifacts(
            document_id=document_id,
            source_pdf=path,
            extraction=extraction,
            normalized_text=normalized_text,
            normalized_pages=normalized_pages,
            chunks=chunks,
            inspection=inspection,
            validation=validation,
            academic_meta=academic_meta,
            images=images or None,
            config=cfg,
        )
        stored_ok = True
    except StorageError as exc:
        extra_errors.append(f"[{ErrorType.STORAGE_ERROR.value}] {exc}")
        logger.error("Storage error: %s", exc)
    except Exception as exc:
        extra_errors.append(f"[{ErrorType.STORAGE_ERROR.value}] Unexpected storage error: {exc}")
        logger.exception("Unexpected storage error")

    # -----------------------------------------------------------------------
    # Stage 7: INDEXING (JSONL legacy + ChromaDB semantic)
    # -----------------------------------------------------------------------
    final_status = ProcessingStatus.INDEXING
    logger.info("[%s] Indexing…", path.name)
    if chunks:
        try:
            # Legacy JSONL index (keyword search, always runs)
            indexed_ok = index_chunks(chunks, document_id, cfg)
            if not indexed_ok:
                extra_warnings.append(
                    f"[{ErrorType.INDEXING_ERROR.value}] JSONL indexing failed"
                )
        except Exception as exc:
            extra_errors.append(f"[{ErrorType.INDEXING_ERROR.value}] Indexing error: {exc}")
            logger.exception("Indexing error")

        try:
            # Semantic ChromaDB index
            chroma_ok = add_to_chroma(chunks, academic_meta, cfg)
            if chroma_ok:
                logger.info("[%s] Added to ChromaDB semantic index", path.name)
            else:
                extra_warnings.append(
                    "ChromaDB semantic indexing failed or unavailable. "
                    "Install: pip install chromadb sentence-transformers"
                )
        except Exception as exc:
            extra_warnings.append(f"ChromaDB error (non-fatal): {exc}")
            logger.warning("ChromaDB indexing error: %s", exc)

    # -----------------------------------------------------------------------
    # Final status
    # -----------------------------------------------------------------------
    if extra_errors:
        final_status = ProcessingStatus.FAILED
    elif extra_warnings or (validation and validation.quality_status != QualityStatus.OK):
        final_status = ProcessingStatus.COMPLETED_WITH_WARNINGS
    else:
        final_status = ProcessingStatus.COMPLETED

    return _finalize(
        document_id, path.name, final_status,
        inspection, extraction, validation,
        len(chunks), stored_ok, indexed_ok,
        extra_warnings, extra_errors, start_time, cfg, verbose,
    )


def ingest_folder(
    folder_path: str | Path,
    config: Optional[Config] = None,
    verbose: bool = True,
    recursive: bool = False,
) -> list:
    """
    Process all PDFs in a folder.

    Args:
        folder_path: Directory containing PDF files.
        config:      Pipeline configuration.
        verbose:     Print per-file reports.
        recursive:   Whether to search subdirectories.

    Returns:
        List of ProcessingReport objects.
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_files = sorted(folder.glob(pattern))

    if not pdf_files:
        logger.warning("No PDF files found in %s", folder)
        return []

    logger.info("Found %d PDF file(s) in %s", len(pdf_files), folder)
    reports = []
    for i, pdf in enumerate(pdf_files, 1):
        logger.info("--- [%d/%d] %s ---", i, len(pdf_files), pdf.name)
        report = ingest_pdf(pdf, config=config, verbose=verbose)
        reports.append(report)

    return reports


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _finalize(
    document_id: str,
    source_file: str,
    status: ProcessingStatus,
    inspection: Optional[InspectionResult],
    extraction: Optional[ExtractionResult],
    validation: Optional[ValidationResult],
    chunks_count: int,
    stored_ok: bool,
    indexed_ok: bool,
    warnings: list,
    errors: list,
    start_time: float,
    cfg: Config,
    verbose: bool,
) -> ProcessingReport:
    elapsed = time.monotonic() - start_time

    report = build_report(
        document_id=document_id,
        source_file=source_file,
        inspection=inspection,
        extraction=extraction,
        validation=validation,
        chunks_created=chunks_count,
        stored_successfully=stored_ok,
        indexed_successfully=indexed_ok,
        processing_status=status,
        processing_time=elapsed,
        extra_warnings=warnings,
        extra_errors=errors,
    )

    try:
        save_report(report, cfg)
    except Exception as exc:
        logger.error("Could not save report: %s", exc)

    if verbose:
        print_report(report)

    return report
