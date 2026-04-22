"""
Local file storage for all pipeline artifacts.

Layout under data/:
  raw/<doc_id>/
    original.pdf          ← copy of the input PDF
    raw_text.txt          ← raw extracted text (pages joined with \f)
  processed/<doc_id>/
    normalized_text.txt   ← cleaned/normalized text
    chunks.jsonl          ← one chunk per line (JSON)
    metadata.json         ← document-level metadata
  reports/<doc_id>.json   ← processing report
  index/
    index.jsonl           ← lightweight RAG index (all chunks across docs)
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any

from .config import Config
from .image_extractor import ImageRecord
from .models import Chunk, ExtractionResult, InspectionResult, ProcessingReport, ValidationResult
from typing import Any  # noqa: F811 – re-import for Optional[Any]

logger = logging.getLogger(__name__)


class StorageError(Exception):
    pass


def save_images(
    document_id: str,
    images: List[ImageRecord],
    config: Config,
) -> Path:
    """
    Save extracted images to data/raw/<doc_id>/images/ and write an
    images_index.json with caption metadata. Returns the images directory.
    """
    images_dir = config.raw_dir / document_id / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    index = []
    for rec in images:
        img_path = images_dir / rec.filename
        img_path.write_bytes(rec.data)
        index.append({
            "index": rec.image_index,
            "page": rec.page,
            "filename": rec.filename,
            "width": rec.width,
            "height": rec.height,
            "caption": rec.caption,
        })

    index_path = images_dir / "images_index.json"
    index_path.write_text(
        json.dumps(index, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved %d image(s) to %s", len(images), images_dir)
    return images_dir


def save_artifacts(
    document_id: str,
    source_pdf: Path,
    extraction: ExtractionResult,
    normalized_text: str,
    normalized_pages: List[str],
    chunks: List[Chunk],
    inspection: InspectionResult,
    validation: ValidationResult,
    config: Config,
    academic_meta: Optional[Any] = None,
    images: Optional[List[ImageRecord]] = None,
) -> Dict[str, Path]:
    """
    Persist all pipeline artifacts.  Returns a mapping of artifact name → path.
    Raises StorageError on critical failures.
    """
    paths: Dict[str, Path] = {}

    # Ensure directories exist
    config.ensure_dirs()
    raw_dir = config.raw_dir / document_id
    proc_dir = config.processed_dir / document_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy original PDF
    try:
        dest_pdf = raw_dir / "original.pdf"
        shutil.copy2(str(source_pdf), str(dest_pdf))
        paths["original_pdf"] = dest_pdf
        logger.debug("Saved original PDF → %s", dest_pdf)
    except Exception as exc:
        raise StorageError(f"Failed to copy original PDF: {exc}") from exc

    # 2. Raw extracted text
    try:
        raw_txt = raw_dir / "raw_text.txt"
        raw_txt.write_text(extraction.raw_text, encoding="utf-8")
        paths["raw_text"] = raw_txt
    except Exception as exc:
        raise StorageError(f"Failed to write raw text: {exc}") from exc

    # 3. Normalized text
    try:
        norm_txt = proc_dir / "normalized_text.txt"
        norm_txt.write_text(normalized_text, encoding="utf-8")
        paths["normalized_text"] = norm_txt
    except Exception as exc:
        raise StorageError(f"Failed to write normalized text: {exc}") from exc

    # 4. Chunks as JSONL
    try:
        chunks_path = proc_dir / "chunks.jsonl"
        with open(chunks_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        paths["chunks"] = chunks_path
    except Exception as exc:
        raise StorageError(f"Failed to write chunks: {exc}") from exc

    # 5. Document metadata
    try:
        meta = {
            "document_id": document_id,
            "source_file": source_pdf.name,
            "source_path": str(source_pdf),
            "page_count": inspection.page_count,
            "layout_hint": inspection.layout_hint.value,
            "has_embedded_text": inspection.has_embedded_text,
            "appears_scanned": inspection.appears_scanned,
            "extraction_method": extraction.method.value,
            "ocr_used": extraction.ocr_used,
            "ocr_pages": extraction.ocr_pages,
            "quality_score": round(validation.quality_score, 3),
            "quality_status": validation.quality_status.value,
            "chunk_count": len(chunks),
        }
        if academic_meta:
            meta["academic"] = academic_meta.to_dict()
        meta_path = proc_dir / "metadata.json"
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        paths["metadata"] = meta_path
    except Exception as exc:
        raise StorageError(f"Failed to write metadata: {exc}") from exc

    # 6. Images (optional)
    if images:
        try:
            paths["images_dir"] = save_images(document_id, images, config)
        except Exception as exc:
            logger.warning("Image storage failed (non-critical): %s", exc)

    return paths


def save_report(report: ProcessingReport, config: Config) -> Path:
    """Save the processing report JSON to data/reports/."""
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = config.reports_dir / f"{report.document_id}.json"
    try:
        report_path.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:
        raise StorageError(f"Failed to write report: {exc}") from exc
    return report_path


def load_chunks(document_id: str, config: Config) -> List[Dict[str, Any]]:
    """Load chunks for a given document from JSONL."""
    chunks_path = config.processed_dir / document_id / "chunks.jsonl"
    if not chunks_path.exists():
        return []
    with open(chunks_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_report(document_id: str, config: Config) -> Optional[Dict[str, Any]]:
    """Load processing report for a given document."""
    report_path = config.reports_dir / f"{document_id}.json"
    if not report_path.exists():
        return None
    with open(report_path, encoding="utf-8") as f:
        return json.load(f)


def list_processed_documents(config: Config) -> List[str]:
    """Return document IDs of all processed documents."""
    if not config.processed_dir.exists():
        return []
    return [p.name for p in config.processed_dir.iterdir() if p.is_dir()]
