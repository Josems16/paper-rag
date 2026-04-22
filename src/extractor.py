"""
Multi-strategy PDF text extraction.

Cascade:
  1. PyMuPDF (fast, handles most digital PDFs)
  2. pdfplumber (fallback; better for tables and multi-column layouts)
  3. OCR via pytesseract (last resort for scanned pages)

OCR and pdfplumber are soft dependencies: missing them degrades gracefully.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import fitz  # PyMuPDF

from .config import Config
from .models import ExtractionMethod, ExtractionResult, InspectionResult, PageInfo

logger = logging.getLogger(__name__)

# Lazy-import optional deps
try:
    import pdfplumber as _pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.debug("pdfplumber not available; fallback extraction disabled")

try:
    import pytesseract
    from PIL import Image
    import io
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logger.debug("pytesseract/Pillow not available; OCR disabled")

# Characters per page below which we consider a page "low quality"
LOW_TEXT_THRESHOLD = 50
# Fraction of total text from best candidate to accept as "good enough"
QUALITY_RATIO_THRESHOLD = 1.3  # pdfplumber must be ≥30% better to switch


def extract_text(
    pdf_path: str | Path,
    inspection: Optional[InspectionResult] = None,
    config: Optional[Config] = None,
) -> ExtractionResult:
    """
    Orchestrate extraction with the cascade strategy:
    PyMuPDF → pdfplumber → OCR.
    """
    cfg = config or Config()
    path = Path(pdf_path)

    if not path.exists():
        return ExtractionResult(
            success=False,
            method=ExtractionMethod.FAILED,
            errors=[f"File not found: {path}"],
        )

    # --- Attempt 1: PyMuPDF ---
    result = _extract_pymupdf(path)

    # --- Attempt 2: pdfplumber fallback ---
    if PDFPLUMBER_AVAILABLE and _quality_score(result) < 0.5:
        logger.info("%s: PyMuPDF quality low, trying pdfplumber", path.name)
        fallback = _extract_pdfplumber(path)
        if _quality_score(fallback) > _quality_score(result) * QUALITY_RATIO_THRESHOLD:
            fallback.fallback_used = True
            result = fallback
        else:
            logger.debug("%s: pdfplumber not better; keeping PyMuPDF result", path.name)

    # --- Attempt 3: OCR for weak pages ---
    if cfg.ocr_enabled and OCR_AVAILABLE:
        result = _apply_ocr_where_needed(path, result, cfg, inspection)
    elif cfg.ocr_enabled and not OCR_AVAILABLE:
        if inspection and inspection.appears_scanned:
            result.warnings.append(
                "Document appears scanned but OCR is unavailable (install pytesseract + Pillow)"
            )

    result.raw_text = _join_pages(result.raw_text_by_page)
    return result


# ---------------------------------------------------------------------------
# PyMuPDF extraction
# ---------------------------------------------------------------------------

def _extract_pymupdf(path: Path) -> ExtractionResult:
    result = ExtractionResult(success=False, method=ExtractionMethod.PYMUPDF)
    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        result.errors.append(f"PyMuPDF open error: {exc}")
        return result

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            try:
                text = page.get_text("text")
            except Exception as exc:
                logger.warning("PyMuPDF page %d error: %s", page_num + 1, exc)
                text = ""

            char_count = len(text.strip())
            page_info = PageInfo(
                page_num=page_num + 1,
                has_text=char_count >= LOW_TEXT_THRESHOLD,
                char_count=char_count,
                is_scanned=char_count < LOW_TEXT_THRESHOLD,
                is_empty=char_count < 10,
                extraction_method="pymupdf",
                quality_score=min(1.0, char_count / 500),
            )
            result.pages.append(page_info)
            result.raw_text_by_page.append(text)

        result.success = True
    except Exception as exc:
        result.errors.append(f"PyMuPDF extraction error: {exc}")
    finally:
        doc.close()

    return result


# ---------------------------------------------------------------------------
# pdfplumber extraction
# ---------------------------------------------------------------------------

def _extract_pdfplumber(path: Path) -> ExtractionResult:
    result = ExtractionResult(success=False, method=ExtractionMethod.PDFPLUMBER)
    try:
        with _pdfplumber.open(str(path)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text() or ""
                except Exception as exc:
                    logger.warning("pdfplumber page %d error: %s", page_num + 1, exc)
                    text = ""

                char_count = len(text.strip())
                page_info = PageInfo(
                    page_num=page_num + 1,
                    has_text=char_count >= LOW_TEXT_THRESHOLD,
                    char_count=char_count,
                    is_scanned=char_count < LOW_TEXT_THRESHOLD,
                    is_empty=char_count < 10,
                    extraction_method="pdfplumber",
                    quality_score=min(1.0, char_count / 500),
                )
                result.pages.append(page_info)
                result.raw_text_by_page.append(text)

        result.success = True
    except Exception as exc:
        result.errors.append(f"pdfplumber error: {exc}")

    return result


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def _apply_ocr_where_needed(
    path: Path,
    result: ExtractionResult,
    cfg: Config,
    inspection: Optional[InspectionResult],
) -> ExtractionResult:
    """
    Run OCR on pages where text extraction yielded insufficient results.
    Mutates result in-place (upgrades weak pages) and returns it.
    """
    # Identify which pages need OCR
    ocr_page_indices = [
        i for i, p in enumerate(result.pages)
        if p.char_count < LOW_TEXT_THRESHOLD
    ]

    # If >cfg.ocr_trigger_ratio pages need OCR, flag the whole doc
    total = len(result.pages) if result.pages else 1
    ocr_ratio = len(ocr_page_indices) / total

    if not ocr_page_indices:
        return result

    logger.info(
        "%s: %d/%d pages need OCR (%.0f%%)",
        path.name, len(ocr_page_indices), total, ocr_ratio * 100,
    )

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        result.warnings.append(f"Cannot open PDF for OCR: {exc}")
        return result

    ocr_successes = 0
    for idx in ocr_page_indices:
        page_num_1indexed = idx + 1
        try:
            page = doc[idx]
            mat = fitz.Matrix(cfg.ocr_dpi / 72, cfg.ocr_dpi / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_bytes = pix.tobytes("png")

            pil_img = Image.open(io.BytesIO(img_bytes))
            ocr_text = pytesseract.image_to_string(
                pil_img,
                lang=cfg.ocr_lang,
                config="--psm 3",
            )
            char_count = len(ocr_text.strip())

            # Only adopt OCR result if it's actually better
            if char_count > result.raw_text_by_page[idx].__len__():
                result.raw_text_by_page[idx] = ocr_text
                result.pages[idx].char_count = char_count
                result.pages[idx].has_text = char_count >= LOW_TEXT_THRESHOLD
                result.pages[idx].is_scanned = True
                result.pages[idx].extraction_method = "ocr"
                result.pages[idx].quality_score = min(1.0, char_count / 500)
                result.ocr_pages.append(page_num_1indexed)
                ocr_successes += 1

        except pytesseract.TesseractNotFoundError:
            result.warnings.append(
                "Tesseract binary not found. Install Tesseract OCR and ensure it is on PATH."
            )
            logger.warning("Tesseract not found; OCR aborted")
            break
        except Exception as exc:
            logger.warning("OCR error on page %d: %s", page_num_1indexed, exc)
            result.pages[idx].warnings.append(f"OCR failed: {exc}")

    doc.close()

    if ocr_successes > 0:
        result.ocr_used = True
        result.method = ExtractionMethod.HYBRID if not result.fallback_used else ExtractionMethod.HYBRID
        if result.method == ExtractionMethod.PYMUPDF or result.method == ExtractionMethod.PDFPLUMBER:
            result.method = ExtractionMethod.HYBRID

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quality_score(result: ExtractionResult) -> float:
    """Simple quality proxy: fraction of pages with ≥ LOW_TEXT_THRESHOLD chars."""
    if not result.pages:
        return 0.0
    good = sum(1 for p in result.pages if p.char_count >= LOW_TEXT_THRESHOLD)
    return good / len(result.pages)


def _join_pages(pages: List[str]) -> str:
    return "\n\f\n".join(pages)
