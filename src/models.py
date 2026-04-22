"""Data models for the PDF ingestion pipeline."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class ProcessingStatus(str, Enum):
    RECEIVED = "received"
    INSPECTING = "inspecting"
    EXTRACTING = "extracting"
    VALIDATING = "validating"
    NORMALIZING = "normalizing"
    CHUNKING = "chunking"
    STORING = "storing"
    INDEXING = "indexing"
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    FAILED = "failed"


class QualityStatus(str, Enum):
    OK = "OK"
    OK_WITH_WARNINGS = "OK_WITH_WARNINGS"
    REVIEW_RECOMMENDED = "REVIEW_RECOMMENDED"
    FAILED = "FAILED"


class ExtractionMethod(str, Enum):
    PYMUPDF = "pymupdf"
    PDFPLUMBER = "pdfplumber"
    OCR = "ocr"
    HYBRID = "hybrid"
    FAILED = "failed"


class ErrorType(str, Enum):
    ACCESS_ERROR = "access_error"
    EXTRACTION_ERROR = "extraction_error"
    LOW_QUALITY = "low_quality"
    OCR_ERROR = "ocr_error"
    NORMALIZATION_ERROR = "normalization_error"
    CHUNKING_ERROR = "chunking_error"
    STORAGE_ERROR = "storage_error"
    INDEXING_ERROR = "indexing_error"


class LayoutHint(str, Enum):
    SINGLE_COL = "single_col"
    TWO_COL = "two_col"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class PageInfo:
    page_num: int
    has_text: bool
    char_count: int
    is_scanned: bool
    is_empty: bool
    extraction_method: Optional[str] = None
    quality_score: float = 0.0
    warnings: List[str] = field(default_factory=list)


@dataclass
class InspectionResult:
    filename: str
    file_size_bytes: int
    page_count: int
    can_open: bool
    has_embedded_text: bool
    appears_scanned: bool
    is_encrypted: bool
    is_corrupted: bool
    empty_pages: List[int] = field(default_factory=list)
    text_page_ratio: float = 0.0
    avg_chars_per_page: float = 0.0
    layout_hint: LayoutHint = LayoutHint.UNKNOWN
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "file_size_bytes": self.file_size_bytes,
            "page_count": self.page_count,
            "can_open": self.can_open,
            "has_embedded_text": self.has_embedded_text,
            "appears_scanned": self.appears_scanned,
            "is_encrypted": self.is_encrypted,
            "is_corrupted": self.is_corrupted,
            "empty_pages": self.empty_pages,
            "text_page_ratio": round(self.text_page_ratio, 3),
            "avg_chars_per_page": round(self.avg_chars_per_page, 1),
            "layout_hint": self.layout_hint.value,
            "warnings": self.warnings,
            "errors": self.errors,
        }


@dataclass
class ExtractionResult:
    success: bool
    method: ExtractionMethod
    pages: List[PageInfo] = field(default_factory=list)
    raw_text: str = ""
    raw_text_by_page: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    fallback_used: bool = False
    ocr_used: bool = False
    ocr_pages: List[int] = field(default_factory=list)


@dataclass
class ValidationResult:
    quality_status: QualityStatus
    quality_score: float
    pages_with_text: int
    pages_empty: int
    avg_chars_per_page: float
    garbage_char_ratio: float
    duplicate_header_footer_ratio: float
    empty_chunks_ratio: float = 0.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quality_status": self.quality_status.value,
            "quality_score": round(self.quality_score, 3),
            "pages_with_text": self.pages_with_text,
            "pages_empty": self.pages_empty,
            "avg_chars_per_page": round(self.avg_chars_per_page, 1),
            "garbage_char_ratio": round(self.garbage_char_ratio, 4),
            "duplicate_header_footer_ratio": round(self.duplicate_header_footer_ratio, 4),
            "empty_chunks_ratio": round(self.empty_chunks_ratio, 4),
            "warnings": self.warnings,
            "errors": self.errors,
        }


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    source_file: str
    text: str
    page_start: int
    page_end: int
    chunk_index: int
    extraction_method: str
    quality_score: float
    processing_status: str
    created_at: str
    section_title: Optional[str] = None
    char_count: int = 0
    token_estimate: int = 0
    embedding: Optional[List[float]] = None

    def __post_init__(self) -> None:
        if not self.char_count:
            self.char_count = len(self.text)
        if not self.token_estimate:
            self.token_estimate = max(1, len(self.text) // 4)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "source_file": self.source_file,
            "text": self.text,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "chunk_index": self.chunk_index,
            "section_title": self.section_title,
            "extraction_method": self.extraction_method,
            "quality_score": round(self.quality_score, 3),
            "processing_status": self.processing_status,
            "char_count": self.char_count,
            "token_estimate": self.token_estimate,
            "created_at": self.created_at,
        }
        if self.embedding is not None:
            d["embedding"] = list(self.embedding)
        return d


@dataclass
class ProcessingReport:
    document_id: str
    source_file: str
    processing_status: ProcessingStatus
    quality_status: QualityStatus
    extraction_method: ExtractionMethod
    page_count: int
    pages_with_issues: int
    quality_score: float
    chunks_created: int
    stored_successfully: bool
    ready_for_rag: bool
    warnings: List[str]
    errors: List[str]
    processing_time_seconds: float
    created_at: str
    inspection: Optional[Dict[str, Any]] = None
    validation: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source_file": self.source_file,
            "processing_status": self.processing_status.value,
            "quality_status": self.quality_status.value,
            "extraction_method": self.extraction_method.value,
            "page_count": self.page_count,
            "pages_with_issues": self.pages_with_issues,
            "quality_score": round(self.quality_score, 3),
            "chunks_created": self.chunks_created,
            "stored_successfully": self.stored_successfully,
            "ready_for_rag": self.ready_for_rag,
            "warnings": self.warnings,
            "errors": self.errors,
            "processing_time_seconds": round(self.processing_time_seconds, 3),
            "created_at": self.created_at,
            "inspection": self.inspection,
            "validation": self.validation,
        }


def new_document_id() -> str:
    return str(uuid.uuid4())


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
