"""
Report generation and console output.

Generates a structured ProcessingReport and prints a human-readable
summary to the console using rich (or plain text if unavailable).
"""

from __future__ import annotations

import logging
from typing import List, Optional

from .models import (
    ExtractionMethod,
    ExtractionResult,
    InspectionResult,
    ProcessingReport,
    ProcessingStatus,
    QualityStatus,
    ValidationResult,
    now_iso,
)

logger = logging.getLogger(__name__)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

_console = None


def _get_console():
    global _console
    if _console is None and _RICH_AVAILABLE:
        from rich.console import Console
        _console = Console()
    return _console


# ---------------------------------------------------------------------------
# Report building
# ---------------------------------------------------------------------------

def build_report(
    document_id: str,
    source_file: str,
    inspection: Optional[InspectionResult],
    extraction: Optional[ExtractionResult],
    validation: Optional[ValidationResult],
    chunks_created: int,
    stored_successfully: bool,
    indexed_successfully: bool,
    processing_status: ProcessingStatus,
    processing_time: float,
    extra_warnings: Optional[List[str]] = None,
    extra_errors: Optional[List[str]] = None,
) -> ProcessingReport:
    """Assemble all information into a ProcessingReport."""

    warnings: List[str] = list(extra_warnings or [])
    errors: List[str] = list(extra_errors or [])

    if inspection:
        warnings.extend(inspection.warnings)
        errors.extend(inspection.errors)
    if extraction:
        warnings.extend(extraction.warnings)
        errors.extend(extraction.errors)
    if validation:
        warnings.extend(validation.warnings)
        errors.extend(validation.errors)

    # Remove duplicates while preserving order
    warnings = _deduplicate(warnings)
    errors = _deduplicate(errors)

    quality_status = (
        validation.quality_status
        if validation else QualityStatus.FAILED
    )
    quality_score = validation.quality_score if validation else 0.0
    extraction_method = (
        extraction.method if extraction else ExtractionMethod.FAILED
    )
    page_count = inspection.page_count if inspection else 0

    # Count pages with any issue
    pages_with_issues = 0
    if extraction:
        pages_with_issues = sum(
            1 for p in extraction.pages
            if p.warnings or not p.has_text
        )

    ready_for_rag = (
        stored_successfully
        and indexed_successfully
        and quality_status != QualityStatus.FAILED
        and chunks_created > 0
    )

    return ProcessingReport(
        document_id=document_id,
        source_file=source_file,
        processing_status=processing_status,
        quality_status=quality_status,
        extraction_method=extraction_method,
        page_count=page_count,
        pages_with_issues=pages_with_issues,
        quality_score=quality_score,
        chunks_created=chunks_created,
        stored_successfully=stored_successfully,
        ready_for_rag=ready_for_rag,
        warnings=warnings,
        errors=errors,
        processing_time_seconds=processing_time,
        created_at=now_iso(),
        inspection=inspection.to_dict() if inspection else None,
        validation=validation.to_dict() if validation else None,
    )


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_report(report: ProcessingReport, verbose: bool = False) -> None:
    """Print a human-readable summary of the processing report."""
    if _RICH_AVAILABLE:
        _print_rich(report, verbose)
    else:
        _print_plain(report, verbose)


def _print_rich(report: ProcessingReport, verbose: bool) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    from rich.text import Text

    console = Console()

    # Status colour
    status_colors = {
        ProcessingStatus.COMPLETED: "green",
        ProcessingStatus.COMPLETED_WITH_WARNINGS: "yellow",
        ProcessingStatus.FAILED: "red",
    }
    quality_colors = {
        QualityStatus.OK: "green",
        QualityStatus.OK_WITH_WARNINGS: "yellow",
        QualityStatus.REVIEW_RECOMMENDED: "orange3",
        QualityStatus.FAILED: "red",
    }

    status_color = status_colors.get(report.processing_status, "white")
    quality_color = quality_colors.get(report.quality_status, "white")

    table = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
    table.add_column("Field", style="bold cyan", width=28)
    table.add_column("Value")

    table.add_row("File", report.source_file)
    table.add_row("Document ID", report.document_id)
    table.add_row(
        "Status",
        Text(report.processing_status.value.upper(), style=status_color),
    )
    table.add_row(
        "Quality",
        Text(
            f"{report.quality_status.value}  ({report.quality_score:.2f})",
            style=quality_color,
        ),
    )
    table.add_row("Extraction method", report.extraction_method.value)
    table.add_row("Pages", str(report.page_count))
    table.add_row("Pages with issues", str(report.pages_with_issues))
    table.add_row("Chunks created", str(report.chunks_created))
    table.add_row("Stored", "[green]yes[/green]" if report.stored_successfully else "[red]no[/red]")
    table.add_row("Ready for RAG", "[green]yes[/green]" if report.ready_for_rag else "[red]no[/red]")
    table.add_row("Processing time", f"{report.processing_time_seconds:.2f}s")

    console.print(
        Panel(table, title=f"[bold]PDF Processing Report[/bold]", border_style=status_color)
    )

    if report.warnings:
        console.print("[yellow]Warnings:[/yellow]")
        for w in report.warnings:
            console.print(f"  [yellow]![/yellow] {w}")

    if report.errors:
        console.print("[red]Errors:[/red]")
        for e in report.errors:
            console.print(f"  [red]x[/red] {e}")


def _print_plain(report: ProcessingReport, verbose: bool) -> None:
    sep = "-" * 60
    print(sep)
    print("PDF Processing Report")
    print(sep)
    print(f"  File            : {report.source_file}")
    print(f"  Document ID     : {report.document_id}")
    print(f"  Status          : {report.processing_status.value.upper()}")
    print(f"  Quality         : {report.quality_status.value} ({report.quality_score:.2f})")
    print(f"  Extraction      : {report.extraction_method.value}")
    print(f"  Pages           : {report.page_count}")
    print(f"  Pages w/ issues : {report.pages_with_issues}")
    print(f"  Chunks created  : {report.chunks_created}")
    print(f"  Stored          : {'yes' if report.stored_successfully else 'no'}")
    print(f"  Ready for RAG   : {'yes' if report.ready_for_rag else 'no'}")
    print(f"  Processing time : {report.processing_time_seconds:.2f}s")
    if report.warnings:
        print("  Warnings:")
        for w in report.warnings:
            print(f"    ⚠ {w}")
    if report.errors:
        print("  Errors:")
        for e in report.errors:
            print(f"    ✗ {e}")
    print(sep)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deduplicate(items: List[str]) -> List[str]:
    seen: set = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
