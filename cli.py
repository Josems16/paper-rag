#!/usr/bin/env python3
"""
PDF RAG Pipeline — Command-Line Interface.

Usage examples:
  python cli.py process-pdf paper.pdf
  python cli.py process-folder ./input --recursive
  python cli.py search "neural networks" --top-k 5
  python cli.py index-stats
  python cli.py list-docs
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

# Ensure src package is importable when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config
from src.ingestor import ingest_folder, ingest_pdf
from src.indexer import get_index_stats, search_index
from src.storage import list_processed_documents, load_report

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
    )
    # Silence noisy third-party loggers
    for noisy in ("pdfminer", "PIL", "pdfplumber"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.option("--config", "-c", default=None, help="Path to config.yaml")
@click.option(
    "--log-level", default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Logging verbosity",
)
@click.pass_context
def cli(ctx: click.Context, config: str, log_level: str) -> None:
    """PDF ingestion pipeline for RAG."""
    _setup_logging(log_level)
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@cli.command("process-pdf")
@click.argument("pdf_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--quiet", "-q", is_flag=True, help="Suppress console report")
@click.pass_context
def process_pdf(ctx: click.Context, pdf_path: str, quiet: bool) -> None:
    """Ingest a single PDF file."""
    cfg = ctx.obj["config"]
    report = ingest_pdf(pdf_path, config=cfg, verbose=not quiet)

    # Exit with non-zero code on failure so scripts can detect it
    if report.processing_status.value == "failed":
        sys.exit(1)


@cli.command("process-folder")
@click.argument("folder_path", type=click.Path(exists=True, file_okay=False))
@click.option("--recursive", "-r", is_flag=True, help="Search subdirectories")
@click.option("--quiet", "-q", is_flag=True, help="Suppress per-file reports")
@click.pass_context
def process_folder(
    ctx: click.Context,
    folder_path: str,
    recursive: bool,
    quiet: bool,
) -> None:
    """Ingest all PDFs in a folder."""
    cfg = ctx.obj["config"]
    reports = ingest_folder(
        folder_path,
        config=cfg,
        verbose=not quiet,
        recursive=recursive,
    )

    if not reports:
        click.echo("No PDF files found.")
        return

    # Summary
    total = len(reports)
    ok = sum(1 for r in reports if r.processing_status.value == "completed")
    warn = sum(1 for r in reports if "warnings" in r.processing_status.value)
    failed = sum(1 for r in reports if r.processing_status.value == "failed")

    click.echo(f"\n{'='*50}")
    click.echo(f"Batch summary: {total} file(s)")
    click.echo(f"  Completed           : {ok}")
    click.echo(f"  Completed w/warnings: {warn}")
    click.echo(f"  Failed              : {failed}")
    click.echo(f"{'='*50}")

    if failed > 0:
        sys.exit(1)


@cli.command("search")
@click.argument("query")
@click.option("--top-k", "-k", default=5, help="Number of results to return")
@click.pass_context
def search(ctx: click.Context, query: str, top_k: int) -> None:
    """Keyword search across the index."""
    cfg = ctx.obj["config"]
    results = search_index(query, cfg, top_k=top_k)
    if not results:
        click.echo("No results found.")
        return
    for i, r in enumerate(results, 1):
        click.echo(f"\n[{i}] {r.get('source_file', '?')} — page {r.get('page_start', '?')}")
        click.echo(f"    Section : {r.get('section_title', 'n/a')}")
        click.echo(f"    Score   : {r.get('quality_score', 0):.2f}")
        preview = r.get("text", "")[:200].replace("\n", " ")
        click.echo(f"    Preview : {preview}…")


@cli.command("index-stats")
@click.pass_context
def index_stats(ctx: click.Context) -> None:
    """Show statistics about the current RAG index."""
    cfg = ctx.obj["config"]
    stats = get_index_stats(cfg)
    click.echo(f"Total chunks   : {stats['total_chunks']}")
    click.echo(f"Documents      : {len(stats['documents'])}")
    click.echo(f"Has embeddings : {stats['has_embeddings']}")
    if stats["documents"]:
        click.echo("Document IDs:")
        for doc_id in stats["documents"]:
            click.echo(f"  {doc_id}")


@cli.command("list-docs")
@click.pass_context
def list_docs(ctx: click.Context) -> None:
    """List all processed documents."""
    cfg = ctx.obj["config"]
    doc_ids = list_processed_documents(cfg)
    if not doc_ids:
        click.echo("No processed documents found.")
        return
    click.echo(f"{'ID':<40}  {'File':<40}  Status")
    click.echo("-" * 90)
    for doc_id in doc_ids:
        report = load_report(doc_id, cfg)
        if report:
            click.echo(
                f"{doc_id:<40}  {report.get('source_file', '?'):<40}  "
                f"{report.get('processing_status', '?')}"
            )
        else:
            click.echo(f"{doc_id:<40}  (no report)")


@cli.command("show-report")
@click.argument("document_id")
@click.pass_context
def show_report(ctx: click.Context, document_id: str) -> None:
    """Show the processing report for a document ID."""
    import json
    cfg = ctx.obj["config"]
    report = load_report(document_id, cfg)
    if not report:
        click.echo(f"No report found for document ID: {document_id}", err=True)
        sys.exit(1)
    click.echo(json.dumps(report, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
