# PDF RAG Pipeline

A local Python pipeline that ingests PDFs, extracts and normalizes text, chunks it automatically, and prepares it for use in Retrieval-Augmented Generation (RAG) systems.

---

## What it does

1. **Inspects** each PDF: detects page count, text vs. scanned content, encryption, empty pages, and layout hints.
2. **Extracts** text using a cascade strategy: PyMuPDF → pdfplumber → OCR (pytesseract).
3. **Validates** extraction quality with objective metrics (garbage ratio, text density, duplicate headers).
4. **Normalizes** the text: removes artifacts, reconstructs hyphenated words, strips repetitive headers/footers.
5. **Chunks** automatically without user intervention — structure-aware first, paragraph-based fallback.
6. **Stores** all artifacts: original PDF, raw text, normalized text, chunks (JSONL), metadata, and a processing report.
7. **Indexes** chunks in a lightweight JSONL index for RAG. Optional dense embeddings via `sentence-transformers`.
8. **Reports** clearly: per-document JSON report + rich console summary with status, warnings, and errors.

---

## Architecture

```
src/
  config.py       Configuration management (YAML + env vars)
  models.py       Data models (dataclasses): Chunk, InspectionResult, ProcessingReport, …
  inspector.py    PDF inspection — detects type, layout, encryption, empty pages
  extractor.py    Multi-strategy text extraction (PyMuPDF → pdfplumber → OCR)
  validator.py    Quality validation — garbage ratio, duplicates, density scores
  normalizer.py   Text normalization — cleans artefacts, reconstructs words, strips boilerplate
  chunker.py      Automatic chunking — structure-aware + paragraph-based + overlap
  storage.py      Local file storage for all artifacts
  indexer.py      RAG index builder (JSONL + optional embeddings)
  reporter.py     Report generation and rich console output
  ingestor.py     Pipeline orchestrator — ties all stages together
cli.py            Click-based CLI entry point
```

### Data layout

```
input/                    ← drop PDFs here (or pass any path)
data/
  raw/<doc_id>/
    original.pdf          ← copy of the original file
    raw_text.txt          ← raw extracted text
  processed/<doc_id>/
    normalized_text.txt   ← cleaned text
    chunks.jsonl          ← one chunk per line (JSON)
    metadata.json         ← document-level metadata
  reports/<doc_id>.json   ← processing report
  index/
    index.jsonl           ← global RAG index (all chunks, all docs)
```

---

## Installation

### Requirements

- Python ≥ 3.10
- (Optional) [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) for scanned PDFs

### Install

```bash
# Clone / copy the project
cd pdf-rag-pipeline

# Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Optional: install as a package (gives `pdf-pipeline` command)
pip install -e .
```

---

## Usage

### Process a single PDF

```bash
python cli.py process-pdf path/to/paper.pdf
```

### Process a folder

```bash
python cli.py process-folder ./input
python cli.py process-folder ./input --recursive
```

### Keyword search across the index

```bash
python cli.py search "neural networks" --top-k 5
```

### View index statistics

```bash
python cli.py index-stats
```

### List all processed documents

```bash
python cli.py list-docs
```

### Show full report for a document

```bash
python cli.py show-report <document-id>
```

### Global options

```bash
python cli.py --log-level DEBUG process-pdf paper.pdf
python cli.py --config /path/to/custom_config.yaml process-folder ./input
```

---

## Configuration

Edit `config.yaml` or set environment variables with prefix `PDF_PIPELINE_`:

| Key | Default | Description |
|-----|---------|-------------|
| `chunk_size` | 1000 | Target characters per chunk |
| `chunk_overlap` | 150 | Overlap between consecutive chunks |
| `min_chunk_size` | 100 | Discard chunks smaller than this |
| `ocr_enabled` | true | Enable OCR for scanned pages |
| `ocr_lang` | eng | Tesseract language code(s) |
| `ocr_dpi` | 300 | Render DPI for OCR |
| `embeddings_enabled` | false | Compute dense embeddings |
| `embeddings_model` | all-MiniLM-L6-v2 | Sentence-transformer model |
| `min_quality_score` | 0.5 | Below this → REVIEW_RECOMMENDED |
| `critical_quality_score` | 0.25 | Below this → FAILED |
| `garbage_char_threshold` | 0.05 | Garbage ratio triggering warning |
| `log_level` | INFO | DEBUG / INFO / WARNING / ERROR |

---

## Processing statuses

| Status | Meaning |
|--------|---------|
| `completed` | All stages passed, document ready for RAG |
| `completed_with_warnings` | Processed but with quality/extraction warnings |
| `failed` | Pipeline stopped due to a critical error |

Quality statuses: `OK` · `OK_WITH_WARNINGS` · `REVIEW_RECOMMENDED` · `FAILED`

---

## Running tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Library choices

| Library | Purpose | Why |
|---------|---------|-----|
| **PyMuPDF** (fitz) | Primary PDF extraction | Fast, robust, handles most digital PDFs; also renders pages for OCR |
| **pdfplumber** | Fallback extraction | Better at tables, multi-column; slower but higher quality for complex layouts |
| **pytesseract** | OCR | Industry-standard Tesseract wrapper; optional, degrades gracefully |
| **Pillow** | Image handling | Required for rendering pages before OCR |
| **click** | CLI | Declarative, composable, widely used |
| **rich** | Console output | Readable colour-coded reports; falls back to plain text if not available |
| **PyYAML** | Config | Human-readable config format |

---

## Known limitations

- **Tesseract must be installed separately** for OCR to work. The pipeline degrades gracefully (logs a warning) when it's missing.
- **Two-column layout**: pdfplumber fallback handles this better than PyMuPDF, but column ordering may still be imperfect on complex layouts.
- **Tables and figures**: text inside images in tables is not extracted; embedded table text is captured but not structured.
- **Mathematical formulas**: LaTeX-rendered formulas extract as garbled text; this is a known PDF limitation.
- **Embeddings require** `pip install sentence-transformers` and the first run downloads the model (~80 MB).
- **Keyword search** in `cli.py search` is naive (no TF-IDF / BM25). Integrate with a vector DB for production RAG.
- **Password-protected PDFs**: not supported in this version.

---

## Next improvements

1. **Table extraction** as structured data (Markdown or JSON) using `pdfplumber` table API.
2. **BM25 index** using `rank_bm25` for better keyword search.
3. **Vector DB integration**: Chroma, Qdrant, or Weaviate adapter in `indexer.py`.
4. **Async batch processing** for large folders.
5. **Metadata enrichment**: auto-detect title, authors, DOI from first pages.
6. **Incremental indexing**: skip already-processed files using a hash registry.
7. **Password-protected PDF** support via PyMuPDF's decrypt API.
8. **Better two-column handling**: use detected layout hint to re-order text blocks before joining.
