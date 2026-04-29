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

---

## RAG Query — consultas directas desde línea de comandos

`query.py` es una herramienta de depuración standalone que llama a `rag_answer`
internamente y **requiere Anthropic API key**. Para el uso normal desde Claude Code,
usa las tools MCP en su lugar.

```bash
# Solo si tienes ANTHROPIC_API_KEY configurada (no Claude Pro)
python query.py "¿Qué parámetros de curado afectan la densidad en DLP cerámico?"
python query.py --top-k 8 "¿Cuál es la viscosidad óptima de la resina?"
```

---

## MCP Server — integración con Claude Code

El servidor MCP expone el RAG como herramientas nativas de Claude Code.
Una vez configurado, Claude Code recupera evidencia desde ChromaDB y responde
directamente en la conversación usando tu sesión de Claude Pro — sin costes de API.

### Claude Pro vs Anthropic API — diferencia importante

| | Claude Pro | Anthropic API |
|---|---|---|
| Qué es | Suscripción web/app en claude.ai | Servicio de acceso programático |
| Facturación | Cuota mensual fija | Por tokens consumidos |
| Cómo se usa | Navegador o Claude Code con tu cuenta | `ANTHROPIC_API_KEY` en código |
| Necesaria para `rag_sources` | No | No |
| Necesaria para `rag_answer` | No | **Sí** |

> **En resumen:** con Claude Pro puedes usar `rag_search`, `rag_sources` y `rag_status`
> directamente desde Claude Code. `rag_answer` es opcional y requiere facturación de API separada.

### Tools disponibles

| Tool | Requiere API | Descripción |
|---|---|---|
| `rag_status` | No | Estado del índice y configuración |
| `rag_search` | No | Búsqueda semántica con ranking y preview |
| `rag_sources` | No | Fuentes detalladas: paper, página, sección, preview |
| `rag_answer` | **Sí (billing)** | Respuesta generada por LLM con citas — opcional |
| `search_papers` | No | Búsqueda semántica (herramienta original) |
| `list_papers` | No | Lista todos los documentos indexados |
| `get_paper_info` | No | Metadata completa de un documento |
| `get_paper_chunks` | No | Chunks ordenados de un documento |
| `knowledge_base_stats` | No | Estadísticas globales |

### Iniciar el servidor MCP

```bash
# Desde la raíz del proyecto:
python mcp_server.py
```

El servidor usa `stdio` como transporte (requerido por Claude Code).

### Configurar en Claude Code

Crea `.mcp.json` en la **raíz del proyecto Claude Code** (el directorio desde el que abres Claude Code, no necesariamente la carpeta del pipeline). Este archivo contiene rutas absolutas específicas de cada máquina — **no lo versiones en Git** (ya está en `.gitignore`).

```json
{
  "mcpServers": {
    "papers-rag": {
      "command": "<RUTA_ABSOLUTA_A_PYTHON>",
      "args": ["mcp_server.py"],
      "cwd": "<RUTA_ABSOLUTA_A_ESTA_CARPETA>"
    }
  }
}
```

> **Nota de configuración local:** Los valores de `command` y `cwd` son
> rutas absolutas de tu máquina. Cada desarrollador debe ajustarlos.
> Nunca incluyas estas rutas en el repositorio Git.

**Cómo obtener la ruta correcta a Python:**

```bash
# En la terminal del proyecto:
python -c "import sys; print(sys.executable)"
```

**Cómo obtener la ruta a esta carpeta:**

```bash
# Windows PowerShell:
(Get-Location).Path
# bash / Git Bash:
pwd
```

**Activar el servidor en Claude Code** — añade en `.claude/settings.local.json`
(o en `~/.claude.json` bajo la entrada del proyecto):

```json
{ "enableAllProjectMcpServers": true }
```

No incluyas `ANTHROPIC_API_KEY` en el bloque `env` a menos que quieras usar `rag_answer`.

### Verificar la conexión

Después de reiniciar Claude Code, ejecuta desde la interfaz:

```
Usa rag_status para comprobar si el sistema RAG está listo.
```

### Flujo recomendado (Claude Pro, sin API billing)

```
1. rag_status    → verificar que el índice tiene datos
2. rag_sources   → recuperar evidencia con top_k suficiente (5-10)
3. [en conversación] → Claude Code lee los fragmentos y responde
                       citando paper, página y sección
```

### Flujo alternativo (solo con Anthropic API key)

```
1. rag_status    → verificar que ANTHROPIC_API_KEY está configurada
2. rag_answer    → recuperación + respuesta generada automáticamente
```

### Ejemplos de preguntas desde Claude Code

```
"Usa rag_status para ver si el índice está listo."

"Usa rag_sources con top_k=8 para encontrar evidencia sobre
 viscosidad de resinas cerámicas en DLP."

"Usa rag_search para buscar resultados de resistencia mecánica
 en piezas fabricadas por estereolitografía."
```

### Variables de entorno

| Variable | Requerida por | Descripción |
|---|---|---|
| `ANTHROPIC_API_KEY` | `rag_answer` (opcional) | API de Anthropic — no Claude Pro |

`rag_search`, `rag_sources` y `rag_status` funcionan **sin ninguna variable de entorno**.

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
