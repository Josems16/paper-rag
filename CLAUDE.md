# CLAUDE.md — Paper RAG Pipeline

## Qué es este proyecto

Pipeline local de ingesta y consulta RAG (Retrieval-Augmented Generation) sobre papers académicos en PDF. Los papers se procesan con `cli.py`, se indexan en ChromaDB con embeddings semánticos, y pueden consultarse mediante el servidor MCP o desde la línea de comandos.

---

## Uso del RAG desde Claude Code

Cuando el usuario pregunte sobre papers, literatura científica, PDFs indexados o resultados de artículos, usa primero las tools MCP del servidor `papers-rag`.

### Flujo recomendado (Claude Pro — sin Anthropic API)

Este es el flujo principal. No requiere `ANTHROPIC_API_KEY` ni facturación de API.

1. Usa `rag_status` para comprobar que el índice tiene datos.
2. Usa `rag_sources` (o `rag_search`) con `top_k` suficiente (5–10) para recuperar evidencia.
3. Lee los fragmentos recuperados y responde directamente en esta conversación usando solo esa evidencia.
4. No inventes información que no esté en los chunks recuperados.
5. Cita siempre paper, página, sección o chunk_id cuando estén disponibles.
6. Si la evidencia recuperada es insuficiente, dilo explícitamente.

### Flujo alternativo con `rag_answer` (requiere Anthropic API billing)

`rag_answer` llama a la API de Anthropic por separado y genera cobro por tokens,
independientemente de tu suscripción a Claude Pro.
Úsala solo si `ANTHROPIC_API_KEY` está configurada y el usuario la solicita explícitamente.

> **Claude Pro no equivale a Anthropic API key.**
> Claude Pro es una suscripción web/app. La API de Anthropic es un servicio
> distinto con facturación por tokens. Son cuentas y pagos independientes.

---

## Estructura del proyecto

```
src/
  config.py          Configuración (YAML + env vars)
  models.py          Dataclasses: Chunk, ProcessingReport, …
  chromadb_index.py  Índice semántico ChromaDB + semantic_search()
  rag_service.py     Capa de servicio RAG reutilizable (search, answer, status)
  llm_adapter.py     Llamada a Claude con prompt RAG estricto
  context_builder.py Formatea hits de búsqueda en bloques de contexto
  ingestor.py        Orquestador del pipeline de ingesta
  indexer.py         Construcción del índice JSONL + embeddings
  storage.py         Almacenamiento de artefactos por documento
  chunker.py         Chunking estructural + párrafo + fallback
  extractor.py       Extracción multiestrategy: PyMuPDF → pdfplumber → OCR
  normalizer.py      Normalización: artefactos, rehifenación, encabezados
  validator.py       Validación de calidad con métricas objetivas
  inspector.py       Inspección de PDF: layout, cifrado, ratio de texto
  academic_meta.py   Extracción de título, autores, DOI, año
  reporter.py        Generación de reportes JSON y consola
  image_extractor.py Extracción de imágenes por página

cli.py              CLI principal de ingesta y búsqueda
query.py            CLI de consulta RAG con LLM (debug/standalone)
mcp_server.py       Servidor MCP — expone herramientas RAG a Claude Code
config.yaml         Parámetros de chunking, calidad, embeddings, OCR
```

## Datos

```
data/
  raw/<doc_id>/          PDF original + texto extraído
  processed/<doc_id>/    Texto normalizado, chunks.jsonl, metadata.json
  reports/<doc_id>.json  Reporte de procesamiento
  index/index.jsonl      Índice RAG global (todos los docs)
  chroma/                ChromaDB persistente
```

---

## Comandos frecuentes

```bash
# Ingestar papers
python cli.py process-folder ./input
python cli.py process-pdf paper.pdf

# Consulta directa con LLM (debug)
python query.py "¿Qué parámetros afectan la densidad en DLP cerámico?"
python query.py --top-k 8 "¿Cuál es la viscosidad óptima de la resina?"

# Servidor MCP (Claude Code lo inicia automáticamente si está configurado)
python mcp_server.py

# Tests
pytest tests/ -v
pytest tests/test_rag_service.py -v
pytest tests/test_context_builder.py -v
```

---

## Configuracion de ANTHROPIC_API_KEY (opcional — solo para rag_answer)

Esta sección solo aplica si decides usar `rag_answer` en el futuro.
Para el flujo recomendado con Claude Pro no necesitas configurar nada aquí.

**Nunca escribas la clave en archivos del repositorio.** `.env` y `.env.*`
están en `.gitignore`. Solo `.env.example` (sin clave real) se versiona.

### Windows PowerShell — variable de entorno permanente de usuario

```powershell
setx ANTHROPIC_API_KEY "sk-ant-..."
```

> Despues de ejecutar `setx` debes **cerrar y reabrir** Claude Code y la
> terminal para que la variable sea visible. `setx` escribe en el registro
> del usuario pero no actualiza la sesion actual.

Para verificar en una nueva terminal:

```powershell
echo $env:ANTHROPIC_API_KEY
```

### Linux / macOS — bash o zsh

Añade esta linea a `~/.bashrc`, `~/.zshrc` o `~/.profile`:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Luego recarga el shell:

```bash
source ~/.bashrc   # o source ~/.zshrc
```

### Alternativa: solo para la sesion actual

```powershell
# PowerShell (solo sesion actual, no persiste)
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

```bash
# bash/zsh (solo sesion actual)
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Verificar con rag_status

Una vez configurada la variable, comprueba desde Claude Code:

```
Usa rag_status para verificar que ANTHROPIC_API_KEY esta configurada.
```

La salida mostrara `ANTHROPIC_API_KEY : Configurada` sin revelar el valor.

### Configuracion MCP sin clave hardcodeada

El archivo `.claude/settings.json` del proyecto no incluye la clave.
El servidor MCP hereda `ANTHROPIC_API_KEY` del entorno del sistema.
Edita `cwd` con la ruta absoluta real al proyecto:

```json
{
  "mcpServers": {
    "papers-rag": {
      "command": "python",
      "args": ["mcp_server.py"],
      "cwd": "C:\\Users\\<usuario>\\paper-rag-tmp"
    }
  }
}
```

---

## Variables de entorno

| Variable | Módulo que la usa | Descripción |
|---|---|---|
| `ANTHROPIC_API_KEY` | `llm_adapter.py`, `rag_service.answer()` | Requerida para `rag_answer` |
| `PDF_PIPELINE_<KEY>` | `config.py` | Override de cualquier campo de Config |

`rag_search`, `rag_sources` y `rag_status` funcionan sin `ANTHROPIC_API_KEY`.

---

## Configuración MCP para Claude Code

Añadir en `.claude/settings.json` (proyecto) o `~/.claude/settings.json` (global).
No incluyas `ANTHROPIC_API_KEY` aquí — no es necesaria para el flujo recomendado:

```json
{
  "mcpServers": {
    "papers-rag": {
      "command": "python",
      "args": ["mcp_server.py"],
      "cwd": "<RUTA_ABSOLUTA_AL_PROYECTO>"
    }
  }
}
```

---

## Restricciones del pipeline actual (iteración 1)

- Sin BM25 — solo búsqueda semántica por embeddings
- Sin extracción de tablas estructuradas
- Sin re-ranking
- `rag_answer` requiere Anthropic API billing (no Claude Pro) — no es el flujo principal
