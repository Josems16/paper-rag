"""
RAG index builder.

Maintains a global index file (data/index/index.jsonl) that aggregates
all chunks across documents.  Optionally computes dense embeddings.

Embeddings are a soft dependency (sentence-transformers).  If not installed,
the index is built without them and the module logs a clear notice.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from .config import Config
from .models import Chunk

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False
    logger.debug(
        "sentence-transformers not installed; embeddings will not be computed. "
        "Install with: pip install sentence-transformers"
    )

INDEX_FILENAME = "index.jsonl"
MODEL_CACHE: Optional[object] = None  # lazy singleton


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def index_chunks(
    chunks: List[Chunk],
    document_id: str,
    config: Config,
) -> bool:
    """
    Add chunks to the global index.  Computes embeddings if configured.
    Returns True on success, False on failure.
    """
    if not chunks:
        logger.warning("No chunks to index for document %s", document_id)
        return False

    config.index_dir.mkdir(parents=True, exist_ok=True)

    # Optionally compute embeddings
    if config.embeddings_enabled:
        chunks = _attach_embeddings(chunks, config)

    index_path = config.index_dir / INDEX_FILENAME
    try:
        with open(index_path, "a", encoding="utf-8") as f:
            for chunk in chunks:
                record = chunk.to_dict()
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Indexed %d chunks for document %s", len(chunks), document_id)
        return True
    except Exception as exc:
        logger.error("Failed to write index: %s", exc)
        return False


def remove_document_from_index(document_id: str, config: Config) -> int:
    """
    Remove all chunks belonging to document_id from the index.
    Returns the number of records removed.
    """
    index_path = config.index_dir / INDEX_FILENAME
    if not index_path.exists():
        return 0

    kept: List[str] = []
    removed = 0

    with open(index_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if record.get("document_id") == document_id:
                    removed += 1
                else:
                    kept.append(line)
            except json.JSONDecodeError:
                kept.append(line)

    with open(index_path, "w", encoding="utf-8") as f:
        f.writelines(kept)

    return removed


def search_index(
    query: str,
    config: Config,
    top_k: int = 5,
) -> List[dict]:
    """
    Simple keyword search over the index (no embeddings required).
    Returns top_k chunks whose text contains all query terms.

    For vector search, use the embedding field with your preferred vector DB.
    """
    index_path = config.index_dir / INDEX_FILENAME
    if not index_path.exists():
        return []

    terms = query.lower().split()
    results = []

    with open(index_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                text = record.get("text", "").lower()
                if all(t in text for t in terms):
                    results.append(record)
                    if len(results) >= top_k:
                        break
            except json.JSONDecodeError:
                continue

    return results


def get_index_stats(config: Config) -> dict:
    """Return basic statistics about the current index."""
    index_path = config.index_dir / INDEX_FILENAME
    if not index_path.exists():
        return {"total_chunks": 0, "documents": [], "has_embeddings": False}

    total = 0
    documents: set = set()
    has_embeddings = False

    with open(index_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                total += 1
                documents.add(record.get("document_id", "unknown"))
                if record.get("embedding"):
                    has_embeddings = True
            except json.JSONDecodeError:
                continue

    return {
        "total_chunks": total,
        "documents": sorted(documents),
        "has_embeddings": has_embeddings,
    }


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def _attach_embeddings(chunks: List[Chunk], config: Config) -> List[Chunk]:
    """Compute and attach embeddings to chunks using sentence-transformers."""
    global MODEL_CACHE

    if not _ST_AVAILABLE:
        logger.warning(
            "sentence-transformers not available; skipping embeddings. "
            "Install with: pip install sentence-transformers"
        )
        return chunks

    if MODEL_CACHE is None:
        logger.info("Loading embedding model: %s", config.embeddings_model)
        try:
            MODEL_CACHE = SentenceTransformer(config.embeddings_model)
        except Exception as exc:
            logger.error("Failed to load embedding model: %s", exc)
            return chunks

    model = MODEL_CACHE
    texts = [c.text for c in chunks]
    batch_size = config.embeddings_batch_size

    try:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb.tolist()
        logger.debug("Embeddings computed for %d chunks", len(chunks))
    except Exception as exc:
        logger.error("Embedding computation failed: %s", exc)

    return chunks
