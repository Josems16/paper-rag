"""Configuration management for the PDF RAG pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

BASE_DIR = Path(__file__).parent.parent


@dataclass
class Config:
    # --- Paths ---
    input_dir: Path = field(default_factory=lambda: BASE_DIR / "input")
    data_dir: Path = field(default_factory=lambda: BASE_DIR / "data")
    raw_dir: Path = field(default_factory=lambda: BASE_DIR / "data" / "raw")
    processed_dir: Path = field(default_factory=lambda: BASE_DIR / "data" / "processed")
    reports_dir: Path = field(default_factory=lambda: BASE_DIR / "data" / "reports")
    index_dir: Path = field(default_factory=lambda: BASE_DIR / "data" / "index")

    # --- Extraction thresholds ---
    # Pages with fewer chars than this are considered "low text"
    min_chars_per_page: int = 50
    # If fewer than this fraction of pages have good text, trigger OCR
    ocr_trigger_ratio: float = 0.4

    # --- Chunking ---
    chunk_size: int = 1000       # target chars per chunk
    chunk_overlap: int = 150     # overlap between consecutive chunks
    min_chunk_size: int = 100    # discard chunks smaller than this
    max_chunk_size: int = 3000   # force-split chunks larger than this

    # --- Quality thresholds ---
    garbage_char_threshold: float = 0.05   # >5% garbage → warning
    min_quality_score: float = 0.5         # below this → REVIEW_RECOMMENDED
    critical_quality_score: float = 0.25   # below this → FAILED

    # --- OCR ---
    ocr_enabled: bool = True
    ocr_lang: str = "eng"
    ocr_dpi: int = 300

    # --- Embeddings / ChromaDB (semantic search) ---
    # Model used both for ChromaDB and optional standalone embeddings.
    # 'paraphrase-multilingual-MiniLM-L12-v2' handles Spanish + English papers.
    embeddings_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    embeddings_batch_size: int = 32
    # Legacy flag kept for backward compatibility (ChromaDB always uses embeddings)
    embeddings_enabled: bool = True

    # --- ChromaDB ---
    chroma_dir: Path = field(default_factory=lambda: BASE_DIR / "data" / "chroma")

    # --- Academic metadata ---
    extract_academic_metadata: bool = True

    # --- Logging ---
    log_level: str = "INFO"

    def ensure_dirs(self) -> None:
        """Create all required data directories."""
        for d in [self.input_dir, self.raw_dir, self.processed_dir,
                  self.reports_dir, self.index_dir, self.chroma_dir]:
            d.mkdir(parents=True, exist_ok=True)


def load_config(config_path: Optional[str] = None) -> Config:
    """Load config from YAML file with env-var overrides."""
    cfg = Config()

    # 1. Load from YAML
    if config_path:
        yaml_path = Path(config_path)
    else:
        yaml_path = BASE_DIR / "config.yaml"

    if yaml_path.exists() and _YAML_AVAILABLE:
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        _apply_dict(cfg, data)
    elif yaml_path.exists() and not _YAML_AVAILABLE:
        import warnings
        warnings.warn("PyYAML not installed; config.yaml will be ignored.")

    # 2. Env-var overrides  (PDF_PIPELINE_<KEY>=value)
    _apply_env_overrides(cfg)

    return cfg


def _apply_dict(cfg: Config, data: dict) -> None:
    path_fields = {
        "input_dir", "data_dir", "raw_dir", "processed_dir",
        "reports_dir", "index_dir", "chroma_dir",
    }
    for key, val in data.items():
        if not hasattr(cfg, key):
            continue
        if key in path_fields:
            setattr(cfg, key, Path(val))
        else:
            setattr(cfg, key, val)


def _apply_env_overrides(cfg: Config) -> None:
    prefix = "PDF_PIPELINE_"
    for key, val in os.environ.items():
        if not key.startswith(prefix):
            continue
        attr = key[len(prefix):].lower()
        if not hasattr(cfg, attr):
            continue
        current = getattr(cfg, attr)
        if isinstance(current, bool):
            setattr(cfg, attr, val.lower() in ("1", "true", "yes"))
        elif isinstance(current, int):
            setattr(cfg, attr, int(val))
        elif isinstance(current, float):
            setattr(cfg, attr, float(val))
        elif isinstance(current, Path):
            setattr(cfg, attr, Path(val))
        else:
            setattr(cfg, attr, val)
