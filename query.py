#!/usr/bin/env python3
"""
CLI de consulta RAG sobre los papers indexados en ChromaDB.

Uso:
    python query.py "tu pregunta"
    python query.py --top-k 8 "tu pregunta"
"""

from __future__ import annotations

import argparse
import sys
import logging

# Silence noisy loggers from chromadb / sentence-transformers
logging.basicConfig(level=logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consulta RAG sobre papers académicos indexados.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            '  python query.py "¿Qué técnicas de fabricación aditiva se comparan?"\n'
            '  python query.py --top-k 8 "¿Cuál es la viscosidad óptima de la resina?"\n'
        ),
    )
    parser.add_argument("question", nargs="+", help="Pregunta a responder")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        metavar="N",
        help="Número de fragmentos a recuperar (default: 5)",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Ruta al archivo config.yaml (opcional)",
    )
    args = parser.parse_args()

    question = " ".join(args.question)

    from src.config import load_config
    import src.rag_service as svc

    config = load_config(args.config)

    # Show index status first
    st = svc.status(config=config)
    if not st["chroma_available"]:
        print(
            "Error: ChromaDB no está disponible.\n"
            "Instala las dependencias con: pip install chromadb sentence-transformers",
            file=sys.stderr,
        )
        sys.exit(1)

    if st["total_chunks"] == 0:
        print(
            "El índice está vacío. Procesa primero los papers con:\n"
            "  python cli.py process-folder <carpeta_con_pdfs>",
            file=sys.stderr,
        )
        sys.exit(0)

    print(
        f"Índice: {st['total_chunks']} fragmentos de "
        f"{st['document_count']} documento(s)."
    )
    print(f"Buscando top-{args.top_k} fragmentos para: {question!r}\n")

    # Search
    try:
        hits = svc.search(question, top_k=args.top_k, config=config)
    except RuntimeError as exc:
        print(f"Error en búsqueda: {exc}", file=sys.stderr)
        sys.exit(1)

    if not hits:
        print("No se encontraron fragmentos relevantes para esta pregunta.")
        sys.exit(0)

    print(f"Fragmentos recuperados: {len(hits)}")
    for i, r in enumerate(hits, 1):
        label = r.get("citation_key") or r.get("document_id", "?")
        page = r.get("page_start", "?")
        dist = r.get("distance", "?")
        print(f"  [{i}] {label}  pág. {page}  distancia={dist}")

    print("\nConsultando al LLM...\n")

    result = svc.answer(question, top_k=args.top_k, config=config)

    if result["warning"]:
        print(f"Advertencia: {result['warning']}", file=sys.stderr)
        if not result["answer"]:
            sys.exit(1)

    print("=" * 64)
    print("RESPUESTA")
    print("=" * 64)
    print(result["answer"])
    print("=" * 64)


if __name__ == "__main__":
    main()
