"""
LLM adapter: sends a RAG-grounded question to Claude and returns the answer.

Reads ANTHROPIC_API_KEY from the environment — never hardcoded.
Raises RuntimeError on configuration or import errors instead of calling
sys.exit(), so callers (CLI or MCP server) can handle errors gracefully.
"""

from __future__ import annotations

import os

_SYSTEM_PROMPT = """\
Eres un asistente científico. Responde ÚNICAMENTE con la información contenida \
en los fragmentos de papers académicos que se te proporcionan.

Reglas estrictas:
1. Usa SOLO los fragmentos de contexto para responder.
2. Cita cada afirmación relevante indicando la clave del paper y la página o \
sección cuando estén disponibles, por ejemplo: (Smith2023, p. 4) o \
(García2024, sec. 3.2).
3. Si los fragmentos no contienen evidencia suficiente para responder, \
indícalo explícitamente: "La evidencia disponible en los fragmentos \
recuperados no es suficiente para responder esta pregunta."
4. No inventes datos, resultados, autores ni referencias.
5. Responde en español.\
"""


def query_with_rag(question: str, retrieved_context: str) -> str:
    """
    Call Claude with RAG context and return a grounded answer in Spanish.

    Raises:
        RuntimeError: if ANTHROPIC_API_KEY is missing or 'anthropic' is not installed.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "La variable de entorno ANTHROPIC_API_KEY no está definida.\n"
            "Defínela con:\n"
            "  export ANTHROPIC_API_KEY='sk-ant-...'   (Linux/macOS)\n"
            "  $env:ANTHROPIC_API_KEY='sk-ant-...'     (PowerShell)"
        )

    try:
        import anthropic
    except ImportError:
        raise RuntimeError(
            "El paquete 'anthropic' no está instalado.\n"
            "Instálalo con: pip install anthropic"
        )

    client = anthropic.Anthropic(api_key=api_key)

    user_message = (
        "Fragmentos recuperados de los papers:\n\n"
        f"{retrieved_context}\n\n"
        "---\n\n"
        f"Pregunta: {question}"
    )

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    return message.content[0].text
