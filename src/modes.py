# -*- coding: utf-8 -*-
"""
modes.py
Placeholder de "modo neutral" para Fase 1.
- Fuerza respuestas en inglés, concisas, sin eco del input.
- No agrega personalidad avanzada (se deja para fases futuras).

API:
  preprocess(user_text: str) -> str   # devuelve el prompt final para el LLM
"""

from __future__ import annotations

import re


_SYSTEM_PREAMBLE = (
    "You are a concise assistant. Reply in plain English, 1–2 short sentences, "
    "clear and helpful. Do not repeat the user's question. No preambles. No lists "
    "unless explicitly requested."
)


def _compact_spaces(text: str) -> str:
    # Normaliza espacios y recorta
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def _strip_control(text: str) -> str:
    # Elimina caracteres de control/raros que a veces rompen streaming
    return "".join(ch for ch in text if ch == "\n" or (31 < ord(ch) < 0x110000))


def preprocess(user_text: str) -> str:
    """
    Construye el prompt final para F1 (neutral). No eco del input en la respuesta.
    Estrategia: pedimos respuesta breve en inglés, explícitamente sin repetir.
    """
    user_text = _strip_control(_compact_spaces(user_text))
    if not user_text:
        return _SYSTEM_PREAMBLE + "\n\nUser: Provide a very brief helpful fact.\nAssistant:"

    # Instrucción explícita de no eco (para modelos que tienden a parafrasear)
    no_echo_rule = (
        "Do not restate, quote, or echo the user's text. "
        "Answer directly and briefly."
    )

    prompt = (
        f"{_SYSTEM_PREAMBLE}\n"
        f"{no_echo_rule}\n\n"
        f"User: {user_text}\n"
        f"Assistant:"
    )
    return prompt
