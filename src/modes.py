"""
modes.py — Fase 1
Stubs de personalidad/modos (sin lógica avanzada).
Provee `preprocess_prompt` para asegurar salida en inglés y evitar eco.
"""

from typing import Optional

_NEUTRAL_SYS = (
    "You are a neutral, concise assistant. "
    "Answer ONLY in English, in short sentences (1–2 lines), "
    "without repeating or paraphrasing the user's input, "
    "no preambles, no apologies, no meta-comments. "
    "Avoid markdown unless explicitly requested."
)

def preprocess_prompt(user_text: str, mode: Optional[str] = "neutral") -> str:
    """
    Fase 1: sin personalidad. Forzamos salida en inglés y prohibimos eco.
    Dejamos estructura para crecer en Fase 2+ (monologue, debate, etc.).
    """
    text = (user_text or "").strip()

    # Sanitización mínima anti-eco
    # (en Fase 2 se podrá aplicar ranking, memoria, jailbreak guard, etc.)
    text = text.replace("\r", " ").replace("\n", " ").strip()

    # En Fase 1 ignoramos `mode` y usamos un encabezado estilo sistema embebido.
    # El cliente Realtime usará este prompt tal cual.
    prompt = f"{_NEUTRAL_SYS}\n\nUser question: {text}\nAssistant:"
    return prompt
