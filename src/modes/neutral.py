"""
modes/neutral.py — Fase 1
Un modo = un archivo. Este módulo define el preprocesado para el modo 'neutral'.

Reglas:
- Responder SIEMPRE en inglés, conciso (1–2 líneas).
- No repetir ni parafrasear la entrada del usuario.
- Sin preámbulos, disculpas ni comentarios meta.
- Evitar markdown salvo que se pida explícitamente.

Para agregar nuevos modos en Fase 2+:
- Crear un archivo nuevo en src/modes/<modo>.py con una función preprocess_prompt(text) homóloga.
"""

def preprocess_prompt(user_text: str) -> str:
    system = (
        "You are a neutral, concise assistant. "
        "Answer ONLY in English, in short sentences (1–2 lines), "
        "without repeating or paraphrasing the user's input, "
        "no preambles, no apologies, no meta-comments. "
        "Avoid markdown unless explicitly requested."
    )

    text = (user_text or "").replace("\r", " ").replace("\n", " ").strip()
    return f"{system}\n\nUser question: {text}\nAssistant:"
