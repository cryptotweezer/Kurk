"""
modes/neutral.py � Fase 1
Un modo = un archivo. Este m�dulo define el preprocesado para el modo 'neutral'.

Reglas:
- Responder SIEMPRE en ingl�s, conciso (1�2 l�neas).
- No repetir ni parafrasear la entrada del usuario.
- Sin pre�mbulos, disculpas ni comentarios meta.
- Evitar markdown salvo que se pida expl�citamente.

Para agregar nuevos modos en Fase 2+:
- Crear un archivo nuevo en src/modes/<modo>.py con una funci�n preprocess_prompt(text) hom�loga.
"""

def preprocess_prompt(user_text: str) -> str:
    system = (
        "You are a neutral, concise assistant. "
        "Answer ONLY in English, in short sentences (1�2 lines), "
        "without repeating or paraphrasing the user's input, "
        "no preambles, no apologies, no meta-comments. "
        "Avoid markdown unless explicitly requested."
    )

    text = (user_text or "").replace("\r", " ").replace("\n", " ").strip()
    return f"{system}\n\nUser question: {text}\nAssistant:"
