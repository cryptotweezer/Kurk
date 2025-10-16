# -*- coding: utf-8 -*-
"""
modes/neutral.py - Phase 1
One mode = one file. This module defines preprocessing for the "neutral" mode.

Rules:
- ALWAYS answer in English, concise (1–2 lines).
- Do not repeat or paraphrase the user input.
- No preambles, no apologies, no meta-comments.
- Avoid markdown unless explicitly requested.

To add new modes in Phase 2+:
- Create a new file at src/modes/<mode>.py with a homologous preprocess_prompt(text).
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
