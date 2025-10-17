# -*- coding: utf-8 -*-
"""
llm_realtime.py — Modo SSE-Only (rápido) + primera frase corta + párrafos word-safe
- Evita el timeout del WebSocket Realtime y va directo a /v1/chat/completions con stream=True.
- Instruye: "First sentence under 10 words. Then 2–3 short paragraphs."
- Segmentación word-safe en modo "párrafo": emitir oraciones completas; si no hay fin aún,
  solo cortar cuando buffer > MAX_CHARS y SIEMPRE en el último espacio (no partir palabras).
- Nunca emite "." sueltos.

Objetivo: bajar "texto→primer audio" quitando el penalti del WS, y sonar natural (sin staccato ni sílabas cortadas).
"""

from __future__ import annotations

import os
import re
import json
import time
import asyncio
import logging
from typing import AsyncGenerator, Tuple, List

import httpx

# ---------- Logging ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | llm_realtime | %(message)s",
)
logger = logging.getLogger("llm_realtime")

# ---------- Tunables (modo párrafo) ----------
FIRST_DELTA_LOG = True                 # loggear tiempo al primer delta
MAX_CHARS = 360                        # si no hay fin de oración, cortar aquí
MIN_CHARS_TO_EMIT = 140                # evita trozos demasiado cortos
SENT_END = re.compile(r"(?<=[\.!?])\s+")  # fin de oración

# ---------- Env helpers ----------
def _env(key: str, default: str = "") -> str:
    v = os.getenv(key)
    return v.strip() if isinstance(v, str) else default

def _cfg():
    api_key = _env("OPENAI_API_KEY")
    fallback_model = _env("FALLBACK_MODEL", "gpt-4o-mini")
    sse_url = "https://api.openai.com/v1/chat/completions"
    return api_key, fallback_model, sse_url

# ---------- Helpers de texto ----------
def _append_with_period(text: str) -> str:
    t = text.strip()
    if not t:
        return t
    if t[-1] not in ".!?":
        t += "."
    return t

def _split_paragraph_mode(buffer: str) -> Tuple[List[str], str]:
    """
    Devuelve (ready_chunks, residue) SIN cortar palabras:
      1) Si hay fin de oración [.?!], emitir oraciones completas.
      2) Si no hay fin y buffer >= MAX_CHARS, cortar en ÚLTIMO ESPACIO antes del límite.
      3) Si no, esperar más texto (no emitir).
    Regla: nunca emitir '.' sueltos.
    """
    ready: List[str] = []
    residue = buffer

    if not buffer:
        return ready, residue

    # (1) Oraciones completas
    parts = SENT_END.split(buffer)
    if len(parts) > 1:
        *complete, residue = parts
        for sent in complete:
            s = sent.strip()
            if not s:
                continue
            ready.append(_append_with_period(s))
        return ready, residue

    # (2) Longitud con corte en palabra
    if len(buffer) >= MAX_CHARS:
        cut = buffer[:MAX_CHARS]
        sp = cut.rfind(" ")
        if sp == -1:
            sp = MAX_CHARS
        head = buffer[:sp].rstrip()
        tail = buffer[sp:].lstrip()
        if len(head) >= MIN_CHARS_TO_EMIT:
            ready.append(_append_with_period(head))
            residue = tail
        else:
            residue = buffer
        return ready, residue

    return ready, residue

# ---------- SSE rápido (sin WS) ----------
async def _sse_stream(prompt: str) -> AsyncGenerator[str, None]:
    api_key, model, sse_url = _cfg()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY no configurado.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    system = (
        "First sentence under 10 words. Then 2–3 short paragraphs. "
        "Be coherent, natural, and avoid mid-word breaks."
    )
    payload = {
        "model": model,
        "stream": True,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }

    buffer = ""
    t0 = time.perf_counter()
    first_delta = False

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        async with client.stream("POST", sse_url, headers=headers, json=payload) as r:
            async for line in r.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    obj = json.loads(data_str)
                except Exception:
                    continue

                choice = obj.get("choices", [{}])[0]
                delta = choice.get("delta", {})
                text_piece = delta.get("content", "")
                if not text_piece:
                    continue

                if not first_delta and FIRST_DELTA_LOG:
                    first_delta = True
                    t1 = time.perf_counter()
                    logger.info(f"SSE primer delta en {(t1 - t0)*1000:.1f} ms")

                buffer += text_piece
                ready, residue = _split_paragraph_mode(buffer)
                for chunk in ready:
                    yield chunk.strip()
                buffer = residue

    tail = buffer.strip()
    if tail:
        yield _append_with_period(tail)

# ---------- Public API ----------
async def stream_text_chunks(prompt: str):
    prompt = (prompt or "").strip()
    if not prompt:
        return
    # Camino único: SSE rápido (evitamos el timeout del WS hasta resolverlo en otra iteración)
    async for chunk in _sse_stream(prompt):
        yield chunk
