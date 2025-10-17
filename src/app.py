# -*- coding: utf-8 -*-
"""
app.py — FastAPI backend (Phase 1)
Endpoints:
  - GET /health  → {"status":"ok","model":..., "sr":48000, "ch":1}
  - POST /say    → {first_audio_ms, total_ms, chunk_count}

Pipeline:
  text → (modes.preprocess) → OpenAI Realtime (partial tokens) → sentence chunks
       → Coqui XTTS v2 (GPU) chunked 48k mono float32 → AudioPipe (VB-CABLE) → OBS

Run (PowerShell, project root):
  C:\AI_Workspace\KURK\.venv\Scripts\Activate.ps1
  uvicorn src.app:app --host 127.0.0.1 --port 8000 --reload
"""

from __future__ import annotations

import os
import time
import asyncio
import logging
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi import Body
from pydantic import BaseModel
from dotenv import load_dotenv

# --------- Logging ---------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | app | %(message)s",
)
logger = logging.getLogger("app")

# --------- Safe imports (both package & local execution) ---------
try:
    # When running as "uvicorn src.app:app"
    from src.audio_pipe import init_audio_output
    from src.tts_coqui import init_tts
    from src.llm_realtime import stream_text_chunks
    from src import modes
except Exception:
    # Fallback for alternative run modes
    from .audio_pipe import init_audio_output  # type: ignore
    from .tts_coqui import init_tts  # type: ignore
    from .llm_realtime import stream_text_chunks  # type: ignore
    from . import modes  # type: ignore

# --------- Env / constants ---------
SR = int(os.getenv("SR", "48000"))
CH = int(os.getenv("CHANNELS", "1"))
REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-4o-realtime-preview")

# --------- FastAPI app ---------
app = FastAPI(title="KURK F1 Backend", version="1.0.0")

# shared singletons
_audio = None
_tts = None


class SayIn(BaseModel):
    text: str


@app.on_event("startup")
def _on_startup():
    """Load .env, init audio + TTS, start audio output and warm up TTS."""
    # .env first (so runtime env is available if not set by shell)
    try:
        if load_dotenv():
            logger.info(".env loaded.")
    except Exception as e:
        logger.warning(f"No se pudo cargar .env automáticamente: {e}")

    global _audio, _tts
    # Init audio (but do not synth here)
    _audio = init_audio_output()
    _audio.start()

    # Init TTS (includes warm-up internally)
    _tts = init_tts()

    logger.info(f"Startup OK. SR={SR}, CH={CH}, MODEL={REALTIME_MODEL}")


@app.on_event("shutdown")
def _on_shutdown():
    try:
        if _audio:
            _audio.stop()
    except Exception:
        pass


@app.get("/health")
def health():
    """Simple readiness check."""
    return {
        "status": "ok",
        "model": REALTIME_MODEL,
        "sr": SR,
        "ch": CH,
    }


@app.post("/say")
async def say(payload: SayIn = Body(...)):
    """
    Accepts {"text": "..."} and streams LLM → TTS in-process, returning latency metrics.
    Guardrails:
      - No audio persisted to disk — buffers only.
      - Sentence-level chunking triggers TTS enqueue ASAP.
      - Waits for audio queue to drain before returning (prevents cut-off in long responses).
    """
    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    # Prepare prompt (neutral, concise, English, no echo)
    prompt = modes.preprocess(text)

    # Timings
    t0_recv = time.perf_counter()
    t1_first_delta: Optional[float] = None
    t2_first_audio: Optional[float] = None
    chunk_count = 0

    # Stream sentences from LLM and enqueue to TTS
    try:
        async for sentence in stream_text_chunks(prompt):
            if sentence:
                if t1_first_delta is None:
                    t1_first_delta = time.perf_counter()
                # TTS synth + enqueue (chunked internally)
                _tts.enqueue(sentence)
                chunk_count += 1
                if t2_first_audio is None:
                    # Approximate "first audible" as first enqueue time
                    t2_first_audio = time.perf_counter()
    except Exception as e:
        logger.error(f"/say pipeline error: {e}")
        raise HTTPException(status_code=502, detail=f"llm/tts pipeline failed: {e}")

    # ✅ NUEVO: Esperar a que la cola de audio se drene completamente
    # Esto evita que respuestas largas se corten antes de terminar
    logger.info("Esperando drenaje completo de audio...")
    try:
        drained = await asyncio.to_thread(_audio.wait_empty, timeout_s=60.0)
        if not drained:
            logger.warning("Audio drain timeout (60s) — respuesta puede estar incompleta")
    except Exception as e:
        logger.error(f"Error en wait_empty: {e}")

    t3_done = time.perf_counter()

    # Metrics
    def ms(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None:
            return None
        return (b - a) * 1000.0

    metrics = {
        "t0_recv": t0_recv,
        "t1_first_delta": t1_first_delta,
        "t2_first_audio": t2_first_audio,
        "t3_done": t3_done,
        "llm_to_first_delta_ms": ms(t0_recv, t1_first_delta),
        "first_audio_ms": ms(t0_recv, t2_first_audio),
        "total_ms": ms(t0_recv, t3_done),
        "chunk_count": chunk_count,
    }

    # Log compact
    logger.info(
        "metrics | first_delta_ms=%.1f first_audio_ms=%.1f total_ms=%.1f chunks=%d",
        metrics["llm_to_first_delta_ms"] or -1.0,
        metrics["first_audio_ms"] or -1.0,
        metrics["total_ms"] or -1.0,
        chunk_count,
    )

    return metrics