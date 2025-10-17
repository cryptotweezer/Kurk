# -*- coding: utf-8 -*-
"""
app.py — FastAPI backend (Phase 1)
Endpoints:
  - GET /health  → {"status":"ok","model":..., "sr":48000, "ch":1, "pending_seconds": 0.0}
  - POST /say    → {first_audio_ms, total_ms, chunk_count}
"""

from __future__ import annotations

import os
import time
import asyncio
import logging
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi import Body
from pydantic import BaseModel
from dotenv import load_dotenv

# -------------------- CARGA ROBUSTA DE .env (RUTA ABSOLUTA + PARSEO MANUAL) --------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # ...\KURK
DOTENV_PATH = PROJECT_ROOT / ".env"

def _force_load_env(dotenv_path: Path) -> None:
    """Carga .env con dotenv y, si falta OPENAI_API_KEY, hace parseo manual (utf-8-sig)."""
    # 1) Intento estándar (override=True)
    try:
        load_dotenv(dotenv_path=dotenv_path, override=True, encoding="utf-8")
    except Exception:
        pass

    # 2) Si OPENAI_API_KEY sigue vacía, parseo manual
    if not (os.getenv("OPENAI_API_KEY") or "").strip():
        try:
            with open(dotenv_path, "r", encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")  # quita comillas si las hay
                    # Evitar valores tipo 'sk-proj' placeholder
                    if k == "OPENAI_API_KEY" and (not v or len(v) < 20):
                        continue
                    os.environ[k] = v
        except Exception:
            # si falla, seguimos; validaremos abajo
            pass

    # 3) Log de confirmación (mascarilla)
    _k = (os.getenv("OPENAI_API_KEY") or "").strip()
    if _k:
        masked = _k[:6] + "..." if len(_k) >= 6 else "***"
        logging.getLogger("app_boot").info(f".env cargado desde: {dotenv_path} | OPENAI_API_KEY={masked}")
    else:
        logging.getLogger("app_boot").warning(f"No se pudo cargar OPENAI_API_KEY desde: {dotenv_path}")

# Ejecuta la carga robusta ANTES de leer cualquier env
_force_load_env(DOTENV_PATH)

# -------------------- Logging --------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | app | %(message)s",
)
logger = logging.getLogger("app")

# -------------------- Env / constants --------------------
SR = int(os.getenv("SR", "48000"))
CH = int(os.getenv("CHANNELS", "1"))
REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-4o-realtime-preview")
AUDIO_DRAIN_TIMEOUT_S = float(os.getenv("AUDIO_DRAIN_TIMEOUT_S", "60.0"))

# -------------------- Safe imports (both package & local execution) --------------------
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

# -------------------- FastAPI app --------------------
app = FastAPI(title="KURK F1 Backend", version="1.1.3")

# shared singletons
_audio = None
_tts = None


class SayIn(BaseModel):
    text: str


@app.on_event("startup")
def _on_startup():
    """Init audio + TTS; validar API key; arrancar audio output y warm-up TTS."""
    # Validación de API key (fail-fast)
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        logger.error(f"OPENAI_API_KEY no configurada. Revisé: {DOTENV_PATH}")
        raise RuntimeError("OPENAI_API_KEY missing")

    global _audio, _tts
    _audio = init_audio_output()
    _audio.start()
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
    """Readiness + estado de la cola de audio (útil para TUI/OBS)."""
    pending = None
    try:
        pending = round(_audio.pending_seconds(), 3) if _audio else None
    except Exception:
        pending = None
    return {
        "status": "ok",
        "model": REALTIME_MODEL,
        "sr": SR,
        "ch": CH,
        "pending_seconds": pending,
    }


@app.post("/say")
async def say(payload: SayIn = Body(...)):
    """
    Accepts {"text": "..."} and streams LLM → TTS in-process, returning latency metrics.
    Guardrails:
      - No audio persisted to disk — buffers only.
      - Paragraph-level chunking triggers TTS enqueue ASAP.
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

    # Stream sentences/paragraphs from LLM and enqueue to TTS
    try:
        async for sentence in stream_text_chunks(prompt):
            if sentence:
                if t1_first_delta is None:
                    t1_first_delta = time.perf_counter()
                _tts.enqueue(sentence)  # tts_coqui trocea internamente
                chunk_count += 1
                if t2_first_audio is None:
                    t2_first_audio = time.perf_counter()
    except Exception as e:
        logger.error(f"/say pipeline error: {e}")
        raise HTTPException(status_code=502, detail=f"llm/tts pipeline failed: {e}")

    # Esperar drenaje de audio para evitar cortes al final
    logger.info("Esperando drenaje completo de audio...")
    try:
        drained = await asyncio.to_thread(_audio.wait_empty, timeout_s=AUDIO_DRAIN_TIMEOUT_S)
        if not drained:
            logger.warning(f"Audio drain timeout ({AUDIO_DRAIN_TIMEOUT_S:.1f}s) — respuesta puede estar incompleta")
    except Exception as e:
        logger.error(f"Error en wait_empty: {e}")

    t3_done = time.perf_counter()

    # Metrics helpers
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

    logger.info(
        "metrics | first_delta_ms=%.1f first_audio_ms=%.1f total_ms=%.1f chunks=%d",
        metrics["llm_to_first_delta_ms"] or -1.0,
        metrics["first_audio_ms"] or -1.0,
        metrics["total_ms"] or -1.0,
        chunk_count,
    )
    return metrics
