"""
KURK – Phase 1
FastAPI backend mínimo con endpoint /say.

Ejecución:
(.venv) PS C:\AI_Workspace\Kurk> uvicorn src.app:app --host 127.0.0.1 --port 8000 --reload

Notas:
- Backend expone /say para pruebas externas (POST con {"text": "..."}).
- Orquesta: OpenAI Realtime (tokens parciales) -> Coqui XTTS -> CABLE Input.
- Sin persistir audio (streaming en memoria).
- Imprime métricas de latencia en logs (timestamps).
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import time
import os

from dotenv import load_dotenv

# Carga variables de entorno
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-4o-realtime-preview")
VOICE_SAMPLE = os.getenv("VOICE_SAMPLE", "assets/voice.wav")
AUDIO_SR = int(os.getenv("AUDIO_SAMPLE_RATE", "48000"))
AUDIO_CH = int(os.getenv("AUDIO_CHANNELS", "1"))

if not OPENAI_API_KEY:
    raise RuntimeError("Falta OPENAI_API_KEY en .env")

# ===== Imports internos (cada módulo aislado) =====
from .llm_realtime import stream_completion  # async generator de tokens parciales
from .tts_coqui import init_tts, stream_tts_from_text_chunks  # inicializa TTS y sintetiza en streaming
from .audio_pipe import init_audio_out, close_audio_out        # gestiona salida a CABLE Input (VB-Audio)
from .modes.neutral import preprocess_prompt                   # MODO NEUTRAL (1 archivo = 1 modo)

app = FastAPI(title="KURK Phase 1 API", version="0.1.0")

# CORS mínimo para pruebas locales
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SayIn(BaseModel):
    text: str
    mode: Optional[str] = "neutral"  # placeholder (para Fase 2+)

@app.on_event("startup")
async def on_startup():
    # Inicializa salida de audio y TTS una sola vez (GPU warmup)
    print("[startup] inicializando audio y TTS…")
    init_audio_out(sample_rate=AUDIO_SR, channels=AUDIO_CH)
    init_tts(voice_sample_path=VOICE_SAMPLE, sample_rate=AUDIO_SR)
    print("[startup] listo.")

@app.on_event("shutdown")
async def on_shutdown():
    print("[shutdown] cerrando audio…")
    try:
        close_audio_out()
    except Exception as e:
        print(f"[shutdown] warning al cerrar audio: {e}")

@app.get("/health")
async def health():
    return {"status": "ok", "model": REALTIME_MODEL, "sr": AUDIO_SR, "ch": AUDIO_CH}

@app.post("/say")
async def say(payload: SayIn):
    """
    Recibe texto, obtiene respuesta con tokens parciales desde OpenAI Realtime
    y sintetiza en voz clonada con Coqui XTTS v2 en streaming hacia CABLE Input.
    """
    user_text = payload.text.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="text vacío")

    # Fase 1: no repetir input del usuario (sin /echo)
    prompt = preprocess_prompt(user_text)

    t0 = time.perf_counter()
    print(f"[/say] ⏱️ start ts={t0:.6f} prompt='{prompt[:120]}'…")

    tokens_count = 0
    first_audio_ts: Optional[float] = None

    try:
        async for chunk in stream_completion(
            api_key=OPENAI_API_KEY,
            model=REALTIME_MODEL,
            prompt=prompt
        ):
            if not chunk:
                continue
            tokens_count += 1

            if first_audio_ts is None:
                first_audio_ts = time.perf_counter()

            # TTS en streaming con prebuffer ~100–200 ms y control de backpressure
            await stream_tts_from_text_chunks(chunk)

    except Exception as e:
        print(f"[ERROR][/say] {e}")
        raise HTTPException(status_code=500, detail=f"Fallo en pipeline: {e}")

    t_end = time.perf_counter()
    total = (t_end - t0) * 1000.0
    first_audio_ms = ((first_audio_ts - t0) * 1000.0) if first_audio_ts else None

    print(f"[/say] ✅ done tokens={tokens_count} total_ms={total:.1f} first_audio_ms={first_audio_ms:.1f if first_audio_ms else -1}")

    return {
        "ok": True,
        "tokens": tokens_count,
        "latency_ms": round(total, 1),
        "first_audio_ms": round(first_audio_ms, 1) if first_audio_ms else None
    }
