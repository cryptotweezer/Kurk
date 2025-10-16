"""
tts_coqui.py — Fase 1
XTTS v2 (Coqui) en GPU con síntesis por chunks (sin escribir a disco).
Convierte texto -> audio numpy, re-muestrea a 48 kHz mono y lo envía a audio_pipe.

Depende de:
- coqui-ai TTS==0.22.0
- torch con CUDA (RTX 3060)
- soxr (resample de alta calidad)
- audio_pipe.write_audio(np.ndarray float32 [N,] mono a SAMPLE_RATE)

API expuesta:
- init_tts(voice_sample_path: str, sample_rate: int) -> None
- async stream_tts_from_text_chunks(text_chunk: str) -> None
"""

from __future__ import annotations
import asyncio
import os
from typing import Optional

import numpy as np
import torch
from TTS.api import TTS
import soxr  # viene vía dependencias (librosa/soxr)

# Audio sink
from .audio_pipe import write_audio  # implementado en el siguiente paso

# ====== Estado global mínimo (Fase 1) ======
_TTS_MODEL: Optional[TTS] = None
_TTS_SR_NATIVE: int = 22050  # XTTS v2 normalmente genera a 22.05 kHz
_OUT_SR: int = 48000
_VOICE_WAV: Optional[str] = None
_LANG: str = "en"

# Ajustes de entrega
_PEAK_LIMIT = 0.95  # limiter suave para evitar clipping
_PREEMPHASIS = 0.0  # 0.0 = off (podemos ajustar luego)
_FADE_HEAD_MS = 5    # fundido suave al inicio para evitar clics
_FADE_TAIL_MS = 8    # fundido suave al final para evitar clics


def _fade_edges(x: np.ndarray, sr: int, head_ms: int, tail_ms: int) -> np.ndarray:
    if x.size == 0:
        return x
    y = x.copy()
    n_head = max(1, int(sr * head_ms / 1000.0))
    n_tail = max(1, int(sr * tail_ms / 1000.0))
    n = y.shape[0]
    n_head = min(n_head, n)
    n_tail = min(n_tail, n)
    # fade-in
    if n_head > 1:
        ramp = np.linspace(0.0, 1.0, n_head, dtype=np.float32)
        y[:n_head] *= ramp
    # fade-out
    if n_tail > 1:
        ramp = np.linspace(1.0, 0.0, n_tail, dtype=np.float32)
        y[-n_tail:] *= ramp
    return y


def _to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    # promedio canales si viene estéreo
    return x.mean(axis=1)


def _normalize_peak(x: np.ndarray, peak: float = _PEAK_LIMIT) -> np.ndarray:
    m = np.max(np.abs(x)) if x.size else 0.0
    if m > 0:
        scale = min(1.0, peak / m)
        if scale < 1.0:
            x = x * scale
    return x


def _resample_if_needed(x: np.ndarray, in_sr: int, out_sr: int) -> np.ndarray:
    if in_sr == out_sr:
        return x
    # soxr pide float32
    x = x.astype(np.float32, copy=False)
    return soxr.resample(x, in_sr, out_sr, quality="HQ")


def _synthesize_sync(text_chunk: str) -> np.ndarray:
    """
    Llama XTTS en modo síncrono y devuelve audio mono float32 a SR nativa del modelo.
    """
    global _TTS_MODEL, _VOICE_WAV, _TTS_SR_NATIVE, _LANG

    if _TTS_MODEL is None:
        raise RuntimeError("TTS no inicializado. Llama init_tts() primero.")

    # Coqui TTS devuelve numpy (float32) con rango [-1, 1]
    # Usamos voice cloning con 'speaker_wav'
    audio = _TTS_MODEL.tts(
        text=text_chunk,
        speaker_wav=_VOICE_WAV,
        language=_LANG,
        split_sentences=False,  # ya recibimos chunks oracionales
    )

    # Asegurar mono
    audio = _to_mono(np.asarray(audio, dtype=np.float32))

    # Fundidos suaves para evitar clics entre chunks
    audio = _fade_edges(audio, _TTS_SR_NATIVE, _FADE_HEAD_MS, _FADE_TAIL_MS)

    # Normalizar pico
    audio = _normalize_peak(audio, _PEAK_LIMIT)

    return audio


def init_tts(voice_sample_path: str, sample_rate: int = 48000) -> None:
    """
    Carga el modelo XTTS v2, fija dispositivo CUDA si disponible y hace warmup corto.
    """
    global _TTS_MODEL, _VOICE_WAV, _OUT_SR, _TTS_SR_NATIVE

    _OUT_SR = int(sample_rate)
    _VOICE_WAV = voice_sample_path

    if not os.path.isfile(_VOICE_WAV):
        raise FileNotFoundError(f"VOICE_SAMPLE no encontrado: {_VOICE_WAV}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Modelo XTTS v2 multilingüe
    _TTS_MODEL = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(device)
    # Nota: La SR nativa de XTTS v2 es 22050. Si cambiase en futuras versiones, detectar:
    try:
        _TTS_SR_NATIVE = getattr(_TTS_MODEL, "output_sample_rate", 22050)  # fallback
    except Exception:
        _TTS_SR_NATIVE = 22050

    # Warmup corto para “despertar” CUDA y cachear pesos
    _ = _synthesize_sync("ready")


async def stream_tts_from_text_chunks(text_chunk: str) -> None:
    """
    Recibe un texto corto (frase/oración) y lo sintetiza de forma asíncrona.
    El audio resultante se re-muestrea a _OUT_SR, se asegura mono float32
    y se envía inmediatamente a audio_pipe.write_audio().
    """
    if not text_chunk or not text_chunk.strip():
        return

    # Ejecutar TTS en hilo para no bloquear el loop asyncio
    audio_native = await asyncio.to_thread(_synthesize_sync, text_chunk.strip())

    # Re-muestrear a la frecuencia de salida solicitada (48 kHz)
    audio = _resample_if_needed(audio_native, _TTS_SR_NATIVE, _OUT_SR)

    # Asegurar float32 mono
    audio = _to_mono(np.asarray(audio, dtype=np.float32))

    # (Opcional) pre-énfasis leve si se requiere más claridad en altas (apagado por defecto)
    if _PREEMPHASIS > 0:
        # y[n] = x[n] - a * x[n-1]
        y = np.empty_like(audio)
        y[0] = audio[0]
        y[1:] = audio[1:] - _PREEMPHASIS * audio[:-1]
        audio = y

    # Empujar al dispositivo de salida (VB-CABLE) via audio_pipe
    write_audio(audio)
