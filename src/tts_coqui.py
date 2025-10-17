# -*- coding: utf-8 -*-
"""
tts_coqui.py
Coqui XTTS v2 (GPU) → síntesis en memoria, mono float32 @ 48 kHz, sin escribir a disco.
Mejoras F1:
- enqueue() NO bloquea: se encola texto y un worker background sintetiza y alimenta audio_pipe.
- Lookahead: el worker mantiene cola de PCM siempre adelantada para evitar underflow/gaps.
- Crossfade de 50 ms entre frases para empalmes naturales (sin cortes audibles).
- Trimea silencios al inicio/final de cada frase para evitar gaps.
- Usa assets/voice.wav si existe (clon), si no, speaker por defecto (configurable).
"""

from __future__ import annotations

import os
import time
import logging
import threading
from collections import deque
from typing import Optional, List, Deque, Dict, Any

import numpy as np
import torch
import torchaudio
from TTS.api import TTS

from .audio_pipe import init_audio_output

# ---------- Config ----------
SR = int(os.getenv("SR", "48000"))
CH = int(os.getenv("CHANNELS", "1"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
DEFAULT_SPEAKER = os.getenv("XTTS_DEFAULT_SPEAKER", "female-en-5")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | tts_coqui | %(message)s",
)
logger = logging.getLogger("tts_coqui")

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
VOICE_WAV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "voice.wav")

# XTTS genera ~24 kHz; escalamos a 48 kHz.
XTTS_NATIVE_SR = 24000
CHUNK_SECONDS = 0.10                   # ~100 ms por buffer para salida fluida
TARGET_FRAMES_PER_CHUNK = int(SR * CHUNK_SECONDS)
XF_MS = 50                             # crossfade entre frases (ms) — AUMENTADO de 10 a 50
XF_SAMPLES = int(SR * XF_MS / 1000.0)  # ~2400 muestras @ 48k

# Threshold para detectar silencio (valor absoluto)
SILENCE_THRESHOLD = 0.01

# ---------- Utils ----------
def _to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        if x.shape[0] < x.shape[1]:
            x = np.mean(x, axis=0)
        else:
            x = np.mean(x, axis=1)
    return x.astype(np.float32, copy=False)

def _normalize_peak(x: np.ndarray, peak: float = 0.97) -> np.ndarray:
    m = float(np.max(np.abs(x)) + 1e-9)
    if m > peak:
        x = (x / m) * peak
    return x.astype(np.float32, copy=False)

def _resample_numpy(wave: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    wav_t = torch.from_numpy(wave).to(device=device, dtype=torch.float32)
    if wav_t.ndim == 1:
        wav_t = wav_t.unsqueeze(0)  # (1, N)
    resampled = torchaudio.functional.resample(wav_t, src_sr, dst_sr)
    resampled = resampled.squeeze(0).detach().to("cpu").numpy().astype(np.float32, copy=False)
    return resampled

def _trim_silence(wave: np.ndarray, threshold: float = SILENCE_THRESHOLD) -> np.ndarray:
    """
    Recorta silencios al inicio y final del audio.
    Silencio = samples con abs(value) < threshold.
    """
    if len(wave) == 0:
        return wave
    
    # Encontrar primer sample NO silencioso
    start = 0
    for i in range(len(wave)):
        if abs(wave[i]) >= threshold:
            start = i
            break
    
    # Encontrar último sample NO silencioso
    end = len(wave)
    for i in range(len(wave) - 1, -1, -1):
        if abs(wave[i]) >= threshold:
            end = i + 1
            break
    
    if start >= end:
        # Todo es silencio
        return np.zeros(0, dtype=np.float32)
    
    return wave[start:end]

def _chunkify(wave: np.ndarray, frames_per_chunk: int) -> List[np.ndarray]:
    if len(wave) == 0:
        return []
    n = len(wave)
    out: List[np.ndarray] = []
    i = 0
    while i < n:
        j = min(i + frames_per_chunk, n)
        out.append(wave[i:j])
        i = j
    return out

def _crossfade_join(prev_tail: Optional[np.ndarray], current: np.ndarray) -> np.ndarray:
    """
    Mezcla 50 ms del inicio de 'current' con el tail de 'prev' para suavizar empalme.
    Si no hay prev_tail suficiente, retorna current.
    """
    if prev_tail is None or len(prev_tail) < XF_SAMPLES or len(current) <= 0:
        return current
    L = min(XF_SAMPLES, len(prev_tail), len(current))
    fade_out = np.linspace(1.0, 0.0, L, dtype=np.float32)
    fade_in  = 1.0 - fade_out
    mixed = prev_tail[-L:] * fade_out + current[:L] * fade_in
    out = np.concatenate([mixed, current[L:]], dtype=np.float32)
    # Limitar pico por seguridad
    out = _normalize_peak(out, peak=0.97)
    return out

# ---------- Clase principal ----------
class TTSCoqui:
    def __init__(self, model_name: str = MODEL_NAME, voice_wav_path: Optional[str] = None, default_speaker: str = DEFAULT_SPEAKER):
        assert CH == 1, "Fase 1 requiere mono (CHANNELS=1)."

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Cargando Coqui TTS modelo='{model_name}' en device={self.device}...")
        self.tts = TTS(model_name=model_name, progress_bar=False, gpu=(self.device == "cuda"))
        try:
            self.tts.to(self.device)
        except Exception:
            pass

        # Voz
        self.voice_wav = voice_wav_path if (voice_wav_path and os.path.isfile(voice_wav_path)) else None
        self.default_speaker = default_speaker
        if self.voice_wav:
            logger.info(f"Usando clon de voz: {self.voice_wav}")
        else:
            logger.info(f"Sin voice.wav; usando speaker por defecto: '{self.default_speaker}'")

        # Audio out
        self.audio = init_audio_output()

        # Worker state
        self._text_q: Deque[str] = deque()
        self._q_lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._prev_tail: Optional[np.ndarray] = None   # para crossfade entre frases

        # Warm-up (descartado)
        try:
            self._warm_up()
        except Exception as e:
            logger.warning(f"Warm-up falló (continuamos): {e}")

        # Iniciar worker
        self._start_worker()

    # ---------- API pública ----------
    def enqueue(self, text_chunk: str):
        """
        Encola texto para síntesis (no bloquea).
        Un worker background sintetiza y va alimentando audio_pipe con lookahead.
        """
        t = (text_chunk or "").strip()
        if not t:
            return
        with self._q_lock:
            self._text_q.append(t)

    # ---------- Internals ----------
    def _start_worker(self):
        if self._worker and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._worker_loop, name="tts_worker", daemon=True)
        self._worker.start()
        logger.info("TTS worker iniciado.")

    def _stop_worker(self):
        self._stop.set()
        if self._worker:
            self._worker.join(timeout=1.0)
            self._worker = None
        logger.info("TTS worker detenido.")

    def _synthesize_xtts(self, text: str) -> np.ndarray:
        """
        XTTS v2 a ~24 kHz, mono float32 [-1,1].
        """
        if not text or not text.strip():
            return np.zeros(0, dtype=np.float32)
        kwargs: Dict[str, Any] = {"text": text.strip(), "language": "en", "split_sentences": False}
        if self.voice_wav:
            kwargs["speaker_wav"] = self.voice_wav
        else:
            kwargs["speaker"] = self.default_speaker

        t0 = time.perf_counter()
        wav = self.tts.tts(**kwargs)
        t1 = time.perf_counter()
        rtf = (t1 - t0) / (len(wav) / XTTS_NATIVE_SR + 1e-9)
        logger.info(f"XTTS synth: len={len(wav)} @24k | proc={t1 - t0:.3f}s | RTF={rtf:.2f}")
        wav = _to_mono(np.asarray(wav, dtype=np.float32))
        return wav

    def _warm_up(self):
        logger.info("Warm-up de TTS (descartado)...")
        dummy = self._synthesize_xtts("warming up.")
        if dummy.size > 0:
            _ = _normalize_peak(_resample_numpy(dummy, XTTS_NATIVE_SR, SR))
        logger.info("Warm-up TTS listo.")

    def _worker_loop(self):
        """
        Toma textos de la cola, sintetiza, aplica trim de silencios, crossfade con la frase anterior,
        trocea en ~100 ms y alimenta audio_pipe sin bloquear la UI.
        Mantiene la cola de PCM por delante para evitar underflow.
        """
        while not self._stop.is_set():
            text = None
            with self._q_lock:
                if self._text_q:
                    text = self._text_q.popleft()
            if text is None:
                time.sleep(0.005)
                continue

            # 1) Sintetizar @24k
            wav_24k = self._synthesize_xtts(text)
            if wav_24k.size == 0:
                continue

            # 2) Resample → 48k y normalizar
            wav_48k = _resample_numpy(wav_24k, XTTS_NATIVE_SR, SR)
            wav_48k = _normalize_peak(wav_48k, peak=0.97)

            # 3) ✅ NUEVO: Trimear silencios al inicio/final
            wav_48k = _trim_silence(wav_48k, threshold=SILENCE_THRESHOLD)
            if len(wav_48k) == 0:
                logger.warning(f"Frase completamente silenciosa después de trim: '{text[:60]}'")
                continue

            # 4) Crossfade con la frase anterior (50 ms)
            wav_48k = _crossfade_join(self._prev_tail, wav_48k)
            self._prev_tail = wav_48k[-XF_SAMPLES:].copy() if len(wav_48k) >= XF_SAMPLES else wav_48k[-len(wav_48k):].copy()

            # 5) Trocear y alimentar audio
            chunks = _chunkify(wav_48k, TARGET_FRAMES_PER_CHUNK)
            logger.info(f"TTS enqueue (worker): '{text[:60].strip()}...' chunks={len(chunks)}")
            for c in chunks:
                self.audio.put_pcm(c)

    # ---------- Context manager (si hiciera falta limpiar) ----------
    def __del__(self):
        try:
            self._stop_worker()
        except Exception:
            pass


# ---------- Singleton ----------
_instance: Optional[TTSCoqui] = None

def init_tts() -> TTSCoqui:
    global _instance
    if _instance is not None:
        return _instance
    voice = VOICE_WAV if os.path.isfile(VOICE_WAV) else None
    _instance = TTSCoqui(model_name=MODEL_NAME, voice_wav_path=voice, default_speaker=DEFAULT_SPEAKER)
    logger.info("TTSCoqui creado (lazy singleton).")
    return _instance