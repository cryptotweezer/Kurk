# -*- coding: utf-8 -*-
"""
audio_pipe.py
WASAPI output → VB-Audio CABLE Input (48 kHz, mono/estéreo), low-latency callback + FIFO.
- Float32 pipeline, sin escritura a disco.
- Soft-clip 0.98 para evitar clipping en OBS.
- Selección de dispositivo por substring (AUDIO_DEVICE_NAME), fallback al default.
- Drenaje de cola: pending_seconds() y wait_empty(timeout_s) para respuestas largas.
- Fallback WASAPI: si extra_settings falla (PaError -9984), reintenta sin extra_settings.
- Menos underruns: blocksize=960 (~20 ms @ 48 kHz) por defecto.
- Sanitiza NaN/Inf al encolar.
"""

from __future__ import annotations

import os
import time
import logging
import threading
from collections import deque
from typing import Optional, Deque, List

import numpy as np
import sounddevice as sd

# ---------- Config ----------
ENV_AUDIO_NAME = os.getenv("AUDIO_DEVICE_NAME", "CABLE Input")
SR = int(os.getenv("SR", "48000"))
CH = int(os.getenv("CHANNELS", "1"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
BLOCKSIZE = int(os.getenv("AUDIO_BLOCKSIZE", "960"))  # ~20 ms @ 48k

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | audio_pipe | %(message)s",
)
logger = logging.getLogger("audio_pipe")


# ---------- Utilidades de dispositivo ----------
def _find_output_device_index(name_substr: str) -> Optional[int]:
    """Busca un dispositivo de salida cuyo nombre contenga 'name_substr' (case-insensitive)."""
    try:
        devices = sd.query_devices()
    except Exception as e:
        logger.error(f"No se pudo consultar dispositivos de audio: {e}")
        return None

    name_substr_low = (name_substr or "").lower()
    for idx, dev in enumerate(devices):
        try:
            if dev.get("max_output_channels", 0) <= 0:
                continue
            name = str(dev.get("name", ""))
            if name_substr_low in name.lower():
                return idx
        except Exception:
            continue
    return None


# ---------- Clase principal ----------
class AudioPipe:
    """
    Consumidor de buffers PCM float32 mono (o estéreo duplicado) a 48 kHz, con callback en WASAPI.
    - put_pcm(): encola buffers (1-D float32 en [-1.0, 1.0]).
    - start()/stop(): controla el stream.
    - pending_seconds(): segundos pendientes en cola.
    - wait_empty(timeout_s): bloquea hasta drenar la cola o vencer el timeout.
    """

    def __init__(self, samplerate: int = SR, channels: int = CH,
                 blocksize: int = BLOCKSIZE, device_name_substr: str = ENV_AUDIO_NAME):
        assert channels in (1, 2), "Fase 1 soporta mono o estéreo (duplica mono a L/R)."

        self.samplerate = int(samplerate)
        self.channels = int(channels)
        self.blocksize = int(blocksize)
        self.device_index = _find_output_device_index(device_name_substr)

        if self.device_index is None:
            # Fallback: default output
            try:
                default_out = sd.default.device[1]  # (in, out)
            except Exception:
                default_out = None
            self.device_index = default_out
            logger.warning(
                f"No se encontró dispositivo que contenga '{device_name_substr}'. "
                f"Se usará salida por defecto: {self.device_index}"
            )
        else:
            logger.info(f"Usando dispositivo de salida index={self.device_index} (match '{device_name_substr}').")

        # FIFO de chunks (cada chunk es np.ndarray float32 mono)
        self._queue: Deque[np.ndarray] = deque()
        self._queue_lock = threading.Lock()
        self._queue_frames = 0  # total de muestras en cola

        # Rate-limit logs
        self._last_underflow_log = 0.0
        self._last_lowbuffer_log = 0.0

        # Condición para wait_empty()
        self._empty_cv = threading.Condition(self._queue_lock)

        self._stream: Optional[sd.OutputStream] = None
        self._running = False

    # ---------- API pública ----------
    def start(self):
        """Inicia el stream; si WASAPI extra_settings falla, reintenta sin ellos."""
        if self._running:
            return

        # Intento con WASAPI extra_settings (si disponible)
        extra = None
        try:
            extra = sd.WasapiSettings(exclusive=False)
        except Exception:
            extra = None

        def _open_stream(extra_settings):
            return sd.OutputStream(
                device=self.device_index,
                samplerate=self.samplerate,
                channels=self.channels,
                dtype="float32",
                blocksize=self.blocksize,
                latency="low",
                callback=self._callback,
                extra_settings=extra_settings,
            )

        # 1) Con extra_settings
        try:
            self._stream = _open_stream(extra)
            self._stream.start()
            self._running = True
            logger.info(
                f"AudioPipe iniciado (con extra={extra is not None}): sr={self.samplerate}, ch={self.channels}, "
                f"blocksize={self.blocksize}, device_index={self.device_index}"
            )
            return
        except Exception as e1:
            logger.warning(f"WASAPI extra_settings falló ({e1}); reintentando sin extra_settings...")

        # 2) Sin extra_settings
        try:
            self._stream = _open_stream(None)
            self._stream.start()
            self._running = True
            logger.info(
                f"AudioPipe iniciado (sin extra_settings): sr={self.samplerate}, ch={self.channels}, "
                f"blocksize={self.blocksize}, device_index={self.device_index}"
            )
        except Exception as e2:
            logger.error(f"Error iniciando OutputStream sin extra_settings: {e2}")
            raise

    def stop(self):
        """Detiene y cierra el stream; vacía la cola."""
        self._running = False
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass
        finally:
            self._stream = None
        with self._queue_lock:
            self._queue.clear()
            self._queue_frames = 0
            self._empty_cv.notify_all()
        logger.info("AudioPipe detenido.")

    def put_pcm(self, pcm_mono_f32: np.ndarray):
        """
        Encola un buffer PCM mono float32 en rango [-1.0, 1.0].
        Se recomienda ~100–200 ms por buffer para latencia baja y estabilidad.
        """
        if pcm_mono_f32 is None:
            return
        if not isinstance(pcm_mono_f32, np.ndarray):
            pcm_mono_f32 = np.asarray(pcm_mono_f32, dtype=np.float32)
        if pcm_mono_f32.dtype != np.float32:
            pcm_mono_f32 = pcm_mono_f32.astype(np.float32, copy=False)

        # Asegurar 1-D y sanitizar (evita NaN/Inf -> distorsión/callback error)
        pcm_mono_f32 = np.ravel(pcm_mono_f32).astype(np.float32, copy=False)
        pcm_mono_f32 = np.nan_to_num(pcm_mono_f32, nan=0.0, posinf=0.0, neginf=0.0)

        with self._queue_lock:
            self._queue.append(pcm_mono_f32)
            self._queue_frames += len(pcm_mono_f32)

    def pending_seconds(self) -> float:
        """Segundos de audio pendientes en la cola."""
        with self._queue_lock:
            return float(self._queue_frames) / float(self.samplerate)

    def wait_empty(self, timeout_s: float = 60.0) -> bool:
        """
        Bloquea hasta que la cola quede completamente vacía o venza el timeout.
        Devuelve True si se vació, False si se agotó el tiempo.
        """
        end = time.time() + max(0.0, float(timeout_s))
        with self._queue_lock:
            while self._queue_frames > 0:
                remaining = end - time.time()
                if remaining <= 0:
                    return False
                self._empty_cv.wait(timeout=remaining)
            return True

    # ---------- Callback de audio ----------
    def _callback(self, outdata, frames, time_info, status):
        """
        Callback de sounddevice. Debe escribir exactamente 'frames' muestras por canal.
        Si no hay datos, rellena con silencio. Aplica soft-clip 0.98.
        Loguea underruns y low-buffer conditions.
        """
        if status:
            # Loguear xruns/underruns con rate limit
            now = time.time()
            if now - self._last_underflow_log > 2.0:  # 1 log / 2s
                logger.warning(f"⚠️ Audio status: {status}")
                self._last_underflow_log = now

        needed = frames
        out: Optional[np.ndarray] = None
        chunks: List[np.ndarray] = []

        with self._queue_lock:
            # Detectar buffer bajo (< 2× blocksize)
            if self._queue_frames > 0 and self._queue_frames < (self.blocksize * 2):
                now = time.time()
                if now - self._last_lowbuffer_log > 2.0:
                    pending_ms = (self._queue_frames / self.samplerate) * 1000.0
                    logger.warning(
                        f"🔴 LOW BUFFER: {self._queue_frames} frames ({pending_ms:.1f} ms) — posible underrun inminente"
                    )
                    self._last_lowbuffer_log = now

            # Consumir de la cola hasta llenar 'frames'
            while needed > 0 and self._queue:
                buf = self._queue[0]
                take = min(len(buf), needed)
                if take == len(buf):
                    chunks.append(buf)
                    self._queue.popleft()
                else:
                    chunks.append(buf[:take])
                    self._queue[0] = buf[take:]
                needed -= take
                self._queue_frames -= take

            # si la cola quedó vacía, notificar a wait_empty()
            if self._queue_frames == 0:
                self._empty_cv.notify_all()

        if chunks:
            # FIX: numpy.concatenate no acepta dtype=..., convertir después
            out = np.concatenate(chunks).astype(np.float32, copy=False)

        if out is None or len(out) < frames:
            # Rellenar silencio si faltan muestras
            if out is None:
                out = np.zeros(frames, dtype=np.float32)
            else:
                pad = np.zeros(frames - len(out), dtype=np.float32)
                out = np.concatenate([out, pad]).astype(np.float32, copy=False)

        # Soft-clip a 0.98
        np.clip(out, -0.98, 0.98, out=out)

        # Expandir a (frames, channels)
        if self.channels == 1:
            outdata[:, 0] = out
        else:
            # Duplicar mono a estéreo
            outdata[:, 0] = out
            outdata[:, 1] = out


# ---------- Singleton perezoso ----------
_instance_lock = threading.Lock()
_instance: Optional[AudioPipe] = None


def init_audio_output() -> AudioPipe:
    """
    Crea (si no existe) y retorna el singleton de AudioPipe.
    No llama start(); eso lo hace app.py en startup.
    """
    global _instance
    if _instance is not None:
        return _instance
    with _instance_lock:
        if _instance is None:
            _instance = AudioPipe(samplerate=SR, channels=CH, blocksize=BLOCKSIZE, device_name_substr=ENV_AUDIO_NAME)
            logger.info("AudioPipe creado (lazy singleton).")
    return _instance
