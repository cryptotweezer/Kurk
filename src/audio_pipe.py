# -*- coding: utf-8 -*-
"""
audio_pipe.py
WASAPI output → VB-Audio CABLE Input (48 kHz, mono), low-latency callback + FIFO.
- Float32 pipeline, sin escritura a disco.
- Soft-clip 0.98 para evitar clipping en OBS.
- Selección de dispositivo por substring (AUDIO_DEVICE_NAME), fallback al default.

Env (.env):
  AUDIO_DEVICE_NAME=CABLE Input (VB-Audio Virtual Cable)
  SR=48000
  CHANNELS=1
  LOG_LEVEL=INFO

API:
  init_audio_output() -> AudioPipe
  audio = init_audio_output(); audio.start(); audio.put_pcm(np.ndarray[float32, mono])
"""

from __future__ import annotations

import os
import time
import logging
import threading
from collections import deque
from typing import Optional

import numpy as np
import sounddevice as sd


# ---------- Config ----------
ENV_AUDIO_NAME = os.getenv("AUDIO_DEVICE_NAME", "CABLE Input")
SR = int(os.getenv("SR", "48000"))
CH = int(os.getenv("CHANNELS", "1"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | audio_pipe | %(message)s",
)
logger = logging.getLogger("audio_pipe")


# ---------- Utilidades de dispositivo ----------
def _find_output_device_index(name_substr: str) -> Optional[int]:
    """
    Busca un dispositivo de salida cuyo nombre contenga 'name_substr' (case-insensitive).
    Retorna el índice de 'sounddevice' o None si no encuentra.
    """
    try:
        devices = sd.query_devices()
    except Exception as e:
        logger.error(f"No se pudo consultar dispositivos de audio: {e}")
        return None

    name_substr_low = (name_substr or "").lower()
    best = None
    for idx, dev in enumerate(devices):
        try:
            if dev.get("max_output_channels", 0) <= 0:
                continue
            name = str(dev.get("name", ""))
            if name_substr_low in name.lower():
                best = idx
                break
        except Exception:
            continue

    return best


# ---------- Clase principal ----------
class AudioPipe:
    """
    Consumidor de buffers PCM float32 mono a 48 kHz (por defecto), con callback en WASAPI.
    - put_pcm(): encola buffers (1-D float32 en [-1.0, 1.0], mono).
    - start()/stop(): controla el stream.
    """

    def __init__(self, samplerate: int = SR, channels: int = CH, blocksize: int = 480,
                 device_name_substr: str = ENV_AUDIO_NAME):
        assert channels == 1, "Fase 1 requiere mono (CHANNELS=1)."

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
                f"Se usará el dispositivo de salida por defecto: {self.device_index}"
            )
        else:
            logger.info(f"Usando dispositivo de salida index={self.device_index} (match '{device_name_substr}').")

        # FIFO de chunks (cada chunk es np.ndarray float32 mono)
        self._queue = deque()
        self._queue_lock = threading.Lock()

        self._stream: Optional[sd.OutputStream] = None
        self._running = False

        # Para rate-limiting de logs de underflow
        self._last_underflow_log = 0.0

    # ---------- API pública ----------
    def start(self):
        """Inicia el stream WASAPI en modo baja latencia."""
        if self._running:
            return

        try:
            extra = sd.WasapiSettings(exclusive=False, low_latency=True)
        except Exception:
            # En algunos entornos la clase puede no estar disponible; continuar sin extra settings.
            extra = None

        try:
            self._stream = sd.OutputStream(
                device=self.device_index,
                samplerate=self.samplerate,
                channels=self.channels,
                dtype="float32",
                blocksize=self.blocksize,
                latency="low",
                callback=self._callback,
                extra_settings=extra,
            )
            self._stream.start()
            self._running = True
            logger.info(
                f"AudioPipe iniciado: sr={self.samplerate}, ch={self.channels}, "
                f"blocksize={self.blocksize}, device_index={self.device_index}"
            )
        except Exception as e:
            logger.error(f"Error iniciando OutputStream: {e}")
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

        # Asegurar 1-D
        pcm_mono_f32 = np.ravel(pcm_mono_f32).astype(np.float32, copy=False)

        with self._queue_lock:
            self._queue.append(pcm_mono_f32)

    # ---------- Callback de audio ----------
    def _callback(self, outdata, frames, time_info, status):
        """
        Callback de sounddevice. Debe escribir exactamente 'frames' muestras por canal.
        Si no hay datos, rellena con silencio. Aplica soft-clip 0.98.
        """
        if status:
            # Loguear xruns/underruns con rate limit
            now = time.time()
            if now - self._last_underflow_log > 2.0:  # 1 log / 2s
                logger.warning(f"Audio status: {status}")
                self._last_underflow_log = now

        needed = frames
        out = None

        with self._queue_lock:
            # Consumir de la cola hasta llenar 'frames'
            chunks = []
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

            if chunks:
                out = np.concatenate(chunks, dtype=np.float32)

        if out is None or len(out) < frames:
            # Rellenar silencio si faltan muestras
            if out is None:
                out = np.zeros(frames, dtype=np.float32)
            else:
                pad = np.zeros(frames - len(out), dtype=np.float32)
                out = np.concatenate([out, pad], dtype=np.float32)

        # Soft-clip a 0.98
        # Más suave que clip duro: y = tanh(gain*x); pero por simplicidad F1 usa clamp.
        np.clip(out, -0.98, 0.98, out=out)

        # Expandir a (frames, channels)
        if self.channels == 1:
            outdata[:, 0] = out
        else:
            # (No se usa en F1, pero dejamos duplicado simple)
            for ch in range(self.channels):
                outdata[:, ch] = out


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
            _instance = AudioPipe(samplerate=SR, channels=CH, blocksize=480, device_name_substr=ENV_AUDIO_NAME)
            logger.info("AudioPipe creado (lazy singleton).")
    return _instance
