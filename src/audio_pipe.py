"""
audio_pipe.py — Fase 1
Salida de audio PCM en tiempo real hacia "CABLE Input (VB-Audio Virtual Cable)"
usando sounddevice (WASAPI en Windows), 48 kHz mono, sin persistir en disco.

API expuesta:
- init_audio_out(sample_rate: int = 48000, channels: int = 1) -> None
- write_audio(samples: np.ndarray float32 mono) -> None
- close_audio_out() -> None

Diseño:
- Cola FIFO de chunks (float32 mono) protegida para hilos.
- Callback WASAPI vacía la cola, aplica limitador suave y rellena con silencio si está vacía.
- Bloques pequeños para baja latencia (p. ej., 256–480 frames).
"""

from __future__ import annotations
import os
import queue
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

# ====== Estado global ======
_QUEUE: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=64)  # buffers ~ cortos
_STREAM: Optional[sd.OutputStream] = None
_SR: int = 48000
_CH: int = 1
_DEVICE_NAME_HINT: str = os.getenv("AUDIO_DEVICE_NAME", "CABLE Input (VB-Audio Virtual Cable)")
_SOFT_CLIP_LIMIT = 0.98  # techo para anti-clipping
_BLOCKSIZE = 480         # ~10 ms a 48 kHz
_LATENCY = "low"         # hint para WASAPI

# ====== Utilidades ======
def _soft_clip(x: np.ndarray, limit: float = _SOFT_CLIP_LIMIT) -> np.ndarray:
    if x.size == 0:
        return x
    # limitador simple por pico
    m = np.max(np.abs(x))
    if m > limit and m > 0:
        x = (x / m) * limit
    return x.astype(np.float32, copy=False)

def _find_output_device(name_hint: str) -> Optional[int]:
    """
    Busca un dispositivo de salida que contenga 'name_hint' (case-insensitive).
    Devuelve el índice de sounddevice o None si no se encuentra.
    """
    try:
        devices = sd.query_devices()
    except Exception as e:
        print(f"[audio] No se pudo consultar dispositivos: {e}")
        return None

    hint_l = (name_hint or "").lower()
    best = None
    for idx, d in enumerate(devices):
        if d.get("max_output_channels", 0) <= 0:
            continue
        label = f"{d.get('name','')} ({d.get('hostapi','')})"
        if hint_l and hint_l in d.get("name", "").lower():
            best = idx
            break
        # fallback: si no se da hint, elegimos el primer output válido
        if not hint_l and best is None:
            best = idx
    return best

# ====== Callback de audio ======
def _audio_callback(outdata, frames, time_info, status):  # sd.OutputStream callback
    del time_info  # no usado
    if status:
        # underrun/overrun warnings
        print(f"[audio][cb] status: {status}")

    # Preparamos un buffer de salida (mono) y rellenamos desde la cola
    buf = np.zeros(frames, dtype=np.float32)
    filled = 0
    try:
        while filled < frames:
            try:
                # small chunk dequeue, no bloquear mucho
                chunk = _QUEUE.get_nowait()
            except queue.Empty:
                break
            if chunk.ndim != 1:
                chunk = chunk.reshape(-1)
            n = min(len(chunk), frames - filled)
            if n > 0:
                buf[filled:filled+n] = chunk[:n]
                filled += n
            # si el chunk era más largo, dejamos el resto para otro callback:
            remain = len(chunk) - n
            if remain > 0:
                _QUEUE.queue.appendleft(chunk[n:])  # reinyectar el sobrante al frente
                break
    except Exception as e:
        # En caso de error, silencio
        print(f"[audio][cb] error: {e}")

    # Anti-clipping suave
    buf = _soft_clip(buf)

    # Expandir a (frames, channels)
    if _CH == 1:
        outdata[:frames, 0] = buf
    else:
        # duplicar a estéreo si alguien lo configura así (no recomendado en Fase 1)
        outdata[:frames, :] = np.tile(buf.reshape(-1, 1), (1, _CH))


# ====== API pública ======
def init_audio_out(sample_rate: int = 48000, channels: int = 1) -> None:
    """Inicializa el stream de salida hacia el dispositivo VB-CABLE (o el que coincida con AUDIO_DEVICE_NAME)."""
    global _STREAM, _SR, _CH
    _SR = int(sample_rate)
    _CH = int(channels)

    if _STREAM is not None:
        return  # ya iniciado

    # Selección de dispositivo por nombre (hint) o default si no se encuentra
    dev_index = _find_output_device(_DEVICE_NAME_HINT)
    if dev_index is None:
        print(f"[audio] ⚠️ No se encontró dispositivo con hint '{_DEVICE_NAME_HINT}'. Usando default.")
    else:
        dev_info = sd.query_devices(dev_index)
        print(f"[audio] ✅ Usando dispositivo: {dev_info['name']} (idx={dev_index})")

    # Seleccionar WASAPI en Windows si está disponible
    try:
        host_api_wasapi = None
        for i, api in enumerate(sd.query_hostapis()):
            if "wasapi" in api.get("name", "").lower():
                host_api_wasapi = i
                break
        if host_api_wasapi is not None:
            sd.default.hostapi = host_api_wasapi
    except Exception:
        pass  # si falla, sounddevice elige el mejor disponible

    # Crear stream
    _STREAM = sd.OutputStream(
        device=dev_index if dev_index is not None else None,
        samplerate=_SR,
        channels=_CH,
        dtype="float32",
        callback=_audio_callback,
        blocksize=_BLOCKSIZE,
        latency=_LATENCY,   # hint, sounddevice puede ajustar
        finished_callback=None,
        dither_off=True,    # menos ruido
    )
    _STREAM.start()
    print(f"[audio] stream iniciado @ {_SR} Hz, ch={_CH}, blocksize={_BLOCKSIZE}, latency={_LATENCY}")

def write_audio(samples: np.ndarray) -> None:
    """
    Encola muestras float32 mono para reproducir en el callback.
    No bloquea si la cola está llena: descarta silenciosamente (anti-flood).
    """
    try:
        if samples is None:
            return
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return
        # Suavizado de picos extremo antes de encolar
        x = _soft_clip(x)
        _QUEUE.put_nowait(x)
    except queue.Full:
        # Si la cola está llena, dejamos caer el chunk para mantener latencia baja.
        pass
    except Exception as e:
        print(f"[audio] write_audio error: {e}")

def close_audio_out() -> None:
    """Detiene y cierra el stream de salida."""
    global _STREAM
    try:
        if _STREAM is not None:
            _STREAM.stop()
            _STREAM.close()
            _STREAM = None
            # Vaciar la cola
            while not _QUEUE.empty():
                try:
                    _QUEUE.get_nowait()
                except Exception:
                    break
            print("[audio] stream cerrado.")
    except Exception as e:
        print(f"[audio] close_audio_out error: {e}")
