"""
llm_realtime.py — Fase 1
Cliente WebSocket para OpenAI Realtime (texto→tokens parciales).

Expone:
- async def stream_completion(api_key: str, model: str, prompt: str)
  Genera strings (trozos de texto) tan pronto como llegan del modelo.
  Hace un pequeño "sentence chunking" para iniciar TTS con la primera oración.

Notas:
- Usa WebSocket Realtime: wss://api.openai.com/v1/realtime?model=<MODEL>
- Cabeceras: Authorization: Bearer <key>, OpenAI-Beta: realtime=v1
- Eventos esperados: response.output_text.delta (texto incremental),
                     response.completed / response.error / error

Fase 1: centrado en texto. Audio se maneja aguas abajo (Coqui).
"""

import asyncio
import json
import os
import re
from typing import AsyncGenerator, Optional

import websockets

REALTIME_URL = "wss://api.openai.com/v1/realtime"

# Separadores para "soltar" chunks útiles al TTS sin esperar el párrafo completo
_SENTENCE_SPLIT_RE = re.compile(r"([\.!\?]+[\)\]]?\s+|[\n\r]+)")

async def _ws_recv_json(ws) -> Optional[dict]:
    """Recibe un frame del WebSocket y lo decodifica como JSON (o None si ping/pong)."""
    try:
        msg = await ws.recv()
        if not msg:
            return None
        if isinstance(msg, (bytes, bytearray)):
            # La API puede enviar binarios para audio. En Fase 1 sólo esperamos texto.
            return None
        return json.loads(msg)
    except websockets.ConnectionClosedOK:
        return None
    except websockets.ConnectionClosedError:
        return None

def _yieldable_chunks_from_buffer(buf: str, force: bool = False):
    """
    Corta el buffer por separadores "suaves" (fin de oración, salto de línea).
    Si force=True, drena lo que quede.
    """
    parts = []
    last = 0
    for m in _SENTENCE_SPLIT_RE.finditer(buf):
        end = m.end()
        seg = buf[last:end].strip()
        if seg:
            parts.append(seg)
        last = end
    if force:
        tail = buf[last:].strip()
        if tail:
            parts.append(tail)
        last = len(buf)
    return parts, buf[last:]


async def stream_completion(
    api_key: str,
    model: str,
    prompt: str,
    *,
    request_id: Optional[str] = None,
    connect_timeout: float = 10.0,
    receive_timeout: float = 120.0,
) -> AsyncGenerator[str, None]:
    """
    Conecta al endpoint Realtime por WebSocket, envía el prompt,
    y rinde texto incremental usable de inmediato por el TTS.

    Yields: str (trozos ya "cortados" por oraciones/pausas)
    """
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY vacío")
    if not model:
        raise RuntimeError("REALTIME_MODEL vacío")

    url = f"{REALTIME_URL}?model={model}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": "realtime=v1",
    }

    # Backoff simple en caso de primer fallo de conexión
    backoff_s = 1.0
    while True:
        try:
            async with websockets.connect(
                url,
                extra_headers=headers,
                open_timeout=connect_timeout,
                ping_interval=20,
                ping_timeout=20,
                max_size=8 * 1024 * 1024,  # tolerante
            ) as ws:
                # 1) (Opcional) actualizar sesión: sólo texto
                await ws.send(json.dumps({
                    "type": "session.update",
                    "session": {
                        "modalities": ["text"],
                        "instructions": "You are a streaming text assistant.",
                    },
                }))

                # 2) Crear respuesta con nuestro prompt (estilo Realtime)
                await ws.send(json.dumps({
                    "type": "response.create",
                    "response": {
                        "modalities": ["text"],
                        "instructions": prompt,
                        # Podrías afinar parámetros en Fase 2+/3:
                        # "temperature": 0.6, "max_output_tokens": 256, ...
                    }
                }))

                # 3) Recibir eventos y extraer deltas de texto
                text_buffer = ""
                ws_rcv_task = asyncio.create_task(_ws_recv_json(ws))
                while True:
                    try:
                        event = await asyncio.wait_for(ws_rcv_task, timeout=receive_timeout)
                    except asyncio.TimeoutError:
                        raise RuntimeError("Timeout recibiendo eventos Realtime")

                    # Preparar siguiente lectura temprana
                    ws_rcv_task = asyncio.create_task(_ws_recv_json(ws))

                    if not event:
                        # Conexión cerrada o frame vacío
                        # Drenar lo que quede en buffer
                        if text_buffer.strip():
                            parts, text_buffer = _yieldable_chunks_from_buffer(text_buffer, force=True)
                            for p in parts:
                                yield p
                        break

                    etype = event.get("type", "")
                    # Deltas de texto llegan como "response.output_text.delta"
                    if etype == "response.output_text.delta":
                        delta = event.get("delta", "")
                        if not isinstance(delta, str):
                            continue
                        text_buffer += delta

                        # Si ya tenemos una oración/coma/línea, soltamos chunks
                        parts, text_buffer = _yieldable_chunks_from_buffer(text_buffer, force=False)
                        for p in parts:
                            yield p

                    elif etype == "response.completed":
                        # Drenar cualquier residuo de buffer
                        if text_buffer.strip():
                            parts, text_buffer = _yieldable_chunks_from_buffer(text_buffer, force=True)
                            for p in parts:
                                yield p
                        break

                    elif etype in ("error", "response.error"):
                        # Error reportado por el servidor
                        err = event.get("error", event)
                        raise RuntimeError(f"Realtime error: {err}")

                    # Otros eventos (ping/pong/metadata) se ignoran en Fase 1

                # Salimos del with (cierre limpio)
                return

        except Exception as e:
            # Reintento simple (1 vez) — Fase 1
            if backoff_s > 1.5:
                raise
            await asyncio.sleep(backoff_s)
            backoff_s *= 2.0
            continue
