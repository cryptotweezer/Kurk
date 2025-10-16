"""
llm_realtime.py — Fase 1 (WS Realtime + fallback no bloqueante)

- Intenta OpenAI Realtime por WebSocket (tokens parciales).
- Si en N segundos no llega el primer delta o falla el WS, hace fallback a
  Chat Completions streaming en un hilo (no bloquea el event loop).
- Emite chunks por oración para disparar TTS cuanto antes.
"""

from __future__ import annotations
import asyncio
import json
import os
import queue
import re
import threading
from typing import AsyncGenerator, Optional

import websockets
from openai import OpenAI

REALTIME_URL = "wss://api.openai.com/v1/realtime"
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gpt-4o-mini")  # modelo para chat.completions

# Detectar fin de oración o salto de línea
_SENTENCE_SPLIT_RE = re.compile(r"([\.!\?]+[\)\]]?\s+|[\n\r]+)")

def _yieldable_chunks_from_buffer(buf: str, force: bool = False):
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


# ===================== FALLBACK (chat.completions stream) =====================

def _chat_stream_thread(api_key: str, model: str, prompt: str, q: "queue.Queue[Optional[str]]"):
    """
    Hilo bloqueante que consume el stream de chat.completions y empuja deltas a la cola.
    Al finalizar, encola None como centinela.
    """
    try:
        client = OpenAI(api_key=api_key)
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for event in stream:
            try:
                choice = event.choices[0]
                delta = getattr(choice.delta, "content", None)
                if delta:
                    q.put(delta, timeout=1)
            except Exception:
                # Ignorar eventos no textuales
                pass
        q.put(None, timeout=1)
    except Exception as e:
        # Propagamos error en texto para que el caller levante 500
        q.put(f"__ERROR__:{repr(e)}", timeout=1)
        q.put(None, timeout=1)

async def _chat_stream_fallback_async(api_key: str, prompt: str) -> AsyncGenerator[str, None]:
    """
    Wrapper asíncrono que ejecuta el hilo de completions stream y va sacando deltas sin
    bloquear el event loop. Hace sentence-chunking antes de yield.
    """
    q: "queue.Queue[Optional[str]]" = queue.Queue(maxsize=1024)
    th = threading.Thread(target=_chat_stream_thread, args=(api_key, FALLBACK_MODEL, prompt, q), daemon=True)
    th.start()

    buffer = ""
    while True:
        # Espera cooperativa al siguiente item de la cola
        item = await asyncio.to_thread(q.get)
        if item is None:
            # flush final
            if buffer.strip():
                parts, buffer = _yieldable_chunks_from_buffer(buffer, force=True)
                for p in parts:
                    yield p
            break
        if item.startswith("__ERROR__:"):
            raise RuntimeError(f"HTTP fallback error: {item}")
        buffer += item
        parts, buffer = _yieldable_chunks_from_buffer(buffer, force=False)
        for p in parts:
            yield p


# ========================= WebSocket Realtime (preferido) =========================

async def _ws_recv_json(ws) -> Optional[dict]:
    try:
        msg = await ws.recv()
        if not msg:
            return None
        if isinstance(msg, (bytes, bytearray)):
            return None
        return json.loads(msg)
    except websockets.ConnectionClosed:
        return None

async def stream_completion(
    api_key: str,
    model: str,
    prompt: str,
    *,
    connect_timeout: float = 10.0,
    receive_timeout: float = 120.0,
    first_delta_deadline: float = 3.0,  # si no hay delta en N s => fallback
) -> AsyncGenerator[str, None]:
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY vacío")
    if not model:
        raise RuntimeError("REALTIME_MODEL vacío")

    # ====== Intento por WebSocket Realtime ======
    url = f"{REALTIME_URL}?model={model}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": "realtime=v1",
    }

    try:
        async with websockets.connect(
            url,
            extra_headers=headers,
            subprotocols=["realtime"],      # importante para algunos gateways
            open_timeout=connect_timeout,
            ping_interval=20,
            ping_timeout=20,
            max_size=8 * 1024 * 1024,
        ) as ws:
            # Configurar sesión sólo-texto
            await ws.send(json.dumps({
                "type": "session.update",
                "session": {"modalities": ["text"], "instructions": "You are a streaming text assistant."},
            }))
            # Solicitar respuesta con nuestro prompt
            await ws.send(json.dumps({
                "type": "response.create",
                "response": {"modalities": ["text"], "instructions": prompt},
            }))

            buf = ""
            got_first = False
            first_deadline = asyncio.get_event_loop().time() + first_delta_deadline

            ws_task = asyncio.create_task(_ws_recv_json(ws))
            while True:
                # Si no llegó ningún delta a tiempo, cambiamos a fallback
                if not got_first and asyncio.get_event_loop().time() > first_deadline:
                    raise TimeoutError("No WS delta in deadline -> fallback")

                try:
                    event = await asyncio.wait_for(ws_task, timeout=receive_timeout)
                except asyncio.TimeoutError:
                    raise RuntimeError("Timeout recibiendo eventos Realtime")

                ws_task = asyncio.create_task(_ws_recv_json(ws))
                if not event:
                    if buf.strip():
                        parts, buf = _yieldable_chunks_from_buffer(buf, force=True)
                        for p in parts:
                            yield p
                    break

                et = event.get("type", "")
                if et == "response.output_text.delta":
                    delta = event.get("delta", "") or ""
                    buf += delta
                    got_first = True
                    parts, buf = _yieldable_chunks_from_buffer(buf, force=False)
                    for p in parts:
                        yield p
                elif et == "response.completed":
                    if buf.strip():
                        parts, buf = _yieldable_chunks_from_buffer(buf, force=True)
                        for p in parts:
                            yield p
                    break
                elif et in ("error", "response.error"):
                    raise RuntimeError(f"Realtime error: {event.get('error', event)}")

        return  # éxito por WS

    except Exception:
        # ====== Fallback no bloqueante (chat.completions stream en hilo) ======
        async for chunk in _chat_stream_fallback_async(api_key=api_key, prompt=prompt):
            yield chunk
