"""
llm_realtime.py — Fase 1
WS Realtime preferido + fallback HTTP streaming puro (sin SDK) con httpx.

- Primero intenta Realtime (WebSocket). Si no llega el primer delta en N s, cae a fallback.
- Fallback: llama /v1/chat/completions con stream=true y parsea SSE "data:".
- Emite trozos por oración para disparar TTS cuanto antes.
"""

from __future__ import annotations
import asyncio
import json
import os
import re
from typing import AsyncGenerator, Optional

import httpx
import websockets

REALTIME_URL = "wss://api.openai.com/v1/realtime"
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gpt-4o-mini")

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


# =================== Fallback HTTP (SSE) con httpx (async) ===================

async def _http_chat_stream_fallback_async(api_key: str, prompt: str) -> AsyncGenerator[str, None]:
    """
    Streaming directo contra /v1/chat/completions (OpenAI) usando httpx.
    Parseo de Server-Sent Events (líneas que empiezan con 'data: ').
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": FALLBACK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }

    timeout = httpx.Timeout(connect=10.0, read=90.0, write=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as r:
            r.raise_for_status()
            text_buffer = ""
            async for raw_line in r.aiter_lines():
                if not raw_line:
                    continue
                if raw_line.startswith("data: "):
                    data = raw_line[6:].strip()
                    if data == "[DONE]":
                        # flush final
                        if text_buffer.strip():
                            parts, text_buffer = _yieldable_chunks_from_buffer(text_buffer, force=True)
                            for p in parts:
                                yield p
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    # Formato chat.completions stream
                    try:
                        delta = obj["choices"][0]["delta"].get("content")
                    except Exception:
                        delta = None
                    if delta:
                        text_buffer += delta
                        parts, text_buffer = _yieldable_chunks_from_buffer(text_buffer, force=False)
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
            subprotocols=["realtime"],      # importante
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
        # ====== Fallback HTTP puro (no SDK) ======
        async for chunk in _http_chat_stream_fallback_async(api_key=api_key, prompt=prompt):
            yield chunk
