"""
llm_realtime.py — Fase 1 (con fallback)
1) Intenta OpenAI Realtime por WebSocket (tokens parciales).
2) Si en 3 s no llegan deltas, hace fallback a Responses HTTP streaming (SDK).

Yields de texto "cortados" por frases para iniciar TTS cuanto antes.
"""

import asyncio
import json
import re
from typing import AsyncGenerator, Optional

import websockets
from openai import OpenAI

REALTIME_URL = "wss://api.openai.com/v1/realtime"

# Split por fin de oración, paréntesis, o saltos
_SENTENCE_SPLIT_RE = re.compile(r"([\.!\?]+[\)\]]?\s+|[\n\r]+)")

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

async def _http_stream_fallback(api_key: str, model: str, prompt: str) -> AsyncGenerator[str, None]:
    """
    Fallback con Responses HTTP streaming (SDK openai >=1.3).
    Emite deltas de texto y hace el mismo sentence-chunking.
    """
    client = OpenAI(api_key=api_key)
    # stream de eventos
    with client.responses.stream(
        model=model,
        input=[{"role": "user", "content": prompt}],
    ) as stream:
        text_buffer = ""
        for event in stream:
            et = getattr(event, "type", "")
            if et == "response.output_text.delta":
                delta = event.delta or ""
                text_buffer += delta
                parts, text_buffer = _yieldable_chunks_from_buffer(text_buffer, force=False)
                for p in parts:
                    yield p
            elif et in ("response.completed", "done"):
                if text_buffer.strip():
                    parts, text_buffer = _yieldable_chunks_from_buffer(text_buffer, force=True)
                    for p in parts:
                        yield p
                break
            elif et in ("response.error", "error"):
                # Levantar para que el caller informe 500
                raise RuntimeError(f"HTTP stream error: {getattr(event, 'error', event)}")

async def stream_completion(
    api_key: str,
    model: str,
    prompt: str,
    *,
    connect_timeout: float = 10.0,
    receive_timeout: float = 120.0,
    first_delta_deadline: float = 3.0,  # si no hay delta en 3s -> fallback
) -> AsyncGenerator[str, None]:
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY vacío")
    if not model:
        raise RuntimeError("REALTIME_MODEL vacío")

    url = f"{REALTIME_URL}?model={model}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": "realtime=v1",
    }

    # Intento WS
    try:
        async with websockets.connect(
            url,
            extra_headers=headers,
            # subprotocolo requerido por algunos gateways
            subprotocols=["realtime"],
            open_timeout=connect_timeout,
            ping_interval=20,
            ping_timeout=20,
            max_size=8 * 1024 * 1024,
        ) as ws:
            # session.update (sólo texto)
            await ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "modalities": ["text"],
                    "instructions": "You are a streaming text assistant."
                },
            }))

            # response.create con nuestro prompt
            await ws.send(json.dumps({
                "type": "response.create",
                "response": {
                    "modalities": ["text"],
                    "instructions": prompt
                }
            }))

            text_buffer = ""
            got_first_delta = False
            # Para detectar ausencia de deltas en N segundos
            first_delta_timer = asyncio.get_event_loop().time() + first_delta_deadline

            ws_rcv_task = asyncio.create_task(_ws_recv_json(ws))
            while True:
                # Si no llegaron deltas a tiempo, forzar fallback
                if not got_first_delta and asyncio.get_event_loop().time() > first_delta_timer:
                    raise TimeoutError("No WS delta in deadline -> fallback")

                try:
                    event = await asyncio.wait_for(ws_rcv_task, timeout=receive_timeout)
                except asyncio.TimeoutError:
                    raise RuntimeError("Timeout recibiendo eventos Realtime")

                ws_rcv_task = asyncio.create_task(_ws_recv_json(ws))

                if not event:
                    # Cerrar limpiamente: drenar
                    if text_buffer.strip():
                        parts, text_buffer = _yieldable_chunks_from_buffer(text_buffer, force=True)
                        for p in parts:
                            yield p
                    break

                etype = event.get("type", "")
                if etype == "response.output_text.delta":
                    delta = event.get("delta", "") or ""
                    text_buffer += delta
                    got_first_delta = True
                    parts, text_buffer = _yieldable_chunks_from_buffer(text_buffer, force=False)
                    for p in parts:
                        yield p

                elif etype == "response.completed":
                    if text_buffer.strip():
                        parts, text_buffer = _yieldable_chunks_from_buffer(text_buffer, force=True)
                        for p in parts:
                            yield p
                    break

                elif etype in ("error", "response.error"):
                    raise RuntimeError(f"Realtime error: {event.get('error', event)}")

        return  # éxito por WS

    except Exception as ws_err:
        # Fallback HTTP streaming
        # Nota: si el error es de permisos/modelo, aquí también fallará y lo propagamos.
        async for chunk in _http_stream_fallback(api_key=api_key, model=model, prompt=prompt):
            yield chunk
