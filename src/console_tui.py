"""
console_tui.py — Fase 1
TUI mínima para enviar preguntas al backend (/say) sin flood y con logs limitados.

Cambios:
- Timeout ampliado (connect=10s, read/write=90s) para evitar ReadTimeout prematuro.
- Logs con duración y cuerpo de error más claro.
"""

from __future__ import annotations
import asyncio
import os
import sys
import time
from collections import deque
from typing import Deque, Optional

import httpx

API_URL = os.getenv("KURK_API_URL", "http://127.0.0.1:8000")
SAY_ENDPOINT = f"{API_URL}/say"
LOG_MAX = 50

quiet_mode = False
busy = False
logs: Deque[str] = deque(maxlen=LOG_MAX)

def log(msg: str) -> None:
    if quiet_mode:
        return
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    logs.append(line)
    print(line)

async def post_say(text: str) -> Optional[dict]:
    """Envía texto a /say y devuelve el JSON de métricas o None si falla."""
    # Timeouts más generosos para streams largos
    timeout = httpx.Timeout(connect=10.0, read=90.0, write=90.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(SAY_ENDPOINT, json={"text": text})
        r.raise_for_status()
        return r.json()

def handle_command(cmd: str) -> bool:
    global quiet_mode
    c = cmd.strip().lower()

    if c in ("/exit", "/quit"):
        print("Bye.")
        return False
    if c in ("/clear", "/cls"):
        os.system("cls" if os.name == "nt" else "clear")
        return True
    if c == "/quiet":
        quiet_mode = not quiet_mode
        print(f"(quiet_mode = {quiet_mode})")
        return True
    if c in ("/f1", "/f2", "/f3", "/f4"):
        print(f"(placeholder) {c} reservado para modos futuros.")
        return True

    print("Comandos disponibles: /quiet, /clear, /exit, /f1..../f4 (placeholder)")
    return True

def banner() -> None:
    print("─" * 64)
    print(" KURK · Phase 1 · TUI (text → /say → TTS → VB-CABLE → OBS) ")
    print(" Commands: /quiet  /clear  /exit   (/f1..../f4 placeholders)")
    print("─" * 64)

async def main() -> None:
    global busy
    banner()
    print(f"API: {SAY_ENDPOINT}")

    loop = asyncio.get_event_loop()
    while True:
        try:
            user = await loop.run_in_executor(None, lambda: input("\n> ").strip())
            if not user:
                continue

            if user.startswith("/"):
                if not handle_command(user):
                    break
                else:
                    continue

            if busy:
                print("(busy) Procesando anterior… intenta de nuevo en unos segundos.")
                continue

            busy = True
            t0 = time.perf_counter()
            print("(sending) …")

            try:
                data = await post_say(user)
                t1 = time.perf_counter()
                rtt_ms = round((t1 - t0) * 1000.0, 1)

                latency = data.get("latency_ms")
                first_audio = data.get("first_audio_ms")
                tokens = data.get("tokens")

                log(f"OK tokens={tokens} first_audio_ms={first_audio} total_ms={latency} (tui_rtt_ms={rtt_ms})")
            except httpx.ReadTimeout:
                t1 = time.perf_counter()
                rtt_ms = round((t1 - t0) * 1000.0, 1)
                log(f"ERROR ReadTimeout (tui waited {rtt_ms} ms). Considera que /say termina al final del stream.")
            except httpx.HTTPStatusError as he:
                body = he.response.text
                t1 = time.perf_counter()
                rtt_ms = round((t1 - t0) * 1000.0, 1)
                log(f"HTTP {he.response.status_code} — {body[:400]} (tui_rtt_ms={rtt_ms})")
            except Exception as e:
                t1 = time.perf_counter()
                rtt_ms = round((t1 - t0) * 1000.0, 1)
                log(f"ERROR {type(e).__name__}: {e} (tui_rtt_ms={rtt_ms})")
            finally:
                busy = False

        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        except Exception as e:
            log(f"Loop error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
