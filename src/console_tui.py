"""
console_tui.py — Fase 1
TUI mínima para enviar preguntas al backend (/say) sin flood y con logs limitados.

Características:
- Input siempre enfocado (loop síncrono con input()).
- Anti-flood: 1 solicitud activa a la vez (cola vacía; descarta si ocupada).
- Comandos: /quiet (toggle logs), /clear (limpia pantalla), /exit (salir),
            /f1..../f4 reservados para futuros modos (placeholders).
- Logs rate-limited (últimos N eventos en memoria, sin bloquear el input).
- No persiste nada en disco.

Uso:
(.venv) PS C:\\AI_Workspace\\Kurk> python -m src.console_tui
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
    # Mostrar solo la última línea para no romper el input
    print(line)

async def post_say(text: str) -> Optional[dict]:
    """Envía texto a /say y devuelve el JSON de métricas o None si falla."""
    timeout = httpx.Timeout(20.0, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(SAY_ENDPOINT, json={"text": text})
        r.raise_for_status()
        return r.json()

def handle_command(cmd: str) -> bool:
    """Procesa comandos locales. Devuelve True si debe continuar el loop."""
    global quiet_mode
    c = cmd.strip().lower()

    if c in ("/exit", "/quit"):
        print("Bye.")
        return False
    if c in ("/clear", "/cls"):
        # Limpia pantalla de forma simple
        os.system("cls" if os.name == "nt" else "clear")
        return True
    if c == "/quiet":
        quiet_mode = not quiet_mode
        print(f"(quiet_mode = {quiet_mode})")
        return True
    if c in ("/f1", "/f2", "/f3", "/f4"):
        # Placeholders para futuros modos (Fase 3)
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
            # Input síncrono para mantener foco y copy/paste simple
            user = await loop.run_in_executor(None, lambda: input("\n> ").strip())
            if not user:
                continue

            if user.startswith("/"):
                if not handle_command(user):
                    break
                else:
                    continue

            if busy:
                # Anti-flood: descarta si aún estamos procesando
                print("(busy) Procesando anterior… intenta de nuevo en unos segundos.")
                continue

            busy = True
            t0 = time.perf_counter()
            print("(sending) …")

            try:
                data = await post_say(user)
                t1 = time.perf_counter()
                rtt_ms = round((t1 - t0) * 1000.0, 1)

                # Métricas desde backend
                latency = data.get("latency_ms")
                first_audio = data.get("first_audio_ms")
                tokens = data.get("tokens")

                log(f"OK tokens={tokens} first_audio_ms={first_audio} total_ms={latency} (rtt={rtt_ms})")
            except httpx.HTTPStatusError as he:
                log(f"HTTP {he.response.status_code} — {he.response.text[:200]}")
            except Exception as e:
                log(f"ERROR {type(e).__name__}: {e}")
            finally:
                busy = False

        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        except Exception as e:
            log(f"Loop error: {e}")

if __name__ == "__main__":
    # Ejecuta el loop asyncio
    try:
        asyncio.run(main())
    except RuntimeError as e:
        # En Windows, si ya hay un loop, fallback
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
