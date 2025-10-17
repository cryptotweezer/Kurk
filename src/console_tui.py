# -*- coding: utf-8 -*-
"""
console_tui.py
TUI anti-flood para enviar texto a POST /say y mostrar métricas.
- Una solicitud activa a la vez.
- Comandos: /quiet (toggle logs), /clear, /exit
- Read timeout ampliado (120 s)

Uso (PowerShell, con venv activo):
  python -m src.console_tui
"""

from __future__ import annotations

import os
import sys
import time
import json
import threading
from typing import Optional

import httpx

API_URL = os.getenv("KURK_API_URL", "http://127.0.0.1:8000/say")

QUIET = False
IN_FLIGHT_LOCK = threading.Lock()
IN_FLIGHT = False


def _print_info(msg: str):
    if not QUIET:
        print(msg, flush=True)


def _pretty_metrics(d: dict) -> str:
    def fmt(k):
        v = d.get(k)
        if v is None:
            return f"{k}=—"
        if isinstance(v, float):
            return f"{k}={v:.1f}"
        return f"{k}={v}"
    keys = ["llm_to_first_delta_ms", "first_audio_ms", "total_ms", "chunk_count"]
    return " | ".join(fmt(k) for k in keys)


def _post_text(text: str, timeout_sec: float = 120.0) -> Optional[dict]:
    global IN_FLIGHT
    payload = {"text": text}
    headers = {"Content-Type": "application/json"}

    with httpx.Client(timeout=httpx.Timeout(timeout_sec)) as client:
        try:
            r = client.post(API_URL, headers=headers, content=json.dumps(payload))
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            _print_info(f"[HTTP {e.response.status_code}] {e.response.text}")
        except httpx.TimeoutException:
            _print_info("[timeout] request exceeded read timeout")
        except Exception as e:
            _print_info(f"[error] {e}")
        finally:
            IN_FLIGHT = False
    return None


def main():
    global QUIET, IN_FLIGHT

    print("KURK F1 — Console TUI")
    print("Type your prompt and press Enter.")
    print("Commands: /quiet  /clear  /exit")
    print(f"Target endpoint: {API_URL}")
    print("-" * 60)

    while True:
        try:
            user = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            break

        if not user:
            continue

        # Comandos
        if user.lower() in ("/exit", "/quit"):
            print("bye.")
            break
        if user.lower() == "/clear":
            os.system("cls" if os.name == "nt" else "clear")
            continue
        if user.lower() == "/quiet":
            QUIET = not QUIET
            print(f"quiet={'on' if QUIET else 'off'}")
            continue

        # Anti-flood: 1 solicitud activa
        if IN_FLIGHT_LOCK.locked() or IN_FLIGHT:
            print("A request is already in flight. Please wait...", flush=True)
            continue

        # Lanzar request en thread para no bloquear la UI
        def worker(text: str):
            global IN_FLIGHT
            t0 = time.perf_counter()
            _print_info("[sending] …")
            try:
                with IN_FLIGHT_LOCK:
                    metrics = _post_text(text)
            finally:
                IN_FLIGHT = False
            t1 = time.perf_counter()

            if metrics is not None:
                print(f"[ok] {_pretty_metrics(metrics)} | tui_ms={(t1 - t0)*1000:.1f}", flush=True)
            else:
                print("[fail] (see logs above)", flush=True)

        IN_FLIGHT = True
        th = threading.Thread(target=worker, args=(user,), daemon=True)
        th.start()


if __name__ == "__main__":
    main()
