"""
envload.py — load .env into os.environ before anything else reads config.

Zero-dependency (no python-dotenv needed). Import this FIRST in main.py so modules that read
env at import time (auth, ai_juice) see the values. Existing env vars win over the file, so a
real deployment's injected secrets are never overridden by a stray local .env.
"""
from __future__ import annotations

import os
from pathlib import Path


def load(path: str | None = None) -> None:
    p = Path(path) if path else Path(__file__).parent / ".env"
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


load()
