"""File-based persistence for saved snapshots (frame + world model).

A snapshot is a JSON file holding the serialised world model, plus an
optional JPEG of the frame it was computed from, named by a sortable
id. Small, dependency-free, and enough for review and training data.
See Chapter 9.
"""
from __future__ import annotations
import json
import os
import time
import uuid
from typing import Any

STORE_DIR = os.path.join(os.path.dirname(__file__), "store")
os.makedirs(STORE_DIR, exist_ok=True)


def _new_id() -> str:
    """Timestamp-prefixed id so files sort chronologically."""
    return f"{int(time.time())}-{uuid.uuid4().hex[:6]}"


def save_snapshot(world: dict[str, Any],
                  image_bytes: bytes | None = None) -> str:
    sid = _new_id()
    with open(os.path.join(STORE_DIR, sid + ".json"), "w") as f:
        json.dump(world, f)
    if image_bytes is not None:
        with open(os.path.join(STORE_DIR, sid + ".jpg"), "wb") as f:
            f.write(image_bytes)
    return sid


def list_snapshots() -> list[str]:
    ids = [f[:-5] for f in os.listdir(STORE_DIR) if f.endswith(".json")]
    return sorted(ids)


def load_snapshot(sid: str) -> dict[str, Any]:
    with open(os.path.join(STORE_DIR, sid + ".json")) as f:
        return json.load(f)
