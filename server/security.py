"""Security helpers: API-key authentication and basic rate limiting.

These are FastAPI dependencies you attach to protected routes. They are
deliberately simple and in-process; a real deployment would use a
gateway or a shared store (e.g. Redis) for keys and rate counters.
See Chapter 34.
"""
from __future__ import annotations
import os
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request

# Comma-separated list of valid keys from the environment.
_VALID_KEYS = set(
    k.strip() for k in os.environ.get("API_KEYS", "").split(",") if k.strip()
)

# Per-key sliding-window request timestamps.
_WINDOW_SECONDS = 60
_MAX_PER_WINDOW = int(os.environ.get("RATE_LIMIT_PER_MIN", "120"))
_HISTORY: dict[str, deque] = defaultdict(deque)


async def require_api_key(x_api_key: str = Header(default="")) -> str:
    """Reject requests without a valid X-API-Key header.

    If no keys are configured, auth is disabled (development mode).
    """
    if not _VALID_KEYS:
        return "anonymous"
    if x_api_key not in _VALID_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key


async def rate_limit(request: Request, x_api_key: str = Header(default="")) -> None:
    """Allow at most _MAX_PER_WINDOW requests per key per window."""
    key = x_api_key or (request.client.host if request.client else "unknown")
    now = time.time()
    hist = _HISTORY[key]
    while hist and now - hist[0] > _WINDOW_SECONDS:
        hist.popleft()                       # drop timestamps outside window
    if len(hist) >= _MAX_PER_WINDOW:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    hist.append(now)
