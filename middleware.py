"""
middleware.py — backend hardening (spec Vol III Ch 60-62).

Three small, non-breaking pieces:
  • RequestContextMiddleware — assigns each request an X-Request-ID and logs
    method / path / status / duration (Ch 62 logging).
  • RateLimitMiddleware — optional per-IP token bucket over /api/* (Ch 61). OFF by default
    (RATE_LIMIT_RPM=0) so it never surprises the local board; set the env to enable.
  • install_error_handler — wraps UNHANDLED exceptions (500s only) in the spec's error
    envelope {error:{code,message,request_id}} (Ch 60). HTTPExceptions keep FastAPI's shape
    so existing 401/404 clients are unaffected.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from collections import defaultdict, deque

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

log = logging.getLogger("juiced.request")

RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "0"))  # 0 = disabled


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = rid
        start = time.perf_counter()
        response = await call_next(request)
        dur_ms = (time.perf_counter() - start) * 1000.0
        response.headers["X-Request-ID"] = rid
        # Keep the hot /api/lines poll out of the log at INFO to avoid noise; log the rest.
        if request.url.path not in ("/api/lines", "/api/status"):
            log.info("%s %s -> %s %.0fms rid=%s",
                     request.method, request.url.path, response.status_code, dur_ms, rid)
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window per-IP limiter over /api/* paths. In-memory (single process); for a
    multi-instance deploy this would move to Redis, but it's a real guard for one node."""

    def __init__(self, app, rpm: int):
        super().__init__(app)
        self.rpm = rpm
        self._hits: dict[str, deque] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        if self.rpm <= 0 or not request.url.path.startswith("/api/"):
            return await call_next(request)
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        q = self._hits[ip]
        cutoff = now - 60.0
        while q and q[0] < cutoff:
            q.popleft()
        if len(q) >= self.rpm:
            rid = getattr(request.state, "request_id", None)
            return JSONResponse(
                status_code=429,
                content={"error": {"code": "RATE_LIMITED",
                                   "message": f"Rate limit of {self.rpm}/min exceeded.",
                                   "request_id": rid}},
                headers={"Retry-After": "60"})
        q.append(now)
        return await call_next(request)


def install_error_handler(app: FastAPI) -> None:
    """Format unhandled exceptions as the spec's error envelope. Only catches genuine 500s —
    HTTPException (401/404/…) is left to FastAPI so existing response bodies don't change."""
    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):  # noqa: ANN001
        rid = getattr(request.state, "request_id", None)
        log.exception("Unhandled error rid=%s on %s: %s", rid, request.url.path, exc)
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "SERVER_ERROR",
                               "message": "An internal error occurred.",
                               "request_id": rid}})


def install(app: FastAPI) -> None:
    """Wire all three. Order matters: request-context outermost so the rate limiter and
    handlers can read request_id."""
    if RATE_LIMIT_RPM > 0:
        app.add_middleware(RateLimitMiddleware, rpm=RATE_LIMIT_RPM)
    app.add_middleware(RequestContextMiddleware)
    install_error_handler(app)
