from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

_gate: asyncio.Semaphore | None = None
_gate_limit: int | None = None


def reset_ai_concurrency_gate() -> None:
    """Test helper — clear the process-wide gate between cases."""
    global _gate, _gate_limit
    _gate = None
    _gate_limit = None


def _semaphore(limit: int) -> asyncio.Semaphore:
    global _gate, _gate_limit
    capped = max(1, limit)
    if _gate is None or _gate_limit != capped:
        _gate = asyncio.Semaphore(capped)
        _gate_limit = capped
    return _gate


@asynccontextmanager
async def ai_concurrency_slot(limit: int) -> AsyncIterator[None]:
    """Limit concurrent Vertex/OpenAI generate calls across the process."""
    sem = _semaphore(limit)
    await sem.acquire()
    try:
        yield
    finally:
        sem.release()
