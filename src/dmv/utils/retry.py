from __future__ import annotations

import asyncio
import json
import logging
import random
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from dmv.logging_config import format_error

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _is_transient_disconnect(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in {
        "RemoteProtocolError",
        "ConnectError",
        "ReadTimeout",
        "WriteTimeout",
        "ConnectTimeout",
        "PoolTimeout",
    }:
        return True
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "server disconnected",
            "connection reset",
            "temporarily unavailable",
            "unavailable",
            "timed out",
            "timeout",
        )
    )


async def retry_async(
    operation: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    base_delay_seconds: float,
    operation_name: str,
) -> T:
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return await operation()
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                logger.error(
                    "%s failed after %s attempt(s): %s",
                    operation_name,
                    max_retries + 1,
                    format_error(exc),
                )
                break
            # Connection drops benefit from longer backoff + jitter.
            multiplier = 3 if _is_transient_disconnect(exc) else 1
            delay = base_delay_seconds * (2**attempt) * multiplier
            delay += random.uniform(0.0, min(delay * 0.25, 5.0))
            logger.warning(
                "%s failed on attempt %s/%s: %s. Retrying in %.1fs.",
                operation_name,
                attempt + 1,
                max_retries + 1,
                format_error(exc),
                delay,
            )
            await asyncio.sleep(delay)

    assert last_error is not None
    raise last_error


def parse_json_response(output_text: str) -> dict[str, Any]:
    text = output_text.strip()
    if not text:
        raise ValueError("AI response was empty")

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"AI response was not valid JSON: {exc}") from exc
