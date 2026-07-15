from __future__ import annotations

import logging
from typing import Any

NOISY_LOGGER_NAMES = (
    "httpcore",
    "httpx",
    "openai",
    "openai._base_client",
)


def configure_logging(*, verbose: bool) -> None:
    root = logging.getLogger()
    root.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if verbose else logging.INFO)

    logging.getLogger("dmv").setLevel(logging.DEBUG if verbose else logging.INFO)

    for logger_name in NOISY_LOGGER_NAMES:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def format_error(exc: BaseException) -> str:
    parts = [f"{type(exc).__name__}: {exc}"]

    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        parts.append(f"status={status_code}")

    for attr in ("code", "type", "param", "request_id"):
        value = getattr(exc, attr, None)
        if value:
            parts.append(f"{attr}={value}")

    body = _extract_error_body(exc)
    if body:
        parts.append(f"body={body}")

    cause = exc.__cause__
    if cause is not None and cause is not exc:
        parts.append(f"cause={format_error(cause)}")

    return " | ".join(parts)


def _extract_error_body(exc: BaseException) -> str | None:
    body: Any = getattr(exc, "body", None)
    if body is None:
        return None

    if isinstance(body, dict):
        message = body.get("message") or body.get("error", {}).get("message")
        if message:
            return str(message)
        return str(body)

    text = str(body).strip()
    return text or None
