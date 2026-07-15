from __future__ import annotations

from typing import Any

from dmv.models.usage import TokenUsage


def parse_openai_usage(usage: Any, *, model: str) -> TokenUsage:
    if usage is None:
        return TokenUsage.empty(model=model)

    cached_input_tokens = 0
    details = getattr(usage, "input_tokens_details", None)
    if details is not None:
        cached_input_tokens = int(getattr(details, "cached_tokens", 0) or 0)

    return TokenUsage(
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
        cached_input_tokens=cached_input_tokens,
        model=model,
    )
