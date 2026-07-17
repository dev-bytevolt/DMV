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


def parse_vertex_usage(usage: Any, *, model: str) -> TokenUsage:
    if usage is None:
        return TokenUsage.empty(model=model)

    input_tokens = int(
        getattr(usage, "prompt_token_count", None)
        or getattr(usage, "input_tokens", 0)
        or 0
    )
    output_tokens = int(
        getattr(usage, "candidates_token_count", None)
        or getattr(usage, "output_tokens", 0)
        or 0
    )
    total_tokens = int(
        getattr(usage, "total_token_count", None)
        or getattr(usage, "total_tokens", 0)
        or (input_tokens + output_tokens)
    )
    cached_input_tokens = int(
        getattr(usage, "cached_content_token_count", None)
        or getattr(usage, "cached_input_tokens", 0)
        or 0
    )
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached_input_tokens,
        model=model,
    )
