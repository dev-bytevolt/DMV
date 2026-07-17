from __future__ import annotations

from dataclasses import dataclass

from dmv.config import Settings
from dmv.models.usage import CostEstimate, TokenUsage


@dataclass(frozen=True)
class ModelPricing:
    input_per_million: float
    output_per_million: float
    cached_input_per_million: float | None = None
    long_context_input_threshold: int | None = None
    long_context_input_per_million: float | None = None
    long_context_output_per_million: float | None = None


MODEL_PRICING_BY_PREFIX: tuple[tuple[str, ModelPricing], ...] = (
    (
        "gpt-5.4-pro",
        ModelPricing(
            input_per_million=30.0,
            output_per_million=180.0,
            long_context_input_threshold=272_000,
            long_context_input_per_million=60.0,
            long_context_output_per_million=270.0,
        ),
    ),
    (
        "gpt-5.4",
        ModelPricing(
            input_per_million=2.50,
            output_per_million=15.00,
            cached_input_per_million=0.25,
            long_context_input_threshold=272_000,
            long_context_input_per_million=5.00,
            long_context_output_per_million=22.50,
        ),
    ),
    (
        "gpt-5",
        ModelPricing(
            input_per_million=1.25,
            output_per_million=10.00,
            cached_input_per_million=0.125,
        ),
    ),
    (
        "gpt-4o",
        ModelPricing(
            input_per_million=2.50,
            output_per_million=10.00,
            cached_input_per_million=1.25,
        ),
    ),
    (
        "gpt-4.1",
        ModelPricing(
            input_per_million=2.00,
            output_per_million=8.00,
            cached_input_per_million=0.50,
        ),
    ),
    (
        "gemini-3.1-pro-preview",
        ModelPricing(
            input_per_million=2.00,
            output_per_million=12.00,
            cached_input_per_million=0.20,
            long_context_input_threshold=200_000,
            long_context_input_per_million=4.00,
            long_context_output_per_million=18.00,
        ),
    ),
    (
        "gemini-3.1-pro",
        ModelPricing(
            input_per_million=2.00,
            output_per_million=12.00,
            cached_input_per_million=0.20,
            long_context_input_threshold=200_000,
            long_context_input_per_million=4.00,
            long_context_output_per_million=18.00,
        ),
    ),
)


def resolve_model_pricing(model: str, settings: Settings) -> tuple[ModelPricing, str] | None:
    if (
        settings.openai_input_price_per_million is not None
        and settings.openai_output_price_per_million is not None
    ):
        return (
            ModelPricing(
                input_per_million=settings.openai_input_price_per_million,
                output_per_million=settings.openai_output_price_per_million,
                cached_input_per_million=settings.openai_cached_input_price_per_million,
            ),
            "env override",
        )

    normalized = model.lower()
    for prefix, pricing in MODEL_PRICING_BY_PREFIX:
        if normalized.startswith(prefix):
            return pricing, f"built-in table ({prefix})"

    return None


def estimate_cost(
    usage: TokenUsage,
    settings: Settings,
) -> CostEstimate | None:
    resolved = resolve_model_pricing(usage.model or settings.active_model, settings)
    if resolved is None:
        return None

    pricing, source = resolved
    input_rate, output_rate = _select_rates(usage.input_tokens, pricing)
    cached_rate = pricing.cached_input_per_million

    cached_tokens = min(usage.cached_input_tokens, usage.input_tokens)
    billable_input_tokens = usage.input_tokens - cached_tokens

    input_cost = (billable_input_tokens / 1_000_000) * input_rate
    cached_cost = 0.0
    if cached_tokens and cached_rate is not None:
        cached_cost = (cached_tokens / 1_000_000) * cached_rate
    elif cached_tokens:
        cached_cost = (cached_tokens / 1_000_000) * input_rate

    output_cost = (usage.output_tokens / 1_000_000) * output_rate
    amount = input_cost + cached_cost + output_cost

    return CostEstimate(
        amount_usd=amount,
        pricing_source=source,
        input_price_per_million=input_rate,
        output_price_per_million=output_rate,
        cached_input_price_per_million=cached_rate,
    )


def _select_rates(input_tokens: int, pricing: ModelPricing) -> tuple[float, float]:
    threshold = pricing.long_context_input_threshold
    if (
        threshold is not None
        and input_tokens > threshold
        and pricing.long_context_input_per_million is not None
        and pricing.long_context_output_per_million is not None
    ):
        return (
            pricing.long_context_input_per_million,
            pricing.long_context_output_per_million,
        )

    return pricing.input_per_million, pricing.output_per_million
