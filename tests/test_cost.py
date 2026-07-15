import pytest

from dmv.config import Settings
from dmv.cost import estimate_cost, resolve_model_pricing
from dmv.models.usage import TokenUsage


def test_resolve_model_pricing_for_gpt_54_snapshot() -> None:
    settings = Settings(
        ai_provider="openai",
        openai_api_key="x",
        openai_model="gpt-5.4-2026-03-05",
        worker_pool_size=1,
        max_ai_retries=1,
        ai_retry_base_delay_seconds=0.01,
        artifacts_dir="artifacts",
        openai_input_price_per_million=None,
        openai_output_price_per_million=None,
        openai_cached_input_price_per_million=None,
        preprocess_dpi=200,
    )

    pricing, source = resolve_model_pricing("gpt-5.4-2026-03-05", settings)

    assert pricing.input_per_million == 2.50
    assert pricing.output_per_million == 15.00
    assert pricing.cached_input_per_million == 0.25
    assert "gpt-5.4" in source


def test_estimate_cost_uses_cached_input_rate() -> None:
    settings = Settings(
        ai_provider="openai",
        openai_api_key="x",
        openai_model="gpt-5.4-2026-03-05",
        worker_pool_size=1,
        max_ai_retries=1,
        ai_retry_base_delay_seconds=0.01,
        artifacts_dir="artifacts",
        openai_input_price_per_million=None,
        openai_output_price_per_million=None,
        openai_cached_input_price_per_million=None,
        preprocess_dpi=200,
    )
    usage = TokenUsage(
        input_tokens=200_000,
        output_tokens=50_000,
        total_tokens=250_000,
        cached_input_tokens=50_000,
        model="gpt-5.4-2026-03-05",
    )

    cost = estimate_cost(usage, settings)

    assert cost is not None
    # 150K billable input @ $2.50 + 50K cached @ $0.25 + 50K output @ $15.00
    assert cost.amount_usd == pytest.approx(1.1375)


def test_estimate_cost_respects_env_override() -> None:
    settings = Settings(
        ai_provider="openai",
        openai_api_key="x",
        openai_model="custom-model",
        worker_pool_size=1,
        max_ai_retries=1,
        ai_retry_base_delay_seconds=0.01,
        artifacts_dir="artifacts",
        openai_input_price_per_million=1.0,
        openai_output_price_per_million=2.0,
        openai_cached_input_price_per_million=None,
        preprocess_dpi=200,
    )
    usage = TokenUsage(
        input_tokens=1_000_000,
        output_tokens=500_000,
        total_tokens=1_500_000,
        model="custom-model",
    )

    cost = estimate_cost(usage, settings)

    assert cost is not None
    assert cost.amount_usd == pytest.approx(2.0)
    assert cost.pricing_source == "env override"
