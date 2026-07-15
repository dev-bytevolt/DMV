from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_input_tokens: int = 0
    model: str = ""

    @classmethod
    def empty(cls, *, model: str = "") -> TokenUsage:
        return cls(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            cached_input_tokens=0,
            model=model,
        )

    def merge(self, other: TokenUsage) -> TokenUsage:
        if self.model and other.model and self.model != other.model:
            model = self.model
        else:
            model = self.model or other.model

        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            model=model,
        )


@dataclass(frozen=True)
class CostEstimate:
    amount_usd: float
    pricing_source: str
    input_price_per_million: float
    output_price_per_million: float
    cached_input_price_per_million: float | None = None

    @property
    def pricing_label(self) -> str:
        cached = (
            f", cached ${self.cached_input_price_per_million:.2f}/1M"
            if self.cached_input_price_per_million is not None
            else ""
        )
        return (
            f"${self.input_price_per_million:.2f}/1M input, "
            f"${self.output_price_per_million:.2f}/1M output{cached} "
            f"({self.pricing_source})"
        )


@dataclass(frozen=True)
class ProcessingStats:
    elapsed_seconds: float
    usage: TokenUsage
    cost: CostEstimate | None = None
