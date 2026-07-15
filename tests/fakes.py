from __future__ import annotations

from pathlib import Path

from dmv.models.classification import ClassificationResult
from dmv.models.usage import TokenUsage
from dmv.providers.base import ClassificationOutcome, ClassificationProvider


class FakeProvider(ClassificationProvider):
    def __init__(
        self,
        classification: ClassificationResult,
        *,
        usage: TokenUsage | None = None,
    ) -> None:
        self.classification = classification
        self.usage = usage or TokenUsage(
            input_tokens=1000,
            output_tokens=200,
            total_tokens=1200,
            model="gpt-4o",
        )
        self.calls: list[Path] = []

    async def classify_pdf(self, pdf_path: Path) -> ClassificationOutcome:
        self.calls.append(pdf_path)
        return ClassificationOutcome(result=self.classification, usage=self.usage)
