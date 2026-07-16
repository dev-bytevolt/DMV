from __future__ import annotations

from pathlib import Path
from typing import Any

from dmv.models.classification import ClassificationResult
from dmv.models.usage import TokenUsage
from dmv.providers.base import (
    ClassificationOutcome,
    ClassificationProvider,
    ExtractionOutcome,
    ExtractionProvider,
)


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


class FakeExtractionProvider(ExtractionProvider):
    def __init__(
        self,
        *,
        result: dict[str, Any] | None = None,
        usage: TokenUsage | None = None,
    ) -> None:
        self.result = result or {
            "document_type": "driver_license",
            "source_document_name": "Driver_License_Copy.pdf",
            "driver_full_name": {"value": "Test Driver", "confidence": 0.95},
        }
        self.usage = usage or TokenUsage(
            input_tokens=500,
            output_tokens=100,
            total_tokens=600,
            model="gpt-4o",
        )
        self.calls: list[tuple[Path, str, str]] = []

    async def extract_pdf(
        self,
        pdf_path: Path,
        *,
        document_type: str,
        document_name: str,
    ) -> ExtractionOutcome:
        self.calls.append((pdf_path, document_type, document_name))
        payload = {
            **self.result,
            "document_type": document_type,
            "source_document_name": pdf_path.name,
        }
        return ExtractionOutcome(result=payload, usage=self.usage)
