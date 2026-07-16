from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dmv.models.classification import ClassificationResult
from dmv.models.usage import TokenUsage


@dataclass(frozen=True)
class ClassificationOutcome:
    result: ClassificationResult
    usage: TokenUsage


@dataclass(frozen=True)
class ExtractionOutcome:
    result: dict[str, Any]
    usage: TokenUsage


class ClassificationProvider(ABC):
    @abstractmethod
    async def classify_pdf(self, pdf_path: Path) -> ClassificationOutcome:
        """Upload, classify, and always delete the remote file."""


class ExtractionProvider(ABC):
    @abstractmethod
    async def extract_pdf(
        self,
        pdf_path: Path,
        *,
        document_type: str,
        document_name: str,
    ) -> ExtractionOutcome:
        """Upload, extract fields, and always delete the remote file."""
