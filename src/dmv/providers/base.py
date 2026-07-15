from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from dmv.models.classification import ClassificationResult
from dmv.models.usage import TokenUsage


@dataclass(frozen=True)
class ClassificationOutcome:
    result: ClassificationResult
    usage: TokenUsage


class ClassificationProvider(ABC):
    @abstractmethod
    async def classify_pdf(self, pdf_path: Path) -> ClassificationOutcome:
        """Upload, classify, and always delete the remote file."""
