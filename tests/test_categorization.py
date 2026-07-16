from __future__ import annotations

from pathlib import Path

import pytest

from dmv.categorization import CategorizationService
from dmv.config import Settings
from dmv.extraction.service import ExtractionService
from dmv.models.classification import ClassificationResult
from tests.fakes import FakeExtractionProvider, FakeProvider


@pytest.mark.asyncio
async def test_categorization_service_processes_file(
    settings: Settings,
    sample_pdf: Path,
    sample_classification,
) -> None:
    provider = FakeProvider(ClassificationResult.from_dict(sample_classification))
    extraction_service = ExtractionService(
        settings,
        provider=FakeExtractionProvider(),
    )
    service = CategorizationService(
        settings,
        provider=provider,
        extraction_service=extraction_service,
    )

    summary = await service.process_files([sample_pdf])

    assert len(summary.results) == 1
    result = summary.results[0]
    assert result.source_pdf == sample_pdf
    assert result.validation.is_valid is True
    assert result.stats.usage.total_tokens == 1200
    assert result.stats.cost is not None
    assert (result.artifact_dir / "original.pdf").exists()
    assert (result.artifact_dir / "doc_classification.json").exists()
    assert (result.classified_dir / "Driver_License_Copy.pdf").exists()
    assert (result.corrected_dir / "Driver_License_Copy.pdf").exists()
    assert result.extraction.stats.documents_processed == 2
    assert (result.extracted_dir / "Driver_License_Copy.json").exists()
    assert (result.consolidation.output_json).exists()
    assert result.preprocessing.stats.documents_processed == 2


@pytest.mark.asyncio
async def test_categorization_service_processes_multiple_files_in_parallel(
    settings: Settings,
    sample_pdf: Path,
    sample_classification,
    tmp_path: Path,
) -> None:
    second_pdf = tmp_path / "second.pdf"
    second_pdf.write_bytes(sample_pdf.read_bytes())

    provider = FakeProvider(ClassificationResult.from_dict(sample_classification))
    extraction_service = ExtractionService(
        settings,
        provider=FakeExtractionProvider(),
    )
    service = CategorizationService(
        settings,
        provider=provider,
        extraction_service=extraction_service,
    )

    summary = await service.process_files([sample_pdf, second_pdf])

    assert len(summary.results) == 2
    assert summary.total_usage.total_tokens == 4800
    assert provider.calls == [sample_pdf, second_pdf]


@pytest.mark.asyncio
async def test_categorization_service_rejects_non_pdf(settings: Settings, tmp_path: Path) -> None:
    text_file = tmp_path / "notes.txt"
    text_file.write_text("hello")

    service = CategorizationService(
        settings,
        provider=FakeProvider(ClassificationResult(documents=[], empty_pages=[])),
    )

    with pytest.raises(ValueError, match="Expected a PDF file"):
        await service.process_file(text_file)
