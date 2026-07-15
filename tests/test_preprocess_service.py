from pathlib import Path

import pytest

from dmv.config import Settings
from dmv.preprocess.service import PreprocessingService


@pytest.mark.asyncio
async def test_preprocessing_service_processes_classified_directory(
    settings: Settings,
    skewed_content_pdf,
    tmp_path: Path,
) -> None:
    classified_dir = tmp_path / "classified"
    corrected_dir = tmp_path / "corrected"
    classified_dir.mkdir()
    target = classified_dir / "sample_doc.pdf"
    target.write_bytes(skewed_content_pdf.read_bytes())

    service = PreprocessingService(settings)
    result = await service.preprocess_directory(classified_dir, corrected_dir)

    assert result.stats.documents_processed == 1
    assert result.stats.pages_processed == 1
    assert (corrected_dir / "sample_doc.pdf").exists()


@pytest.mark.asyncio
async def test_preprocessing_service_handles_empty_directory(
    settings: Settings,
    tmp_path: Path,
) -> None:
    classified_dir = tmp_path / "classified"
    corrected_dir = tmp_path / "corrected"
    classified_dir.mkdir()

    service = PreprocessingService(settings)
    result = await service.preprocess_directory(classified_dir, corrected_dir)

    assert result.stats.documents_processed == 0
    assert result.stats.pages_processed == 0
