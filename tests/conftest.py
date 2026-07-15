from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from dmv.config import Settings
from tests.fakes import FakeProvider


@pytest.fixture
def skewed_content_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "skewed.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    rect = fitz.Rect(72, 72, 540, 720)
    page.draw_rect(rect, color=(0, 0, 0), fill=(0, 0, 0))
    page.insert_text((100, 120), "Driver License Copy", fontsize=18)
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    for page_index in range(4):
        page = doc.new_page(width=612, height=792)
        if page_index < 3:
            page.draw_rect(fitz.Rect(72, 72, 540, 720), color=(0, 0, 0), fill=(0, 0, 0))
            page.insert_text((100, 120), f"Document page {page_index + 1}", fontsize=18)
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def sample_classification() -> dict:
    return {
        "documents": [
            {
                "id": "doc-001",
                "name": "Driver License Copy",
                "type": "driver_license",
                "pages": [1, 2],
            },
            {
                "id": "doc-002",
                "name": "Insurance Card",
                "type": "insurance_card",
                "pages": [3],
            },
        ],
        "empty_pages": [4],
    }


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        ai_provider="openai",
        openai_api_key="test-key",
        openai_model="gpt-4o",
        worker_pool_size=2,
        max_ai_retries=1,
        ai_retry_base_delay_seconds=0.01,
        artifacts_dir=tmp_path / "artifacts",
        openai_input_price_per_million=None,
        openai_output_price_per_million=None,
        openai_cached_input_price_per_million=None,
        preprocess_dpi=200,
    )
