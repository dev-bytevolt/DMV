import json
from pathlib import Path

import cv2
import fitz
import numpy as np
import pytest

from dmv.extraction.fields import CANONICAL_EXTRACTION_FIELDS
from dmv.extraction.schemas import (
    EXTRACTION_JSON_SCHEMA,
    FIELD_VALUE_SCHEMA,
    build_extraction_json_schema,
    normalize_extraction_payload,
)
from dmv.extraction.service import EXTRACTION_UPLOAD_MAX_BYTES, ExtractionService
from dmv.models.classification import ClassificationResult
from dmv.models.usage import TokenUsage
from tests.fakes import FakeExtractionProvider


def _make_image_heavy_pdf(
    path: Path,
    *,
    pages: int = 5,
    width: int = 900,
    height: int = 1200,
    quality: int = 90,
) -> None:
    doc = fitz.open()
    for page_num in range(pages):
        page = doc.new_page(width=612, height=792)
        base = np.full((height, width, 3), (page_num * 40) % 256, dtype=np.uint8)
        noise = np.random.default_rng(page_num).integers(0, 256, base.shape, dtype=np.uint8)
        image = cv2.addWeighted(base, 0.5, noise, 0.5, 0)
        ok, buf = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
        assert ok
        page.insert_image(page.rect, stream=bytes(buf))
    doc.save(path)
    doc.close()


def test_extraction_json_schema_includes_canonical_fields() -> None:
    properties = EXTRACTION_JSON_SCHEMA["properties"]
    for field in CANONICAL_EXTRACTION_FIELDS:
        assert field in properties
    assert "extra" in properties
    required = EXTRACTION_JSON_SCHEMA["required"]
    assert required == ["document_type", "source_document_name"]
    for field in CANONICAL_EXTRACTION_FIELDS:
        assert field not in required
    assert "extra" not in required
    assert EXTRACTION_JSON_SCHEMA["properties"]["vehicle_vin"] == FIELD_VALUE_SCHEMA
    assert build_extraction_json_schema("other") == EXTRACTION_JSON_SCHEMA
    lease_schema = build_extraction_json_schema("lease_agreement")
    assert "lessee_name" in lease_schema["properties"]
    assert "owner_full_name" not in lease_schema["properties"]


def test_normalize_extraction_payload_drops_empty_fields() -> None:
    payload = {
        "document_type": "check_payment",
        "source_document_name": "Check_Payment.pdf",
        "check_amount": {"value": "1500.00", "confidence": 0.98},
        "driver_first_name": {"value": "", "confidence": 0.5},
        "driver_last_name": {"value": "  ", "confidence": 0.9},
        "extra": [
            {
                "field_name": "memo",
                "value": "down payment",
                "confidence": 0.85,
            },
            {"field_name": "blank", "value": "", "confidence": 0.5},
            {
                "field_name": "uncertain",
                "value": "maybe",
                "confidence": 1.5,
            },
        ],
    }
    normalized = normalize_extraction_payload(payload)
    assert normalized == {
        "document_type": "check_payment",
        "source_document_name": "Check_Payment.pdf",
        "check_amount": {"value": "1500.00", "confidence": 0.98},
        "extra": [
            {
                "field_name": "memo",
                "value": "down payment",
                "confidence": 0.85,
            },
            {
                "field_name": "uncertain",
                "value": "maybe",
                "confidence": 1.0,
            },
        ],
    }


def test_normalize_extraction_payload_accepts_legacy_string_values() -> None:
    payload = {
        "document_type": "check_payment",
        "source_document_name": "Check_Payment.pdf",
        "check_amount": "1500.00",
    }
    normalized = normalize_extraction_payload(payload)
    assert normalized["check_amount"] == {"value": "1500.00", "confidence": 1.0}


@pytest.mark.asyncio
async def test_extraction_service_writes_json_per_corrected_pdf(
    settings,
    tmp_path: Path,
) -> None:
    classification = ClassificationResult.from_dict(
        {
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
    )
    corrected_dir = tmp_path / "corrected"
    extracted_dir = tmp_path / "extracted"
    corrected_dir.mkdir()
    (corrected_dir / "Driver_License_Copy.pdf").write_bytes(b"%PDF-1.4")
    (corrected_dir / "Insurance_Card.pdf").write_bytes(b"%PDF-1.4")

    provider = FakeExtractionProvider(
        result={
            "document_type": "driver_license",
            "source_document_name": "Driver_License_Copy.pdf",
            "driver_full_name": {"value": "Test Driver", "confidence": 0.92},
            "driver_first_name": {"value": "", "confidence": 0.1},
        }
    )
    service = ExtractionService(settings, provider=provider)
    result = await service.extract_directory(
        corrected_dir,
        extracted_dir,
        classification=classification,
        debug_mode=False,
    )

    assert result.stats.documents_processed == 2
    assert len(provider.calls) == 2
    assert (extracted_dir / "Driver_License_Copy.json").is_file()
    assert (extracted_dir / "Insurance_Card.json").is_file()

    payload = json.loads(
        (extracted_dir / "Driver_License_Copy.json").read_text(encoding="utf-8")
    )
    assert payload["document_type"] == "driver_license"
    assert payload["source_document_name"] == "Driver_License_Copy.pdf"
    assert payload["driver_full_name"] == {
        "value": "Test Driver",
        "confidence": 0.92,
    }
    assert "driver_first_name" not in payload
    assert "extra" not in payload


@pytest.mark.asyncio
async def test_extraction_service_skips_debug_excluded_documents(
    settings,
    tmp_path: Path,
) -> None:
    classification = ClassificationResult.from_dict(
        {
            "documents": [
                {
                    "id": "doc-001",
                    "name": "MV Express Cover Letter to NJ DMV",
                    "type": "cover_letter",
                    "pages": [1],
                },
                {
                    "id": "doc-002",
                    "name": "Driver License Copy",
                    "type": "driver_license",
                    "pages": [2],
                },
            ],
            "empty_pages": [],
        }
    )
    corrected_dir = tmp_path / "corrected"
    extracted_dir = tmp_path / "extracted"
    corrected_dir.mkdir()
    (corrected_dir / "MV_Express_Cover_Letter_to_NJ_DMV.pdf").write_bytes(b"%PDF-1.4")
    (corrected_dir / "Driver_License_Copy.pdf").write_bytes(b"%PDF-1.4")

    provider = FakeExtractionProvider()
    service = ExtractionService(settings, provider=provider)
    result = await service.extract_directory(
        corrected_dir,
        extracted_dir,
        classification=classification,
        debug_mode=True,
    )

    assert result.stats.documents_processed == 1
    assert len(provider.calls) == 1
    assert provider.calls[0][0].name == "Driver_License_Copy.pdf"
    assert not (extracted_dir / "MV_Express_Cover_Letter_to_NJ_DMV.json").exists()


def test_split_pdf_for_upload_produces_parts_under_limit(
    settings,
    tmp_path: Path,
) -> None:
    source_pdf = tmp_path / "large.pdf"
    extracted_dir = tmp_path / "extracted"
    _make_image_heavy_pdf(
        source_pdf,
        pages=6,
        width=240,
        height=180,
        quality=70,
    )
    max_bytes = 80_000

    service = ExtractionService(settings, provider=FakeExtractionProvider())
    part_paths = service._split_pdf_for_upload(
        source_pdf,
        extracted_dir=extracted_dir,
        max_bytes=max_bytes,
    )

    assert len(part_paths) > 1
    total_pages = 0
    for part_path in part_paths:
        assert part_path.stat().st_size <= max_bytes
        with fitz.open(part_path) as part_doc:
            total_pages += len(part_doc)
    with fitz.open(source_pdf) as source_doc:
        assert total_pages == len(source_doc)


def test_split_pdf_for_upload_keeps_single_oversized_page(
    settings,
    tmp_path: Path,
) -> None:
    source_pdf = tmp_path / "huge_page.pdf"
    extracted_dir = tmp_path / "extracted"
    _make_image_heavy_pdf(source_pdf, pages=1)
    max_bytes = 10_000

    service = ExtractionService(settings, provider=FakeExtractionProvider())
    part_paths = service._split_pdf_for_upload(
        source_pdf,
        extracted_dir=extracted_dir,
        max_bytes=max_bytes,
    )

    assert len(part_paths) == 1
    with fitz.open(part_paths[0]) as part_doc:
        assert len(part_doc) == 1


@pytest.mark.asyncio
async def test_extraction_service_splits_oversized_pdf(
    settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dmv.extraction.service.EXTRACTION_UPLOAD_MAX_BYTES",
        80_000,
    )

    classification = ClassificationResult.from_dict(
        {
            "documents": [
                {
                    "id": "doc-001",
                    "name": "Motor Vehicle Lease Agreement",
                    "type": "motor_vehicle_lease_agreement",
                    "pages": [1, 2, 3, 4, 5, 6],
                },
            ],
            "empty_pages": [],
        }
    )
    corrected_dir = tmp_path / "corrected"
    extracted_dir = tmp_path / "extracted"
    corrected_dir.mkdir()
    large_pdf = corrected_dir / "Motor_Vehicle_Lease_Agreement.pdf"
    _make_image_heavy_pdf(
        large_pdf,
        pages=6,
        width=240,
        height=180,
        quality=70,
    )
    assert large_pdf.stat().st_size > 80_000

    provider = FakeExtractionProvider()
    service = ExtractionService(settings, provider=provider)
    result = await service.extract_directory(
        corrected_dir,
        extracted_dir,
        classification=classification,
        debug_mode=False,
    )

    assert result.stats.documents_processed == 1
    assert len(provider.calls) > 1
    assert all("part" in call[2] for call in provider.calls)

    payload = json.loads(
        (extracted_dir / "Motor_Vehicle_Lease_Agreement.json").read_text(
            encoding="utf-8"
        )
    )
    assert isinstance(payload, list)
    assert len(payload) == len(provider.calls)
    assert all(item["document_type"] == "motor_vehicle_lease_agreement" for item in payload)
    assert (extracted_dir / "_chunks").is_dir()


def test_extraction_upload_max_bytes_is_below_provider_limit() -> None:
    assert EXTRACTION_UPLOAD_MAX_BYTES < 32 * 1024 * 1024
