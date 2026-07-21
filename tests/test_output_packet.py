from __future__ import annotations

import json
from pathlib import Path

import fitz
from pypdf import PdfReader

from dmv.models.classification import ClassificationResult
from dmv.output.cover_letter import build_cover_letter_pdf
from dmv.output.mappings import (
    OUTPUT_BA49,
    OUTPUT_COVER,
    OUTPUT_OWNERSHIP,
    OUTPUT_UTA,
    build_ba49_fields,
    build_ownership_fields,
    build_uta_fields,
)
from dmv.output.service import build_output_packet
from dmv.output.values import first_value, get_consolidated_value, truthy_flag


def test_get_consolidated_value_nested_and_flat() -> None:
    data = {
        "vehicle_vin": {"value": "VIN123", "confidence": 0.9, "review_required": False},
        "owner": {
            "full_name": {
                "value": "Jane Doe",
                "confidence": 0.9,
                "review_required": False,
            },
            "address": {
                "street": {
                    "value": "1 Main St",
                    "confidence": 0.8,
                    "review_required": False,
                }
            },
        },
    }
    assert get_consolidated_value(data, "vehicle_vin") == "VIN123"
    assert get_consolidated_value(data, "owner.full_name") == "Jane Doe"
    assert get_consolidated_value(data, "owner.address.street") == "1 Main St"
    assert get_consolidated_value(data, "missing.path") is None


def test_first_value_fallbacks() -> None:
    data = {
        "owner": {
            "name": {"value": "Alt Name", "confidence": 0.9, "review_required": False}
        }
    }
    assert first_value(data, "owner.full_name", "owner.name") == "Alt Name"


def test_truthy_flag() -> None:
    assert truthy_flag("YES") is True
    assert truthy_flag("no") is False
    assert truthy_flag("maybe") is None


def test_uta_mapping_uses_nested_entities() -> None:
    data = {
        "vehicle_vin": {"value": "5NMMBDTL6TH064502", "confidence": 1.0},
        "owner": {
            "full_name": {"value": "TOYOTA LEASE TRUST", "confidence": 1.0},
            "address": {"street": {"value": "PO BOX 1", "confidence": 1.0}},
        },
        "lienholder": {"name": {"value": "TOYOTA LEASE TRUST", "confidence": 1.0}},
        "odometer_not_actual": {"value": "yes", "confidence": 1.0},
    }
    fields = build_uta_fields(data)
    assert fields["Vehicle Identification Number VIN"] == "5NMMBDTL6TH064502"
    assert fields["Owner Full Name or Entity Name"] == "TOYOTA LEASE TRUST"
    assert fields["Address"] == "PO BOX 1"
    assert fields["Lienholder Name"] == "TOYOTA LEASE TRUST"
    assert fields["N  Not actual mileage"] == "/Yes"


def test_ba49_and_ownership_mappings() -> None:
    data = {
        "vehicle_vin": {"value": "ABC", "confidence": 1.0},
        "plate_number": {"value": "XYZ12", "confidence": 1.0},
        "dealership": {
            "name": {"value": "Interstate Toyota", "confidence": 1.0},
            "address": {"city": {"value": "Paramus", "confidence": 1.0}},
        },
        "vehicle_make": {"value": "TOYOTA", "confidence": 1.0},
    }
    ba49 = build_ba49_fields(data)
    ownership = build_ownership_fields(data)
    assert ba49["Vehicle Identification Number (VIN)"] == "ABC"
    assert ba49["Plate Number"] == "XYZ12"
    assert ownership["New Vehicle Dealership Name"] == "Interstate Toyota"
    assert ownership["City  Town"] == "Paramus"
    assert ownership["Make"] == "TOYOTA"


def test_build_cover_letter_contains_vin(tmp_path: Path) -> None:
    data = {
        "vehicle_vin": {"value": "TESTVIN00000000001", "confidence": 1.0},
        "customer_name": {"value": "Angela Hajal", "confidence": 1.0},
        "collect_taxes": {"value": "yes", "confidence": 1.0},
        "sales_tax": {"value": "100.00", "confidence": 1.0},
    }
    out = tmp_path / "cover.pdf"
    build_cover_letter_pdf(data, out)
    assert out.is_file()
    doc = fitz.open(out)
    assert doc.page_count == 1
    page = doc[0]
    # US Letter (matches Word cover-sheet blank)
    assert abs(page.rect.width - 612.0) < 1.0
    assert abs(page.rect.height - 792.0) < 1.0
    text = page.get_text().replace("\xa0", " ").replace("\xad", "")
    drawings = page.get_drawings()
    doc.close()
    assert "TESTVIN00000000001" in text
    assert "Angela Hajal" in text
    assert "YES" in text
    assert "NO" in text
    assert "Purchase Price" in text
    assert "NJ DMV SHOULD COLLECT LFIS" in text
    assert "PLEASE SEND PLATES, REGISTRATION" in text
    assert "RETURN DOCUMENTS TO MV EXPRESS" in text
    assert "postage" in text and "return envelope" in text
    assert "Shloime" in text
    assert "shloime@getplatesfast.com" in text
    # Selected YES/NO is circled with an oval path.
    assert any(d.get("items") for d in drawings)


def _write_tiny_pdf(path: Path, label: str) -> None:
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    page.insert_text((20, 40), label, fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    doc.close()


def test_build_output_packet_respects_debug_mode(tmp_path: Path) -> None:
    blanks_dir = Path("artifacts/blanks")
    assert (blanks_dir / "BA-49 BLANK-2022.pdf").is_file()

    artifact_dir = tmp_path / "packet"
    classified_dir = artifact_dir / "classified"
    classified_dir.mkdir(parents=True)

    classification = ClassificationResult.from_dict(
        {
            "documents": [
                {
                    "id": "doc-001",
                    "name": "MV Express Cover Letter",
                    "type": "cover_letter",
                    "pages": [1],
                },
                {
                    "id": "doc-002",
                    "name": "Driver License",
                    "type": "driver_license",
                    "pages": [2],
                },
                {
                    "id": "doc-003",
                    "name": "Dealer Invoice",
                    "type": "dealer_invoice",
                    "pages": [3],
                },
            ],
            "empty_pages": [],
        }
    )
    _write_tiny_pdf(classified_dir / "MV_Express_Cover_Letter.pdf", "cover-src")
    _write_tiny_pdf(classified_dir / "Driver_License.pdf", "dl")
    _write_tiny_pdf(classified_dir / "Dealer_Invoice.pdf", "invoice")

    consolidated = {
        "vehicle_vin": {"value": "VINPACKED000000001", "confidence": 1.0},
        "customer_name": {"value": "Test Customer", "confidence": 1.0},
        "owner": {"full_name": {"value": "Owner Name", "confidence": 1.0}},
        "dealership": {"name": {"value": "Dealer", "confidence": 1.0}},
    }
    consolidated_json = artifact_dir / "consolidated_data.json"
    consolidated_json.write_text(json.dumps(consolidated), encoding="utf-8")

    result = build_output_packet(
        artifact_dir=artifact_dir,
        classified_dir=classified_dir,
        consolidated_json=consolidated_json,
        classification=classification,
        blanks_dir=blanks_dir,
        debug_mode=True,
    )

    assert result.cover_letter_pdf.is_file()
    assert result.uta_pdf.is_file()
    assert result.ba49_pdf.is_file()
    assert result.ownership_pdf.is_file()
    assert (result.output_dir / OUTPUT_COVER).is_file()
    assert (result.output_dir / OUTPUT_UTA).is_file()
    assert (result.output_dir / OUTPUT_BA49).is_file()
    assert (result.output_dir / OUTPUT_OWNERSHIP).is_file()
    assert result.output_pdf == artifact_dir / "output.pdf"
    assert result.output_pdf.is_file()
    # cover excluded by debug_mode → only DL + invoice
    assert result.appended_document_count == 2
    # cover(1) + UTA(2) + BA49(1) + ownership(1) + 2 classified = 7
    assert result.page_count == 7

    filled_ba49 = PdfReader(str(result.ba49_pdf))
    vin_field = filled_ba49.get_fields()["Vehicle Identification Number (VIN)"]
    assert vin_field.get("/V") == "VINPACKED000000001"

    # Without debug mode, cover letter classified doc is also appended.
    result_all = build_output_packet(
        artifact_dir=artifact_dir,
        classified_dir=classified_dir,
        consolidated_json=consolidated_json,
        classification=classification,
        blanks_dir=blanks_dir,
        debug_mode=False,
    )
    assert result_all.appended_document_count == 3
    assert result_all.page_count == 8
