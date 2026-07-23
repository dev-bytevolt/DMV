from __future__ import annotations

import json
from pathlib import Path

import fitz
from pypdf import PdfReader

from dmv.models.classification import ClassificationResult
from dmv.output.cover_letter import build_cover_letter_pdf
from dmv.output.fill_pdf import FORM_FIELD_FONTSIZE, fill_acroform_pdf
from dmv.output.mappings import (
    BLANK_UTA,
    OUTPUT_BA49,
    OUTPUT_COVER,
    OUTPUT_OWNERSHIP,
    OUTPUT_UTA,
    build_ba49_fields,
    build_ownership_fields,
    build_uta_fields,
)
from dmv.output.formatting import today_form_date
from dmv.output.cover_letter import cover_letter_fields
from dmv.output.service import build_output_packet
from dmv.output.tax_stamp import apply_uta_tax_stamp
from dmv.output.values import first_value, get_consolidated_value, truthy_flag


def test_uta_uses_mv_express_representative_defaults() -> None:
    fields = build_uta_fields({"vehicle_vin": {"value": "ABC", "confidence": 1.0}})
    assert fields["First Name"] == "DINA"
    assert fields["Last Name"] == "NAMDAR"
    assert "Telephone Number_3" not in fields
    assert fields["Address_2"] == "160 EMPIRE BLVD"
    assert fields["CityTown_3"] == "BROOKLYN"
    assert fields["State_3"] == "NY"
    assert fields["Zip Code_3"] == "11225"


def test_cover_and_ownership_use_today_date() -> None:
    data = {
        "vehicle_vin": {"value": "ABC", "confidence": 1.0},
        "sale_date": {"value": "01/01/2020", "confidence": 1.0},
        "check_date": {"value": "02/02/2020", "confidence": 1.0},
    }
    today = today_form_date()
    assert cover_letter_fields(data)["date"] == today
    assert build_ownership_fields(data)["Date"] == today


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
        "plate_type": {"value": "TRANSFER", "confidence": 1.0},
        "dealership": {
            "name": {"value": "Interstate Toyota", "confidence": 1.0},
            "address": {"city": {"value": "Paramus", "confidence": 1.0}},
        },
        "vehicle_make": {"value": "TOYOTA", "confidence": 1.0},
        "vehicle_epa_mpg_rating": {"value": "36", "confidence": 1.0},
        "gross_sales_lease_price": {"value": "45863.54", "confidence": 1.0},
    }
    ba49 = build_ba49_fields(data)
    ownership = build_ownership_fields(data)
    assert ba49["Vehicle Identification Number (VIN)"] == "ABC"
    assert ba49["Plate Number"] == "XYZ12"
    assert ownership["New Vehicle Dealership Name"] == "Interstate Toyota"
    assert ownership["City  Town"] == "Paramus"
    assert ownership["Make"] == "TOYOTA"
    assert ownership["Check Box2"] == "/Yes"
    assert ownership["Grose"].startswith("$")
    assert ownership[
        "ModelList the Average EPA miles per gallon rating Add both city and highway ratings and divide by 2 OR designate as Not Rated and skip to Step 4"
    ] == "36"


def test_uta_defaults_axles_to_two() -> None:
    fields = build_uta_fields({"vehicle_vin": {"value": "ABC", "confidence": 1.0}})
    assert fields["No of Axles"] == "2"


def test_ba49_puts_driver_on_owner_row_without_lessee() -> None:
    data = {
        "vehicle_vin": {"value": "VIN123", "confidence": 1.0},
        "owner": {"full_name": {"value": "WALTER A JAMROS", "confidence": 1.0}},
        "driver": {
            "license_number": {"value": "J0373 77661 04554", "confidence": 1.0},
            "gender": {"value": "M", "confidence": 1.0},
            "eyes_color": {"value": "BLU", "confidence": 1.0},
            "dob": {"value": "04-02-1955", "confidence": 1.0},
            "ssn": {"value": "148-50-4320", "confidence": 1.0},
        },
    }
    ba49 = build_ba49_fields(data)
    assert ba49["Text24.0.0"] == "J0373"
    assert ba49["Gender.0"] == "M"
    assert ba49["Text29.0.0"] == "148"
    assert "Text24.0.2" not in ba49
    assert "Name/Lessee" not in ba49


def test_ba49_lease_puts_corpcode_on_owner_and_driver_on_lessee() -> None:
    data = {
        "vehicle_vin": {"value": "VIN123", "confidence": 1.0},
        "owner": {
            "full_name": {"value": "TOYOTA LEASE TRUST", "confidence": 1.0},
            "license_or_entity_id": {"value": "89550 59097 78420", "confidence": 1.0},
        },
        "lessee": {"name": {"value": "ANGELA HAJAL", "confidence": 1.0}},
        "driver": {
            "license_number": {"value": "H0203 04300 62662", "confidence": 1.0},
            "gender": {"value": "F", "confidence": 1.0},
            "ssn": {"value": "157-15-4821", "confidence": 1.0},
        },
    }
    ba49 = build_ba49_fields(data)
    assert ba49["Text24.0.0"] == "89550"
    assert ba49["Text24.0.2"] == "H0203"
    assert ba49["Gender.2"] == "F"
    assert ba49["Text29.0.2"] == "157"
    assert "Gender.0" not in ba49

    fields = build_uta_fields(
        {
            "vehicle_year": {"value": "2026", "confidence": 1.0},
            "vehicle_make": {"value": "TOYOTA", "confidence": 1.0},
            "odometer_reading": {"value": "0000001", "confidence": 1.0},
            "plate_number": {"value": "ZLJ7766", "confidence": 1.0},
        }
    )
    assert fields["Year_2"] == "2026"
    assert fields["Make_2"] == "TOYOTA"
    assert "Year" not in fields
    assert "Make" not in fields
    assert fields["Odometer Reading at time of purchase"] == "1"
    assert "NJ License Plate Number" not in fields


def test_cover_vin_uses_last_eight() -> None:
    data = {
        "vehicle_vin": {"value": "5TDAAAB53TS143834", "confidence": 1.0},
    }
    assert cover_letter_fields(data)["vin"] == "TS143834"


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
    # Cover prints last 8 of the VIN.
    assert "00000001" in text
    assert "TESTVIN00000000001" not in text
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


def test_filled_form_text_uses_uniform_fontsize(tmp_path: Path) -> None:
    """Blank DA uses auto-size (0 Tf); bake must pin a single point size."""
    blanks_dir = Path("artifacts/blanks")
    data = {
        "vehicle_vin": {"value": "5TDAAAB53TS143834", "confidence": 1.0},
        "vehicle_year": {"value": "2026", "confidence": 1.0},
        "vehicle_make": {"value": "TOYOTA", "confidence": 1.0},
        "vehicle_model": {"value": "GRAND HIGHLANDER", "confidence": 1.0},
        "odometer_reading": {"value": "1", "confidence": 1.0},
        "owner": {"full_name": {"value": "TOYOTA LEASE TRUST", "confidence": 1.0}},
    }
    out = tmp_path / "uta.pdf"
    fill_acroform_pdf(blanks_dir / BLANK_UTA, out, build_uta_fields(data))
    doc = fitz.open(out)
    needles = {
        "5TDAAAB53TS143834",
        "2026",
        "TOYOTA",
        "GRAND HIGHLANDER",
        "1",
        "TOYOTA LEASE TRUST",
    }
    sizes: set[float] = set()
    for block in doc[0].get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if span["text"].strip() in needles:
                    sizes.add(round(float(span["size"]), 1))
    doc.close()
    assert sizes == {FORM_FIELD_FONTSIZE}

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
    assert filled_ba49.get_fields() in (None, {})
    ba49_doc = fitz.open(result.ba49_pdf)
    ba49_text = ba49_doc[0].get_text()
    assert not list(ba49_doc[0].widgets() or [])
    ba49_doc.close()
    assert "VINPACKED000000001" in ba49_text

    uta_doc = fitz.open(result.uta_pdf)
    uta_text = uta_doc[0].get_text().replace("\xa0", " ")
    assert not list(uta_doc[0].widgets() or [])
    # No sales tax prepaid → customer-pays stamp
    assert "Sales/Use Tax" in uta_text
    assert "Purchase Price" in uta_text
    assert "N.J. SALES TAX SATISFIED" not in uta_text
    uta_doc.close()

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


def test_uta_tax_stamp_both_variants(tmp_path: Path) -> None:
    blanks_dir = Path("artifacts/blanks")
    blank = blanks_dir / BLANK_UTA
    assert blank.is_file()
    data = {
        "purchase_price": {"value": "28279.70", "confidence": 1.0},
        "sales_tax": {"value": "1873.53", "confidence": 1.0},
        "sale_date": {"value": "6/17/2026", "confidence": 1.0},
        "dealership_entity_id": {"value": "7086161", "confidence": 1.0},
        "vehicle_vin": {"value": "TESTVIN00000000001", "confidence": 1.0},
    }
    customer = tmp_path / "uta_customer.pdf"
    dealer = tmp_path / "uta_dealer.pdf"
    fill_acroform_pdf(blank, customer, build_uta_fields(data))
    fill_acroform_pdf(blank, dealer, build_uta_fields(data))
    apply_uta_tax_stamp(customer, data, force_dealer_paid=False)
    apply_uta_tax_stamp(dealer, data, force_dealer_paid=True)

    customer_text = fitz.open(customer)[0].get_text()
    dealer_text = fitz.open(dealer)[0].get_text()
    assert "Sales/Use Tax" in customer_text
    assert "N.J. SALES TAX SATISFIED" not in customer_text
    assert "N.J. SALES TAX SATISFIED" in dealer_text
    assert "M.V. Ident No." in dealer_text
    assert "Dealer's Signature" in dealer_text
    assert "28279.70" in customer_text
    assert "1873.53" in customer_text
