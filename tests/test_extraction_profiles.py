from dmv.document_types import DOCUMENT_TYPE_IDS
from dmv.extraction.fields import CANONICAL_EXTRACTION_FIELDS
from dmv.extraction.profiles import (
    EXTRACTION_FIELDS_BY_DOCUMENT_TYPE,
    fields_for_document_type,
)
from dmv.extraction.prompt import build_extraction_prompt
from dmv.extraction.schemas import (
    EXTRACTION_JSON_SCHEMA,
    build_extraction_json_schema,
    normalize_extraction_payload,
)


def test_fields_for_document_type_narrows_known_types() -> None:
    driver_fields = fields_for_document_type("driver_license")
    assert "driver_full_name" in driver_fields
    assert "driver_license_number" in driver_fields
    assert "owner_full_name" not in driver_fields
    assert "lessee_name" not in driver_fields
    assert "insured_name" not in driver_fields

    insurance_fields = fields_for_document_type("insurance_card")
    assert "insured_name" in insurance_fields
    assert "vehicle_vin" in insurance_fields
    assert "owner_full_name" not in insurance_fields
    assert "driver_full_name" not in insurance_fields

    lease_fields = fields_for_document_type("lease_agreement")
    assert "lessee_name" in lease_fields
    assert "lessor_name" in lease_fields
    assert "owner_full_name" not in lease_fields


def test_fields_for_other_and_unknown_use_bounded_other_profile() -> None:
    other_fields = fields_for_document_type("other")
    unknown_fields = fields_for_document_type("totally_unknown_type")
    assert other_fields == EXTRACTION_FIELDS_BY_DOCUMENT_TYPE["other"]
    assert unknown_fields == other_fields
    # Bounded so Vertex structured output accepts the schema.
    assert len(other_fields) < len(CANONICAL_EXTRACTION_FIELDS)
    assert "vehicle_vin" in other_fields
    assert "owner_full_name" in other_fields
    assert "extra" not in other_fields


def test_every_profile_field_is_canonical() -> None:
    for document_type, fields in EXTRACTION_FIELDS_BY_DOCUMENT_TYPE.items():
        for field in fields:
            assert field in CANONICAL_EXTRACTION_FIELDS, (
                f"{document_type} references unknown field {field}"
            )


def test_build_extraction_json_schema_is_type_specific() -> None:
    driver_schema = build_extraction_json_schema("driver_license")
    properties = driver_schema["properties"]
    assert "driver_full_name" in properties
    assert "owner_full_name" not in properties
    assert "extra" in properties
    assert driver_schema["required"] == ["document_type", "source_document_name"]

    other_schema = build_extraction_json_schema("other")
    other_fields = fields_for_document_type("other")
    for field in other_fields:
        assert field in other_schema["properties"]
    assert "driver_full_name" not in other_schema["properties"]
    assert other_schema == EXTRACTION_JSON_SCHEMA


def test_build_extraction_prompt_lists_only_allowed_fields() -> None:
    prompt = build_extraction_prompt(
        document_type="driver_license",
        document_name="Driver License Copy",
        source_filename="Driver_License_Copy.pdf",
    )
    assert "driver_first_name" in prompt
    assert "driver_full_name" in prompt
    assert "owner_full_name" not in prompt
    assert "vehicle_vin" not in prompt
    assert "Do not use owner_*" in prompt or "Do not use owner_" in prompt
    assert "Never output a field that is not in the allowed list" in prompt


def test_build_extraction_prompt_for_other_uses_bounded_list() -> None:
    prompt = build_extraction_prompt(
        document_type="other",
        document_name="Misc Form",
        source_filename="Misc.pdf",
    )
    assert "unrecognized or miscellaneous" in prompt
    assert "vehicle_vin" in prompt
    assert "owner_full_name" in prompt
    assert "driver_full_name" not in prompt
    assert "full canonical field list" not in prompt


def test_normalize_extraction_payload_drops_disallowed_fields() -> None:
    payload = {
        "document_type": "insurance_card",
        "source_document_name": "GEICO.pdf",
        "insured_name": {"value": "Jane Doe", "confidence": 0.9},
        "owner_full_name": {"value": "Should Not Appear", "confidence": 0.9},
        "vehicle_vin": {"value": "1HGCM82633A004352", "confidence": 0.95},
    }
    normalized = normalize_extraction_payload(payload, document_type="insurance_card")
    assert "insured_name" in normalized
    assert "vehicle_vin" in normalized
    assert "owner_full_name" not in normalized


def test_profiles_cover_known_document_types() -> None:
    for type_id in DOCUMENT_TYPE_IDS:
        assert type_id in EXTRACTION_FIELDS_BY_DOCUMENT_TYPE
