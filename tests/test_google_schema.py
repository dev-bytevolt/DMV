from __future__ import annotations

from dmv.extraction.schemas import build_extraction_json_schema
from dmv.providers.google_schema import openai_schema_to_google
from dmv.providers.schemas import CLASSIFICATION_JSON_SCHEMA


def test_openai_schema_to_google_strips_additional_properties() -> None:
    converted = openai_schema_to_google(CLASSIFICATION_JSON_SCHEMA)

    assert "additionalProperties" not in converted
    assert converted["type"] == "object"
    assert converted["propertyOrdering"] == ["documents", "empty_pages"]

    document_item = converted["properties"]["documents"]["items"]
    assert "additionalProperties" not in document_item
    assert document_item["propertyOrdering"] == ["id", "name", "type", "pages"]
    assert document_item["properties"]["type"]["enum"]


def test_openai_schema_to_google_preserves_constraints_and_nested_objects() -> None:
    openai_schema = build_extraction_json_schema("driver_license")
    converted = openai_schema_to_google(openai_schema)

    assert "additionalProperties" not in converted
    assert converted["required"] == ["document_type", "source_document_name"]
    field = converted["properties"]["driver_full_name"]
    assert field["type"] == "object"
    assert "additionalProperties" not in field
    assert field["properties"]["confidence"]["minimum"] == 0
    assert field["properties"]["confidence"]["maximum"] == 1
    assert field["propertyOrdering"] == ["value", "confidence"]


def test_openai_schema_to_google_normalizes_uppercase_types() -> None:
    converted = openai_schema_to_google(
        {
            "type": "OBJECT",
            "properties": {
                "name": {"type": "STRING"},
                "count": {"type": "INTEGER", "minimum": 1},
            },
            "required": ["name"],
            "additionalProperties": False,
        }
    )
    assert converted["type"] == "object"
    assert converted["properties"]["name"]["type"] == "string"
    assert converted["properties"]["count"]["type"] == "integer"
    assert converted["propertyOrdering"] == ["name", "count"]
    assert "additionalProperties" not in converted
