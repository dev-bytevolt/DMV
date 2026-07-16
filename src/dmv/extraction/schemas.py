from __future__ import annotations

from dmv.extraction.fields import CANONICAL_EXTRACTION_FIELDS
from dmv.extraction.profiles import fields_for_document_type

FIELD_VALUE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "value": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["value", "confidence"],
    "additionalProperties": False,
}

_EXTRA_ITEM_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "field_name": {"type": "string"},
        "value": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["field_name", "value", "confidence"],
    "additionalProperties": False,
}


def build_extraction_json_schema(document_type: str = "other") -> dict:
    """Build a JSON schema limited to fields allowed for ``document_type``."""
    allowed_fields = fields_for_document_type(document_type)
    properties: dict = {
        "document_type": {"type": "string"},
        "source_document_name": {"type": "string"},
        **{field: FIELD_VALUE_SCHEMA for field in allowed_fields},
        "extra": {
            "type": "array",
            "items": _EXTRA_ITEM_SCHEMA,
        },
    }
    return {
        "type": "object",
        "properties": properties,
        "required": [
            "document_type",
            "source_document_name",
        ],
        "additionalProperties": False,
    }


# Backward-compatible full schema (equivalent to document_type="other").
EXTRACTION_JSON_SCHEMA: dict = build_extraction_json_schema("other")


def _normalize_confidence(raw: object) -> float | None:
    try:
        confidence = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, confidence))


def _normalize_field_entry(raw: object) -> dict[str, object] | None:
    if isinstance(raw, dict):
        value = str(raw.get("value", "")).strip()
        confidence = _normalize_confidence(raw.get("confidence"))
        if not value or confidence is None:
            return None
        return {"value": value, "confidence": confidence}
    if isinstance(raw, str):
        value = raw.strip()
        if value:
            return {"value": value, "confidence": 1.0}
    return None


def _normalize_extra_item(raw: object) -> dict[str, object] | None:
    if not isinstance(raw, dict):
        return None
    field_name = str(raw.get("field_name", "")).strip()
    value = str(raw.get("value", "")).strip()
    confidence = _normalize_confidence(raw.get("confidence"))
    if not field_name or not value or confidence is None:
        return None
    return {
        "field_name": field_name,
        "value": value,
        "confidence": confidence,
    }


def normalize_extraction_payload(
    payload: dict,
    *,
    document_type: str | None = None,
) -> dict:
    """Drop empty fields and normalize each value to {value, confidence}.

    When ``document_type`` is provided, drop canonical fields that are not
    allowed for that type (defense in depth against model leakage).
    """
    payload_type = document_type or str(payload.get("document_type", ""))
    allowed = set(fields_for_document_type(payload_type)) | {
        "document_type",
        "source_document_name",
        "extra",
    }

    normalized: dict = {
        "document_type": str(payload.get("document_type", "")),
        "source_document_name": str(payload.get("source_document_name", "")),
    }

    for key, value in payload.items():
        if key in {"document_type", "source_document_name"}:
            continue
        if key not in allowed and key in CANONICAL_EXTRACTION_FIELDS:
            continue
        if key == "extra":
            if not isinstance(value, list):
                continue
            extra_items = []
            for item in value:
                normalized_item = _normalize_extra_item(item)
                if normalized_item is not None:
                    extra_items.append(normalized_item)
            if extra_items:
                normalized["extra"] = extra_items
            continue

        field_entry = _normalize_field_entry(value)
        if field_entry is not None:
            normalized[key] = field_entry

    return normalized
