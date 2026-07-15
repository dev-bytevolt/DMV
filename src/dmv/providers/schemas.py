from __future__ import annotations

from dmv.document_types import DOCUMENT_TYPE_IDS

CLASSIFICATION_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "documents": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": list(DOCUMENT_TYPE_IDS)},
                    "pages": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1},
                    },
                },
                "required": ["id", "name", "type", "pages"],
                "additionalProperties": False,
            },
        },
        "empty_pages": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
        },
    },
    "required": ["documents", "empty_pages"],
    "additionalProperties": False,
}
