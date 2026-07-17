from __future__ import annotations

import copy
from typing import Any

# Keywords commonly present in OpenAI structured-output schemas that Gemini's
# response_schema path rejects, or that are irrelevant for Vertex generation.
_DROP_KEYS = frozenset(
    {
        "$schema",
        "$id",
        "additionalProperties",
        "examples",
        "default",
        "title",
        "unevaluatedProperties",
        "patternProperties",
        "dependencies",
        "dependentRequired",
        "dependentSchemas",
    }
)


def openai_schema_to_google(schema: dict[str, Any]) -> dict[str, Any]:
    """Translate an OpenAI JSON Schema to Gemini / Vertex ``response_json_schema``.

    OpenAI structured outputs use a strict JSON Schema dialect (notably
    ``additionalProperties: false`` everywhere). Vertex / Gemini accept a
    JSON Schema subset via ``response_json_schema``; this helper:

    * deep-copies the input
    * strips unsupported / noisy keywords (e.g. ``additionalProperties``)
    * adds ``propertyOrdering`` for objects (preferred key order for Gemini)
    * normalizes ``type`` to lowercase JSON Schema strings
    * preserves ``$defs`` / ``definitions`` / ``$ref`` / ``enum`` / constraints
    """
    if not isinstance(schema, dict):
        raise TypeError(f"schema must be a dict, got {type(schema)!r}")
    return _convert_node(copy.deepcopy(schema))


def _convert_node(node: Any) -> Any:
    if isinstance(node, list):
        return [_convert_node(item) for item in node]
    if not isinstance(node, dict):
        return node

    converted: dict[str, Any] = {}
    for key, value in node.items():
        if key in _DROP_KEYS:
            continue
        if key == "type":
            converted[key] = _normalize_type(value)
            continue
        if key in {"properties", "$defs", "definitions"}:
            if not isinstance(value, dict):
                converted[key] = _convert_node(value)
                continue
            converted[key] = {
                prop_name: _convert_node(prop_schema)
                for prop_name, prop_schema in value.items()
            }
            continue
        converted[key] = _convert_node(value)

    if converted.get("type") == "object" and isinstance(converted.get("properties"), dict):
        properties = converted["properties"]
        # Gemini prefers an explicit property order matching schema declaration.
        if "propertyOrdering" not in converted and properties:
            converted["propertyOrdering"] = list(properties.keys())

    return converted


def _normalize_type(value: Any) -> Any:
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, list):
        return [_normalize_type(item) if isinstance(item, str) else item for item in value]
    return value
