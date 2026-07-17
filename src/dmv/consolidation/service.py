from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from dmv.consolidation.field import consolidate_field
from dmv.consolidation.groups import (
    ENTITY_GROUPS,
    consolidate_entity_group,
    count_consolidated_fields,
    grouped_canonical_fields,
)
from dmv.extraction.fields import CANONICAL_EXTRACTION_FIELDS

logger = logging.getLogger(__name__)

CONSOLIDATED_DATA_FILENAME = "consolidated_data.json"


@dataclass(frozen=True)
class ConsolidationResult:
    artifact_dir: Path
    output_json: Path
    field_count: int
    fields_without_review: int
    extra_document_count: int

    @property
    def review_pass_percent(self) -> float:
        if self.field_count <= 0:
            return 100.0
        return 100.0 * self.fields_without_review / self.field_count


def _field_value(raw: object) -> tuple[str, float] | None:
    if isinstance(raw, dict):
        value = str(raw.get("value", "")).strip()
        confidence = raw.get("confidence")
        if not value or confidence is None:
            return None
        try:
            confidence_f = float(confidence)
        except (TypeError, ValueError):
            return None
        return value, max(0.0, min(1.0, confidence_f))
    if isinstance(raw, str):
        value = raw.strip()
        if value:
            return value, 1.0
    return None


def _load_payloads(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def consolidate_extractions(
    extracted_dir: Path,
    artifact_dir: Path,
) -> ConsolidationResult:
    # field -> list[(value, confidence, document_name, document_type)]
    field_hypotheses: dict[str, list[tuple[str, float, str, str]]] = {
        field: [] for field in CANONICAL_EXTRACTION_FIELDS
    }
    extra_by_document: dict[str, list[dict]] = {}

    json_paths = sorted(
        path
        for path in extracted_dir.glob("*.json")
        if path.is_file() and path.parent.name != "_chunks"
    )

    for json_path in json_paths:
        for payload in _load_payloads(json_path):
            payload_document_type = str(payload.get("document_type", "")).strip()
            document_name = str(payload.get("source_document_name", "")).strip()
            if not document_name:
                document_name = json_path.stem.replace("_", " ")

            for field in CANONICAL_EXTRACTION_FIELDS:
                parsed = _field_value(payload.get(field))
                if parsed is not None:
                    value, confidence = parsed
                    field_hypotheses[field].append(
                        (value, confidence, document_name, payload_document_type)
                    )

            extra_items = payload.get("extra")
            if not isinstance(extra_items, list) or not extra_items:
                continue
            extra_by_document.setdefault(document_name, []).extend(
                item for item in extra_items if isinstance(item, dict)
            )

    consolidated: dict[str, object] = {}
    nested_fields = grouped_canonical_fields()

    # Nested entity groups (owner / lienholder / dealership / …).
    for group in ENTITY_GROUPS:
        nest = consolidate_entity_group(field_hypotheses, group)
        if nest:
            consolidated[group.nest_key] = nest

    # Remaining flat canonical fields.
    for field in CANONICAL_EXTRACTION_FIELDS:
        if field in nested_fields:
            continue
        hypotheses = field_hypotheses[field]
        if not hypotheses:
            continue

        consolidated_field = consolidate_field(
            hypotheses,
            use_vin=field == "vehicle_vin",
        )
        if consolidated_field is not None:
            consolidated[field] = consolidated_field.to_dict()

    if extra_by_document:
        consolidated["extra"] = extra_by_document

    field_count, fields_without_review = count_consolidated_fields(consolidated)

    output_json = artifact_dir / CONSOLIDATED_DATA_FILENAME
    output_json.write_text(
        json.dumps(consolidated, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Consolidated %s field(s) and %s extra document(s) -> %s",
        field_count,
        len(extra_by_document),
        output_json.name,
    )
    return ConsolidationResult(
        artifact_dir=artifact_dir,
        output_json=output_json,
        field_count=field_count,
        fields_without_review=fields_without_review,
        extra_document_count=len(extra_by_document),
    )
