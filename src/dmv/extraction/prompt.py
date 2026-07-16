from __future__ import annotations

from dmv.document_types import DOCUMENT_TYPES
from dmv.extraction.fields import CANONICAL_EXTRACTION_FIELDS, FIELD_DESCRIPTIONS


def build_extraction_prompt(
    *,
    document_type: str,
    document_name: str,
    source_filename: str,
) -> str:
    type_labels = {item.type_id: item.label for item in DOCUMENT_TYPES}
    type_label = type_labels.get(document_type, document_type)

    lines = [
        "You are extracting structured data from a scanned DMV-related document.",
        "",
        f"Document type: {document_type} ({type_label})",
        f"Document name: {document_name}",
        f"Source file: {source_filename}",
        "",
        "Extract every value you can find in the document.",
        "Use the canonical field names listed below whenever the data matches.",
        "Only include canonical fields that have a value in the document.",
        "Do not include fields with empty or unknown values — omit them entirely.",
        "Each canonical field must be an object with value (string) and confidence "
        "(number from 0 to 1).",
        "Put any additional values that do not map to a canonical field in the "
        "extra array as objects with field_name, value, and confidence. Omit extra "
        "when empty.",
        "",
        "Canonical fields:",
    ]
    for field in CANONICAL_EXTRACTION_FIELDS:
        description = FIELD_DESCRIPTIONS.get(field, field.replace("_", " "))
        lines.append(f"- {field}: {description}")

    lines.extend(
        [
            "",
            "Important:",
            "- Return dates in the format shown on the document when possible.",
            "- For names, split into first/last when clearly available; also fill "
            "driver_full_name or owner_full_name when appropriate.",
            "- For currency amounts, include digits and decimal point only unless "
            "the document shows a currency symbol.",
            "- For YES/NO or checkbox fields, use YES or NO when checked; omit when not applicable.",
            "- document_type and source_document_name must match the values above.",
            "- Never output empty strings for missing fields.",
            "- confidence is your certainty in the extracted value: 1.0 = clearly "
            "legible and unambiguous, ~0.7 = readable but slightly unclear, "
            "~0.4 = partially obscured or inferred, below 0.3 = very uncertain.",
        ]
    )
    return "\n".join(lines)
