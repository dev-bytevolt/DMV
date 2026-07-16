from __future__ import annotations

from dmv.document_types import DOCUMENT_TYPES
from dmv.extraction.profiles import (
    describe_field,
    field_guidance_for_document_type,
    fields_for_document_type,
)


def build_extraction_prompt(
    *,
    document_type: str,
    document_name: str,
    source_filename: str,
) -> str:
    type_labels = {item.type_id: item.label for item in DOCUMENT_TYPES}
    type_label = type_labels.get(document_type, document_type)
    allowed_fields = fields_for_document_type(document_type)
    guidance = field_guidance_for_document_type(document_type)
    uses_full_field_list = document_type not in {
        item.type_id for item in DOCUMENT_TYPES if item.type_id != "other"
    } or document_type == "other"
    # Unknown / other keep the full list; known types use narrowed profiles.
    uses_full_field_list = (
        document_type == "other"
        or document_type not in {
            "cover_letter",
            "universal_title_application",
            "vehicle_registration_application",
            "dealership_supplemental_title_form",
            "manufacturer_certificate",
            "retail_certificate_of_sale",
            "odometer_disclosure",
            "limited_power_of_attorney",
            "lease_agreement",
            "insurance_card",
            "driver_license",
            "dealer_invoice",
            "check_payment",
            "lessor_lender_paperwork",
        }
    )

    lines = [
        "You are extracting structured data from a scanned DMV-related document.",
        "",
        f"Document type: {document_type} ({type_label})",
        f"Document name: {document_name}",
        f"Source file: {source_filename}",
        "",
        "Extract every value you can find in the document that maps to the "
        "allowed fields below.",
        "Only include allowed fields that have a value in the document.",
        "Do not include fields with empty or unknown values — omit them entirely.",
        "Each allowed field must be an object with value (string) and confidence "
        "(number from 0 to 1).",
        "Put any additional values that do not map to an allowed field in the "
        "extra array as objects with field_name, value, and confidence. Omit extra "
        "when empty.",
        "",
    ]

    if uses_full_field_list:
        lines.append(
            "This document type is unrecognized or miscellaneous — search across "
            "the full canonical field list and use the most specific matching field."
        )
        lines.append("")
    else:
        lines.append(
            "Only the fields listed below are allowed for this document type. "
            "Do not invent or fill fields outside this list."
        )
        lines.append("")

    if guidance:
        lines.append("Field mapping rules for this document type:")
        for rule in guidance:
            lines.append(f"- {rule}")
        lines.append("")

    lines.append("Allowed fields:")
    for field in allowed_fields:
        lines.append(f"- {field}: {describe_field(field)}")

    lines.extend(
        [
            "",
            "Important:",
            "- Return dates in the format shown on the document when possible.",
            "- For names, split into first/last when clearly available; also fill "
            "the most specific full-name field allowed for this document type.",
            "- For currency amounts, include digits and decimal point only unless "
            "the document shows a currency symbol.",
            "- For YES/NO or checkbox fields, use YES or NO when checked; omit when not applicable.",
            "- document_type and source_document_name must match the values above.",
            "- Never output empty strings for missing fields.",
            "- Never output a field that is not in the allowed list above.",
            "- confidence is your certainty in the extracted value: 1.0 = clearly "
            "legible and unambiguous, ~0.7 = readable but slightly unclear, "
            "~0.4 = partially obscured or inferred, below 0.3 = very uncertain.",
        ]
    )
    return "\n".join(lines)
