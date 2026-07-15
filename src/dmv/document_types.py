from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentTypeDefinition:
    type_id: str
    label: str
    aliases: tuple[str, ...]


DOCUMENT_TYPES: tuple[DocumentTypeDefinition, ...] = (
    DocumentTypeDefinition(
        "cover_letter",
        "Cover letter to NJ DMV or MVC",
        ("cover letter", "MVC cover letter", "DMV cover letter"),
    ),
    DocumentTypeDefinition(
        "universal_title_application",
        "Universal Title Application",
        ("title application", "universal title"),
    ),
    DocumentTypeDefinition(
        "vehicle_registration_application",
        "Application for Vehicle Registration",
        ("vehicle registration", "registration application"),
    ),
    DocumentTypeDefinition(
        "dealership_supplemental_title_form",
        "New Car Dealership Supplemental Passenger Vehicle Title Form / Luxury and Fuel Inefficient Surcharge Calculation",
        ("dealership supplemental", "luxury surcharge", "fuel inefficient surcharge"),
    ),
    DocumentTypeDefinition(
        "manufacturer_certificate",
        "Manufacturer Certificate / Certificate of Origin",
        ("certificate of origin", "MSO", "manufacturer certificate"),
    ),
    DocumentTypeDefinition(
        "retail_certificate_of_sale",
        "Retail Certificate of Sale Receipt",
        ("certificate of sale", "retail certificate"),
    ),
    DocumentTypeDefinition(
        "odometer_disclosure",
        "Odometer and damage disclosure statement",
        ("odometer disclosure", "damage disclosure"),
    ),
    DocumentTypeDefinition(
        "limited_power_of_attorney",
        "Limited power of attorney documents",
        ("power of attorney", "POA", "limited POA"),
    ),
    DocumentTypeDefinition(
        "lease_agreement",
        "Lease agreement pages",
        ("lease agreement", "lease contract"),
    ),
    DocumentTypeDefinition(
        "insurance_card",
        "Insurance cards or temporary insurance evidence",
        ("insurance card", "insurance evidence", "proof of insurance"),
    ),
    DocumentTypeDefinition(
        "driver_license",
        "Driver license copy",
        ("driver license", "driving license", "DL copy"),
    ),
    DocumentTypeDefinition(
        "dealer_invoice",
        "Dealer invoice / bill of sale",
        ("dealer invoice", "bill of sale"),
    ),
    DocumentTypeDefinition(
        "check_payment",
        "Check or payment stub",
        ("check", "payment stub", "payment receipt"),
    ),
    DocumentTypeDefinition(
        "lessor_lender_paperwork",
        "Lessor or lender paperwork",
        ("lessor paperwork", "lender paperwork", "lienholder documents"),
    ),
    DocumentTypeDefinition(
        "other",
        "Other document not matching known types",
        ("unknown", "miscellaneous", "other document"),
    ),
)

DOCUMENT_TYPE_IDS: tuple[str, ...] = tuple(dt.type_id for dt in DOCUMENT_TYPES)


def build_classification_prompt() -> str:
    lines = [
        "You are classifying scanned pages from a PDF that may contain multiple "
        "DMV-related forms and documents.",
        "",
        "For each distinct document or form found, identify:",
        "- id: a short unique identifier like doc-001, doc-002, ...",
        "- name: a human-readable name for the document",
        "- type: one of the predefined type IDs listed below",
        "- pages: 1-based page numbers from the uploaded PDF that belong to this document",
        "",
        "Also list any pages that are blank, separator pages, or contain no meaningful "
        "document content in empty_pages.",
        "",
        "IMPORTANT — blank and separator pages are NOT documents:",
        "- Never create a document entry for blank, separator, divider, or empty pages.",
        "- Do not create one document per blank page and do not group blank pages into "
        "a document entry.",
        "- Every blank/separator page must appear only in empty_pages.",
        "- Use type=other only for pages that contain real but unrecognized document "
        "content, never for blank pages.",
        "",
        "Page grouping rules:",
        "- The PDF pages are in sequential scan order. Pages from different documents "
        "never interleave with each other.",
        "- If two instances of the same document type appear with other documents' pages "
        "in between, they are separate documents. Example: Limited Power of Attorney "
        "on page 15 and another Limited Power of Attorney on page 25, with other forms "
        "on pages 16-24, must be two document entries (not one).",
        "- A single multi-page document may have blank or separator pages between its "
        "content pages. Put those blank pages in empty_pages, not in the document's "
        "pages list. Example: a lease agreement whose content is on pages 27, 29, 31, "
        "33, 35, 37 with blank pages 26, 28, 30, 32, 34, 36 between them is ONE "
        "document with pages [27, 29, 31, 33, 35, 37] and those blank pages listed "
        "in empty_pages.",
        "- Do not combine pages into one document if any page between its lowest and "
        "highest page number belongs to a different document.",
        "",
        "Every page in the PDF must appear exactly once: either in a document's pages "
        "list or in empty_pages.",
        "",
        "Predefined document types (type_id — label; aliases):",
    ]
    for doc_type in DOCUMENT_TYPES:
        alias_text = ", ".join(doc_type.aliases)
        lines.append(f"- {doc_type.type_id} — {doc_type.label}; aliases: {alias_text}")
    return "\n".join(lines)
