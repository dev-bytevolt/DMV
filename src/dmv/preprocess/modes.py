from __future__ import annotations

from enum import Enum


class PreprocessMode(str, Enum):
    FULL_PAGE_FORM = "full_page_form"
    EMBEDDED_CARD = "embedded_card"
    DEFAULT = "default"


FULL_PAGE_FORM_TYPES = frozenset(
    {
        "cover_letter",
        "universal_title_application",
        "vehicle_registration_application",
        "dealership_supplemental_title_form",
        "manufacturer_certificate",
        "retail_certificate_of_sale",
        "odometer_disclosure",
        "limited_power_of_attorney",
        "lease_agreement",
        "dealer_invoice",
        "check_payment",
        "lessor_lender_paperwork",
    }
)

EMBEDDED_CARD_TYPES = frozenset(
    {
        "driver_license",
    }
)


def preprocess_mode_for_document_type(document_type: str) -> PreprocessMode:
    if document_type in EMBEDDED_CARD_TYPES:
        return PreprocessMode.EMBEDDED_CARD
    if document_type in FULL_PAGE_FORM_TYPES:
        return PreprocessMode.FULL_PAGE_FORM
    return PreprocessMode.DEFAULT
