from __future__ import annotations

from dmv.extraction.fields import CANONICAL_EXTRACTION_FIELDS, FIELD_DESCRIPTIONS

# Shared field groups reused across document-type profiles.
_VEHICLE_CORE: tuple[str, ...] = (
    "vehicle_vin",
    "vehicle_year",
    "vehicle_make",
    "vehicle_model",
    "vehicle_fuel_type",
    "vehicle_body_type",
    "vehicle_weight",
    "vehicle_color",
    "odometer_reading",
    "odometer_not_actual",
    "odometer_exceeded_mechanical",
)

_OWNER: tuple[str, ...] = (
    "owner_full_name",
    "owner_name",
    "owner_phone",
    "owner_license_or_entity_id",
    "owner_address_street",
    "owner_address_city",
    "owner_address_state",
    "owner_address_zip",
    "owner_county",
    "co_owner_first_name",
    "co_owner_last_name",
    "co_owner_name",
    "co_owner_license_number",
)

_LESSEE: tuple[str, ...] = (
    "lessee_name",
    "lessee_address_street",
    "lessee_address_city",
    "lessee_address_state",
    "lessee_address_zip",
)

_LIENHOLDER: tuple[str, ...] = (
    "lien_holder",
    "lienholder_name",
    "lienholder_id",
    "lienholder_phone",
    "lienholder_address_street",
    "lienholder_address_city",
    "lienholder_address_state",
    "lienholder_address_zip",
)

_DEALERSHIP: tuple[str, ...] = (
    "dealership_name",
    "dealership_entity_id",
    "dealership_address_street",
    "dealership_address_city",
    "dealership_address_state",
    "dealership_address_zip",
    "dealer_name",
)

_DRIVER: tuple[str, ...] = (
    "driver_first_name",
    "driver_last_name",
    "driver_middle_name",
    "driver_full_name",
    "driver_address_street",
    "driver_address_city",
    "driver_address_state",
    "driver_address_zip",
    "driver_license_number",
    "driver_dob",
    "driver_gender",
    "driver_height",
    "driver_eyes_color",
    "driver_license_issue_date",
    "driver_license_expiration_date",
    "driver_license_class",
    "driver_endorsements",
    "driver_restrictions",
)

_INSURANCE: tuple[str, ...] = (
    "insured_name",
    "insurance_company",
    "insurance_policy_number",
    "policy_effective_date",
    "policy_expiration_date",
)


def _fields(*groups: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for group in groups:
        for field in group:
            if field in seen:
                continue
            if field not in CANONICAL_EXTRACTION_FIELDS:
                continue
            seen.add(field)
            ordered.append(field)
    return tuple(ordered)


# Document types without an entry (or "other") use the full canonical list.
EXTRACTION_FIELDS_BY_DOCUMENT_TYPE: dict[str, tuple[str, ...]] = {
    "cover_letter": _fields(
        (
            "cover_date",
            "customer_name",
            "vehicle_vin",
            "lien_holder",
            "plate_type",
            "vehicle_color",
            "purchase_price",
            "sales_tax",
            "lfis_amount",
            "collect_taxes",
            "collect_lfis",
            "document_date",
            "phone_number",
            "email_address",
        ),
        _VEHICLE_CORE[:4],
    ),
    "universal_title_application": _fields(
        _VEHICLE_CORE,
        _OWNER,
        _LIENHOLDER,
        (
            "representative_first_name",
            "representative_last_name",
            "representative_phone",
            "representative_address_street",
            "representative_address_city",
            "representative_address_state",
            "representative_address_zip",
            "plate_type",
            "plate_number",
            "document_date",
        ),
    ),
    "vehicle_registration_application": _fields(
        _VEHICLE_CORE,
        _OWNER,
        _LESSEE,
        _LIENHOLDER,
        _INSURANCE,
        (
            "plate_number",
            "plate_prefix",
            "plate_type",
            "registration_code",
            "vehicle_weight_or_passengers",
            "lease_signed_date",
            "lease_term_months",
            "lease_cancelled_date",
            "document_date",
        ),
    ),
    "dealership_supplemental_title_form": _fields(
        _VEHICLE_CORE,
        _DEALERSHIP,
        _OWNER,
        (
            "vehicle_epa_mpg_rating",
            "gross_sales_lease_price",
            "surcharge_amount",
            "lfis_amount",
            "purchase_price",
            "sales_tax",
            "document_date",
        ),
    ),
    "manufacturer_certificate": _fields(
        _VEHICLE_CORE,
        _DEALERSHIP,
        (
            "certificate_number",
            "manufacturer_name",
            "seller_name",
            "buyer_name",
            "vehicle_epa_mpg_rating",
            "document_date",
        ),
    ),
    "retail_certificate_of_sale": _fields(
        _VEHICLE_CORE,
        _OWNER,
        _DEALERSHIP,
        _LIENHOLDER,
        (
            "buyer_name",
            "seller_name",
            "sale_date",
            "purchase_price",
            "gross_sales_lease_price",
            "sales_tax",
            "sales_tax_amount",
            "sales_tax_paid",
            "sales_tax_exemption_code",
            "lfis_amount",
            "surcharge_amount",
            "plate_type",
            "plate_number",
            "certificate_number",
            "vehicle_epa_mpg_rating",
            "document_date",
        ),
    ),
    "odometer_disclosure": _fields(
        _VEHICLE_CORE,
        (
            "seller_name",
            "buyer_name",
            "sale_date",
            "vehicle_epa_mpg_rating",
            "document_date",
        ),
    ),
    "limited_power_of_attorney": _fields(
        (
            "vehicle_vin",
            "poa_grantor_name",
            "poa_agent_name",
            "poa_date",
            "document_date",
            "entity_name",
        ),
        _VEHICLE_CORE[:4],
    ),
    "lease_agreement": _fields(
        _VEHICLE_CORE,
        _LESSEE,
        _DEALERSHIP,
        (
            "lessor_name",
            "owner_phone",
            "owner_license_or_entity_id",
            "lienholder_name",
            "lien_holder",
            "monthly_payment",
            "lease_start_date",
            "lease_end_date",
            "lease_signed_date",
            "lease_term_months",
            "purchase_price",
            "gross_sales_lease_price",
            "sales_tax",
            "sales_tax_amount",
            "sales_tax_paid",
            "lfis_amount",
            "surcharge_amount",
            "vehicle_epa_mpg_rating",
            "document_date",
        ),
    ),
    "insurance_card": _fields(
        (
            "vehicle_vin",
            "vehicle_year",
            "vehicle_make",
            "vehicle_model",
        ),
        _INSURANCE,
        (
            "document_date",
        ),
    ),
    "driver_license": _fields(_DRIVER),
    "dealer_invoice": _fields(
        _VEHICLE_CORE,
        _OWNER,
        _LESSEE,
        _DEALERSHIP,
        _LIENHOLDER,
        (
            "buyer_name",
            "seller_name",
            "sale_date",
            "purchase_price",
            "sales_tax_amount",
            "sales_tax",
            "sales_tax_paid",
            "sales_tax_exemption_code",
            "lfis_amount",
            "surcharge_amount",
            "gross_sales_lease_price",
            "plate_type",
            "vehicle_epa_mpg_rating",
            "document_date",
        ),
    ),
    "check_payment": _fields(
        (
            "payee_name",
            "check_amount",
            "check_date",
            "check_number",
            "check_memo",
            "document_date",
            "vehicle_vin",
            "customer_name",
        ),
    ),
    "lessor_lender_paperwork": _fields(
        _VEHICLE_CORE,
        _LESSEE,
        _LIENHOLDER,
        (
            "lessor_name",
            "owner_phone",
            "owner_license_or_entity_id",
            "entity_name",
            "document_date",
            "phone_number",
            "vehicle_epa_mpg_rating",
        ),
    ),
    # "other" intentionally omitted — falls back to full canonical list.
}

DOCUMENT_TYPE_FIELD_GUIDANCE: dict[str, tuple[str, ...]] = {
    "driver_license": (
        "Extract only the license holder's identity and license attributes into driver_* fields.",
        "Do not use owner_*, lessee_*, buyer_name, or insured_name on a driver license.",
    ),
    "insurance_card": (
        "Put the named insured in insured_name — never owner_full_name or driver_full_name.",
        "Put the insurer in insurance_company and the policy number in insurance_policy_number.",
        "Do not populate owner_*, lessee_*, or driver_* fields.",
    ),
    "lease_agreement": (
        "Put the person or entity leasing the vehicle in lessee_name / lessee_address_*.",
        "Put the leasing company in lessor_name (or lienholder_name if labeled as lienholder).",
        "Extract lessor_name, purchase_price (capitalized cost / amount subject to tax), "
        "lfis_amount/surcharge_amount, and gross_sales_lease_price when shown.",
        "Extract owner_phone / owner_license_or_entity_id for the titled owner/lessor when present.",
        "vehicle_body_type is body style only; vehicle_model is model name without unnecessary trim.",
        "Do not put the lessee into owner_* name/address fields. Do not use driver_* or insured_name.",
    ),
    "retail_certificate_of_sale": (
        "owner_* is the titled owner of record (often a lease titling trust on lease deals).",
        "buyer_name is the purchaser named on the certificate when distinct from owner.",
        "plate_type is NEW PLATES / plate-request wording when present — not the plate number.",
        "sales_tax should include TAX SATISFIED / PAID wording when shown.",
        "vehicle_body_type is body style only; vehicle_color prefers full color name.",
        "Do not use lessee_* or driver_* unless those labels appear explicitly.",
    ),
    "dealer_invoice": (
        "On cash/retail sales put the purchaser in owner_* (name and address) and buyer_name.",
        "On lease deals, put the lease trust/titling owner in owner_* and the customer in lessee_*.",
        "purchase_price is the vehicle selling/retail price line (e.g. NEW CAR RETAIL), "
        "preferring the amount sales tax was computed on when identifiable — not total cash with fees.",
        "gross_sales_lease_price is the LFIS/surcharge basis (NEW CAR RETAIL / vehicle price before add-ons).",
        "lfis_amount / surcharge_amount is the luxury/fuel inefficient surcharge (often 0.4% of gross).",
        "vehicle_body_type is body style only (never trim like XLE); vehicle_color prefers full name.",
        "vehicle_epa_mpg_rating is combined MPG only (10–80) — never vehicle weight.",
        "dealership_entity_id is the numeric MVC/facility number, not state codes like NY704.",
        "plate_type is NEW PLATES / plate-request wording when present — not the plate number.",
        "Do not use driver_* or insured_name.",
    ),
    "manufacturer_certificate": (
        "Extract vehicle and manufacturer/dealership data only.",
        "vehicle_body_type is body style (SEDAN, SUV, etc.), not series/trim.",
        "vehicle_model is the model name without unnecessary trim unless trim is part of the model designation.",
        "Do not invent a person owner; use buyer_name/seller_name only if printed.",
        "Extract vehicle_epa_mpg_rating when present. Do not use driver_*, lessee_*, or insured_name.",
    ),
    "limited_power_of_attorney": (
        "Put the person granting authority in poa_grantor_name and the agent in poa_agent_name.",
        "Do not map grantor/agent into owner_* or driver_* fields.",
    ),
    "check_payment": (
        "Extract payee, amount, date, number, and memo only.",
        "Do not populate owner_*, driver_*, or lessee_* from a check.",
    ),
    "cover_letter": (
        "Use customer_name for the customer named on the cover letter.",
        "Do not invent owner/lessee/driver identity fields unless clearly labeled.",
    ),
    "odometer_disclosure": (
        "Focus on odometer_* and seller/buyer names when present.",
        "vehicle_body_type is body style only; vehicle_model is model name without unnecessary trim.",
        "Do not populate driver_* or insured_name.",
    ),
    "lessor_lender_paperwork": (
        "Put the lessor/lender in lessor_name or lienholder_*; put the customer in lessee_name.",
        "Extract owner_phone / owner_license_or_entity_id for the titled owner/lessor when present.",
        "Do not use owner_* name/address fields for the lessee/customer.",
    ),
    "universal_title_application": (
        "owner_* is the applicant/owner on the title application.",
        "Use lienholder_* for any lienholder section; do not use insured_name or driver_*.",
    ),
    "vehicle_registration_application": (
        "Use owner_* for registered owner and lessee_* when a lease section is present.",
        "Do not use driver_* or insured_name unless those labels appear.",
    ),
    "dealership_supplemental_title_form": (
        "Extract dealership, vehicle, price, and surcharge fields.",
        "Use owner_* only when an owner section is clearly labeled.",
    ),
}


def fields_for_document_type(document_type: str) -> tuple[str, ...]:
    """Return allowed canonical fields for a document type.

    Unknown types and ``other`` keep the full canonical field list so extraction
    can still search broadly when the document is unrecognized.
    """
    if document_type in EXTRACTION_FIELDS_BY_DOCUMENT_TYPE:
        return EXTRACTION_FIELDS_BY_DOCUMENT_TYPE[document_type]
    return CANONICAL_EXTRACTION_FIELDS


def field_guidance_for_document_type(document_type: str) -> tuple[str, ...]:
    return DOCUMENT_TYPE_FIELD_GUIDANCE.get(document_type, ())


def describe_field(field: str) -> str:
    return FIELD_DESCRIPTIONS.get(field, field.replace("_", " "))
