from __future__ import annotations

import re
from typing import Protocol

PRIMARY_SOURCE_DOCUMENT_TYPE = "manufacturer_certificate"

# Vehicle identity fields where Certificate of Origin is authoritative.
# Body type / color / weight / odometer are often wrong or absent on the MCO
# (trim mislabeled as body, no color), so those stay on normal consensus.
PRIMARY_SOURCE_FIELDS: frozenset[str] = frozenset(
    {
        "vehicle_vin",
        "vehicle_year",
        "vehicle_make",
        "vehicle_model",
        "vehicle_fuel_type",
        "certificate_number",
        "manufacturer_name",
    }
)

REVIEW_DISAGREEMENT_THRESHOLD = 0.15
_VIN_NORMALIZE_RE = re.compile(r"[^A-Z0-9]")


class VariantLike(Protocol):
    value: str
    confidence: float
    source_document_name: str
    source_document_type: str


def primary_document_type_for_field(field_name: str) -> str | None:
    if field_name in PRIMARY_SOURCE_FIELDS:
        return PRIMARY_SOURCE_DOCUMENT_TYPE
    if field_name == "vehicle_body_type":
        return "odometer_disclosure"
    if field_name in {"purchase_price", "gross_sales_lease_price"}:
        return "dealer_invoice"
    if field_name in {"dealership_name", "dealership_entity_id"}:
        return "manufacturer_certificate"
    return None


# Document types that must not win owner-of-record consolidation.
OWNER_SOURCE_EXCLUDE_TYPES: frozenset[str] = frozenset(
    {
        "driver_license",
        "insurance_card",
        "other",
        "check_payment",
    }
)


def _normalize_for_compare(value: str) -> str:
    value = value.strip()
    value = " ".join(value.split())
    return value.upper()


def _levenshtein(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j - 1] + cost,
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
            )
    return dp[m][n]


def _distance_ratio(left: str, right: str) -> float:
    if not left and not right:
        return 0.0
    return _levenshtein(left, right) / max(len(left), len(right), 1)


def _normalize_vin(value: str) -> str:
    return _VIN_NORMALIZE_RE.sub("", value.upper())


def normalize_vin(raw: str | None) -> str:
    if not raw:
        return ""
    return _normalize_vin(raw)


def variant_confirms_value(
    primary_value: str,
    variant: VariantLike,
    *,
    use_vin: bool = False,
) -> bool:
    if use_vin:
        return _normalize_vin(primary_value) == _normalize_vin(variant.value)

    primary_norm = _normalize_for_compare(primary_value)
    variant_norm = _normalize_for_compare(variant.value)
    if primary_norm == variant_norm:
        return True
    return _distance_ratio(variant_norm, primary_norm) <= REVIEW_DISAGREEMENT_THRESHOLD


def has_confirmation(
    primary_value: str,
    other_variants: list[VariantLike],
    *,
    use_vin: bool = False,
) -> bool:
    """True when no other sources exist or at least one corroborates the primary."""
    if not other_variants:
        return True
    return any(
        variant_confirms_value(primary_value, variant, use_vin=use_vin)
        for variant in other_variants
    )


def pick_primary_source_variant(
    source_variants: list[VariantLike],
    *,
    primary_document_type: str,
) -> VariantLike | None:
    primary_variants = [
        variant
        for variant in source_variants
        if variant.source_document_type == primary_document_type
    ]
    if not primary_variants:
        return None
    return max(primary_variants, key=lambda variant: variant.confidence)
