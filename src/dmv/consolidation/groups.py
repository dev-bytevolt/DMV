from __future__ import annotations

from dataclasses import dataclass

from dmv.consolidation.field import (
    ConsolidatedField,
    collect_source_variants,
    compute_field_confidence,
    consolidate_field,
    needs_manual_review,
)
from dmv.consolidation.priority import OWNER_SOURCE_EXCLUDE_TYPES


@dataclass(frozen=True)
class AddressGroupSpec:
    """Nested address group whose parts must come from one source document."""

    nest_key: str  # e.g. "lienholder"
    address_key: str  # e.g. "address"
    # flat canonical field -> nested leaf name under address
    parts: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class EntityGroupSpec:
    """Entity nest that may contain scalar fields plus an address subgroup."""

    nest_key: str
    # flat canonical field -> nested leaf name (non-address)
    scalars: tuple[tuple[str, str], ...]
    address: AddressGroupSpec | None = None


# Coherent address groups: pick one trusted source, take all parts from it.
ENTITY_GROUPS: tuple[EntityGroupSpec, ...] = (
    EntityGroupSpec(
        nest_key="owner",
        scalars=(
            ("owner_full_name", "full_name"),
            ("owner_name", "name"),
            ("owner_phone", "phone"),
            ("owner_license_or_entity_id", "license_or_entity_id"),
            ("owner_county", "county"),
        ),
        address=AddressGroupSpec(
            nest_key="owner",
            address_key="address",
            parts=(
                ("owner_address_street", "street"),
                ("owner_address_city", "city"),
                ("owner_address_state", "state"),
                ("owner_address_zip", "zip"),
            ),
        ),
    ),
    EntityGroupSpec(
        nest_key="lienholder",
        scalars=(
            ("lienholder_name", "name"),
            ("lienholder_id", "id"),
            ("lienholder_phone", "phone"),
        ),
        address=AddressGroupSpec(
            nest_key="lienholder",
            address_key="address",
            parts=(
                ("lienholder_address_street", "street"),
                ("lienholder_address_city", "city"),
                ("lienholder_address_state", "state"),
                ("lienholder_address_zip", "zip"),
            ),
        ),
    ),
    EntityGroupSpec(
        nest_key="dealership",
        scalars=(
            ("dealership_name", "name"),
            ("dealership_entity_id", "entity_id"),
        ),
        address=AddressGroupSpec(
            nest_key="dealership",
            address_key="address",
            parts=(
                ("dealership_address_street", "street"),
                ("dealership_address_city", "city"),
                ("dealership_address_state", "state"),
                ("dealership_address_zip", "zip"),
            ),
        ),
    ),
    EntityGroupSpec(
        nest_key="lessee",
        scalars=(("lessee_name", "name"),),
        address=AddressGroupSpec(
            nest_key="lessee",
            address_key="address",
            parts=(
                ("lessee_address_street", "street"),
                ("lessee_address_city", "city"),
                ("lessee_address_state", "state"),
                ("lessee_address_zip", "zip"),
            ),
        ),
    ),
    EntityGroupSpec(
        nest_key="driver",
        scalars=(
            ("driver_first_name", "first_name"),
            ("driver_last_name", "last_name"),
            ("driver_middle_name", "middle_name"),
            ("driver_full_name", "full_name"),
            ("driver_license_number", "license_number"),
            ("driver_dob", "dob"),
            ("driver_gender", "gender"),
            ("driver_height", "height"),
            ("driver_eyes_color", "eyes_color"),
            ("driver_license_issue_date", "license_issue_date"),
            ("driver_license_expiration_date", "license_expiration_date"),
            ("driver_license_class", "license_class"),
            ("driver_endorsements", "endorsements"),
            ("driver_restrictions", "restrictions"),
        ),
        address=AddressGroupSpec(
            nest_key="driver",
            address_key="address",
            parts=(
                ("driver_address_street", "street"),
                ("driver_address_city", "city"),
                ("driver_address_state", "state"),
                ("driver_address_zip", "zip"),
            ),
        ),
    ),
    EntityGroupSpec(
        nest_key="representative",
        scalars=(
            ("representative_first_name", "first_name"),
            ("representative_last_name", "last_name"),
            ("representative_phone", "phone"),
        ),
        address=AddressGroupSpec(
            nest_key="representative",
            address_key="address",
            parts=(
                ("representative_address_street", "street"),
                ("representative_address_city", "city"),
                ("representative_address_state", "state"),
                ("representative_address_zip", "zip"),
            ),
        ),
    ),
)


def grouped_canonical_fields() -> frozenset[str]:
    fields: set[str] = set()
    for group in ENTITY_GROUPS:
        for flat, _ in group.scalars:
            fields.add(flat)
        if group.address is not None:
            for flat, _ in group.address.parts:
                fields.add(flat)
    return frozenset(fields)


def _entity_flat_fields(group: EntityGroupSpec) -> list[str]:
    fields = [flat for flat, _ in group.scalars]
    if group.address is not None:
        fields.extend(flat for flat, _ in group.address.parts)
    return fields


def _best_per_source(
    hypotheses: list[tuple[str, float, str, str]],
) -> dict[tuple[str, str], tuple[str, float]]:
    """Map (document_name, document_type) -> best (value, confidence)."""
    best: dict[tuple[str, str], tuple[str, float]] = {}
    for value, confidence, document_name, document_type in hypotheses:
        key = (document_name, document_type)
        existing = best.get(key)
        if existing is None or confidence > existing[1]:
            best[key] = (value, confidence)
    return best


def pick_coherent_entity_source(
    field_hypotheses: dict[str, list[tuple[str, float, str, str]]],
    flat_fields: list[str],
    *,
    address_fields: frozenset[str] | None = None,
    exclude_document_types: frozenset[str] | None = None,
    prefer_entity_names: bool = False,
) -> tuple[str, str] | None:
    """Choose the source document that best covers an entity field group."""
    address_fields = address_fields or frozenset()
    exclude_document_types = exclude_document_types or frozenset()
    per_source: dict[tuple[str, str], dict[str, tuple[str, float]]] = {}

    for flat in flat_fields:
        for source_key, value_conf in _best_per_source(field_hypotheses.get(flat, [])).items():
            _name, doc_type = source_key
            if doc_type in exclude_document_types:
                continue
            per_source.setdefault(source_key, {})[flat] = value_conf

    if not per_source:
        return None

    def _entity_name_boost(fields: dict[str, tuple[str, float]]) -> int:
        if not prefer_entity_names:
            return 0
        for key in ("owner_full_name", "owner_name"):
            if key not in fields:
                continue
            upper = fields[key][0].upper()
            if any(token in upper for token in ("TRUST", "LLC", "LTD", "INC", "LEASE", "TITLING")):
                return 1
        return 0

    def score(source_key: tuple[str, str]) -> tuple[int, int, int, int, float, float]:
        fields = per_source[source_key]
        _name, doc_type = source_key
        address_completeness = sum(1 for flat in fields if flat in address_fields)
        completeness = len(fields)
        total_conf = sum(conf for _, conf in fields.values())
        avg_conf = total_conf / max(completeness, 1)
        type_boost = 0
        if doc_type == "manufacturer_certificate":
            type_boost = 3
        elif doc_type == "dealer_invoice":
            type_boost = 2
        elif doc_type == "retail_certificate_of_sale":
            type_boost = 1
        # Prefer a complete address over a partial mix of scalars.
        return (
            address_completeness,
            _entity_name_boost(fields),
            type_boost,
            completeness,
            avg_conf,
            total_conf,
        )

    return max(per_source.keys(), key=score)


def _field_from_source(
    hypotheses: list[tuple[str, float, str, str]],
    winner: tuple[str, str],
) -> dict[str, object] | None:
    if not hypotheses:
        return None

    source_variants = collect_source_variants(hypotheses)
    by_source = _best_per_source(hypotheses)
    chosen = by_source.get(winner)
    if chosen is None:
        return None

    value, _confidence = chosen
    confidence = compute_field_confidence(value, source_variants)
    review_required = needs_manual_review(value, source_variants, confidence)
    return ConsolidatedField(
        value=value,
        variants=source_variants,
        confidence=confidence,
        review_required=review_required,
    ).to_dict()


def consolidate_entity_group(
    field_hypotheses: dict[str, list[tuple[str, float, str, str]]],
    group: EntityGroupSpec,
) -> dict[str, object] | None:
    """Build a nested entity object with coherent multi-part values.

    Address (and other combined) parts always come from one trusted source.
    Scalars prefer that same source when present; otherwise they fall back to
    normal field consolidation so split-document OCR can still merge.
    """
    flat_fields = _entity_flat_fields(group)
    if not any(field_hypotheses.get(flat) for flat in flat_fields):
        return None

    address_fields = (
        frozenset(flat for flat, _ in group.address.parts)
        if group.address is not None
        else frozenset()
    )
    winner = pick_coherent_entity_source(
        field_hypotheses,
        flat_fields,
        address_fields=address_fields,
        exclude_document_types=(
            OWNER_SOURCE_EXCLUDE_TYPES if group.nest_key == "owner" else frozenset()
        ),
        prefer_entity_names=group.nest_key == "owner",
    )

    nest: dict[str, object] = {}

    for flat, leaf in group.scalars:
        hypotheses = field_hypotheses.get(flat, [])
        if not hypotheses:
            continue
        field_dict: dict[str, object] | None = None
        if winner is not None and group.address is not None:
            field_dict = _field_from_source(hypotheses, winner)
        if field_dict is None:
            consolidated = consolidate_field(hypotheses, use_vin=False)
            field_dict = consolidated.to_dict() if consolidated else None
        if field_dict is not None:
            nest[leaf] = field_dict

    if group.address is not None and winner is not None:
        address: dict[str, object] = {}
        for flat, leaf in group.address.parts:
            hypotheses = field_hypotheses.get(flat, [])
            field_dict = _field_from_source(hypotheses, winner)
            if field_dict is not None:
                address[leaf] = field_dict
        if address:
            winner_name, winner_type = winner
            address["source_document_name"] = winner_name
            address["source_document_type"] = winner_type
            nest[group.address.address_key] = address

    if not nest:
        return None

    if winner is not None and group.address is not None:
        winner_name, winner_type = winner
        nest["source_document_name"] = winner_name
        nest["source_document_type"] = winner_type
    return nest


def count_consolidated_fields(payload: dict[str, object]) -> tuple[int, int]:
    """Return (total_fields, fields_with_review_required_false)."""
    total = 0
    ok = 0

    def walk(node: object) -> None:
        nonlocal total, ok
        if not isinstance(node, dict):
            return
        if "value" in node and "review_required" in node and "confidence" in node:
            total += 1
            if node.get("review_required") is False:
                ok += 1
            return
        for key, child in node.items():
            if key == "extra":
                continue
            walk(child)

    walk(payload)
    return total, ok
