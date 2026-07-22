from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from dmv.consolidation.priority import (
    has_confirmation,
    normalize_vin,
    pick_primary_source_variant,
    primary_document_type_for_field,
)
from dmv.consolidation.rover import rover_consensus, vin_consensus

REVIEW_CONFIDENCE_THRESHOLD = 0.75
# Near-miss OCR distance. Variants farther than this are treated as outliers
# and do not drag confidence / force review by themselves.
REVIEW_DISAGREEMENT_THRESHOLD = 0.15


@dataclass(frozen=True)
class FieldVariant:
    value: str
    confidence: float
    source_document_name: str = ""
    source_document_type: str = ""


@dataclass(frozen=True)
class ConsolidatedField:
    value: str
    variants: list[FieldVariant]
    confidence: float
    review_required: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "value": self.value,
            "variants": [
                {
                    "value": variant.value,
                    "confidence": variant.confidence,
                    "source_document_name": variant.source_document_name,
                    "source_document_type": variant.source_document_type,
                }
                for variant in self.variants
            ],
            "confidence": round(self.confidence, 4),
            "review_required": self.review_required,
        }


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


def collect_variants(hypotheses: list[tuple[str, float]]) -> list[FieldVariant]:
    """Deduplicate hypotheses, keeping the highest confidence per unique value."""
    best_by_normalized: dict[str, FieldVariant] = {}
    for value, confidence in hypotheses:
        stripped = value.strip()
        if not stripped:
            continue
        key = _normalize_for_compare(stripped)
        existing = best_by_normalized.get(key)
        if existing is None or confidence > existing.confidence:
            best_by_normalized[key] = FieldVariant(value=stripped, confidence=confidence)
    return sorted(
        best_by_normalized.values(),
        key=lambda variant: (-variant.confidence, variant.value),
    )


def collect_source_variants(
    hypotheses: list[tuple[str, float, str, str]],
) -> list[FieldVariant]:
    variants: list[FieldVariant] = []
    for value, confidence, source_document_name, source_document_type in hypotheses:
        stripped = value.strip()
        if not stripped:
            continue
        variants.append(
            FieldVariant(
                value=stripped,
                confidence=confidence,
                source_document_name=source_document_name,
                source_document_type=source_document_type,
            )
        )
    return sorted(
        variants,
        key=lambda variant: (
            -variant.confidence,
            variant.source_document_name,
            variant.value,
        ),
    )


def _partition_variants(
    consensus: str,
    source_variants: list[FieldVariant],
) -> tuple[list[FieldVariant], list[tuple[FieldVariant, float]], list[FieldVariant]]:
    """Split sources into exact matches, near-miss OCR variants, and distant outliers."""
    consensus_norm = _normalize_for_compare(consensus)
    exact: list[FieldVariant] = []
    near: list[tuple[FieldVariant, float]] = []
    distant: list[FieldVariant] = []
    for variant in source_variants:
        variant_norm = _normalize_for_compare(variant.value)
        distance = _distance_ratio(variant_norm, consensus_norm)
        if variant_norm == consensus_norm:
            exact.append(variant)
        elif distance <= REVIEW_DISAGREEMENT_THRESHOLD:
            near.append((variant, distance))
        else:
            distant.append(variant)
    return exact, near, distant


def _plurality_strength(
    consensus: str,
    source_variants: list[FieldVariant],
) -> tuple[int, int, float]:
    """Return (exact_count, second_place_count, strength in (0, 1])."""
    consensus_norm = _normalize_for_compare(consensus)
    counts = Counter(
        _normalize_for_compare(variant.value) for variant in source_variants
    )
    exact_count = counts.get(consensus_norm, 0)
    other_counts = [count for key, count in counts.items() if key != consensus_norm]
    second_count = max(other_counts) if other_counts else 0
    strength = exact_count / max(exact_count + second_count, 1)
    return exact_count, second_count, strength


def compute_field_confidence(
    consensus: str,
    source_variants: list[FieldVariant],
    *,
    unique_variants: list[FieldVariant] | None = None,
) -> float:
    del unique_variants  # kept for call-site compatibility
    if not source_variants:
        return 0.0
    if len(source_variants) == 1:
        return source_variants[0].confidence

    exact, near, _distant = _partition_variants(consensus, source_variants)
    if not exact:
        # Consensus is synthesized / only near-misses support it.
        if not near:
            return 0.0
        peak = max(variant.confidence for variant, _ in near)
        return max(0.0, min(1.0, peak * 0.6))

    peak = max(variant.confidence for variant in exact)
    mean_exact = sum(variant.confidence for variant in exact) / len(exact)
    base = 0.65 * peak + 0.35 * mean_exact

    exact_count, second_count, plurality_strength = _plurality_strength(
        consensus,
        source_variants,
    )

    # Soft OCR penalty only from near-miss variants — distant outliers are ignored.
    near_penalty = 0.0
    if near:
        total_weight = sum(variant.confidence for variant in source_variants)
        if total_weight > 0:
            near_weight = sum(variant.confidence for variant, _ in near)
            avg_near_distance = sum(distance for _, distance in near) / len(near)
            near_penalty = 0.35 * avg_near_distance * (near_weight / total_weight)

    confidence = base * (0.85 + 0.15 * plurality_strength) * (1.0 - near_penalty)

    # Reward clear repeated agreement.
    if exact_count >= 2:
        confidence = min(1.0, confidence * 1.04)
    if exact_count > second_count and exact_count >= 2:
        confidence = min(1.0, confidence * 1.03)

    return max(0.0, min(1.0, confidence))


def needs_manual_review(
    consensus: str,
    source_variants: list[FieldVariant],
    confidence: float,
    *,
    unique_variants: list[FieldVariant] | None = None,
) -> bool:
    del unique_variants  # kept for call-site compatibility
    if not consensus or not source_variants:
        return True
    if confidence < REVIEW_CONFIDENCE_THRESHOLD:
        return True

    exact, _near, _distant = _partition_variants(consensus, source_variants)
    # Synthesized consensus that does not exactly match any source.
    if not exact:
        return True

    exact_count, second_count, _strength = _plurality_strength(
        consensus,
        source_variants,
    )
    # Require review when the winner is not a clear plurality over other values.
    if exact_count <= second_count:
        return True

    return False


def consolidate_field(
    hypotheses: list[tuple[str, float, str, str]],
    *,
    use_vin: bool = False,
    field_name: str | None = None,
) -> ConsolidatedField | None:
    source_variants = collect_source_variants(hypotheses)
    deduped_variants = collect_variants([(v, c) for v, c, _, _ in hypotheses])
    if not source_variants or not deduped_variants:
        return None

    primary_document_type = (
        primary_document_type_for_field(field_name) if field_name else None
    )
    primary_variant = (
        pick_primary_source_variant(
            source_variants,
            primary_document_type=primary_document_type,
        )
        if primary_document_type
        else None
    )

    if primary_variant is not None:
        value = primary_variant.value
        if use_vin:
            value = normalize_vin(value) or value
        other_variants = [
            variant
            for variant in source_variants
            if variant.source_document_type != primary_document_type
        ]
        confirmed = has_confirmation(
            value,
            other_variants,
            use_vin=use_vin,
        )
        confidence = compute_field_confidence(
            value,
            source_variants,
            unique_variants=deduped_variants,
        )
        if confirmed:
            review_required = False
        else:
            review_required = True
        return ConsolidatedField(
            value=value,
            variants=source_variants,
            confidence=confidence,
            review_required=review_required,
        )

    vote_hypotheses = [(v.value, v.confidence) for v in source_variants]

    if use_vin:
        value = vin_consensus(vote_hypotheses)
    else:
        value = rover_consensus(
            [(variant.value, variant.confidence) for variant in deduped_variants]
        )

    if not value:
        return None

    confidence = compute_field_confidence(
        value,
        source_variants,
        unique_variants=deduped_variants,
    )
    review_required = needs_manual_review(
        value,
        source_variants,
        confidence,
        unique_variants=deduped_variants,
    )
    return ConsolidatedField(
        value=value,
        variants=source_variants,
        confidence=confidence,
        review_required=review_required,
    )
