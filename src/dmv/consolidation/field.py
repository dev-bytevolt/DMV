from __future__ import annotations

from dataclasses import dataclass

from dmv.consolidation.rover import rover_consensus, vin_consensus

REVIEW_CONFIDENCE_THRESHOLD = 0.75
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


def compute_field_confidence(
    consensus: str,
    source_variants: list[FieldVariant],
    *,
    unique_variants: list[FieldVariant] | None = None,
) -> float:
    if not source_variants:
        return 0.0
    if len(source_variants) == 1:
        return source_variants[0].confidence

    unique = unique_variants or collect_variants(
        [(variant.value, variant.confidence) for variant in source_variants]
    )
    consensus_norm = _normalize_for_compare(consensus)
    total_weight = sum(variant.confidence for variant in source_variants)
    if total_weight <= 0:
        return 0.0

    matching_weight = 0.0
    agreeing_confidences: list[float] = []
    for variant in source_variants:
        variant_norm = _normalize_for_compare(variant.value)
        distance = _distance_ratio(variant_norm, consensus_norm)
        if distance <= REVIEW_DISAGREEMENT_THRESHOLD:
            matching_weight += variant.confidence
            agreeing_confidences.append(variant.confidence)

    distances = [
        _distance_ratio(_normalize_for_compare(variant.value), consensus_norm)
        for variant in unique
    ]
    support_ratio = matching_weight / total_weight
    peak_agreeing = max(agreeing_confidences) if agreeing_confidences else 0.0
    avg_disagreement = sum(distances) / len(distances)

    # Weighted agreement among sources, discounted by how far unique variants
    # drift from the chosen consensus.
    confidence = peak_agreeing * support_ratio * (1.0 - 0.6 * avg_disagreement)

    if any(
        _normalize_for_compare(variant.value) == consensus_norm
        for variant in source_variants
    ):
        confidence = min(1.0, confidence * 1.05)

    return max(0.0, min(1.0, confidence))


def needs_manual_review(
    consensus: str,
    source_variants: list[FieldVariant],
    confidence: float,
    *,
    unique_variants: list[FieldVariant] | None = None,
) -> bool:
    if not consensus or not source_variants:
        return True
    if confidence < REVIEW_CONFIDENCE_THRESHOLD:
        return True

    unique = unique_variants or collect_variants(
        [(variant.value, variant.confidence) for variant in source_variants]
    )
    consensus_norm = _normalize_for_compare(consensus)
    distinct_distances = [
        _distance_ratio(_normalize_for_compare(variant.value), consensus_norm)
        for variant in unique
    ]
    if len(unique) > 1 and max(distinct_distances) > REVIEW_DISAGREEMENT_THRESHOLD:
        return True

    # Synthesized value that does not exactly match any source variant.
    if len(unique) > 1 and not any(
        _normalize_for_compare(variant.value) == consensus_norm
        for variant in source_variants
    ):
        return True

    return False


def consolidate_field(
    hypotheses: list[tuple[str, float, str, str]],
    *,
    use_vin: bool = False,
) -> ConsolidatedField | None:
    source_variants = collect_source_variants(hypotheses)
    deduped_variants = collect_variants([(v, c) for v, c, _, _ in hypotheses])
    if not source_variants or not deduped_variants:
        return None

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
