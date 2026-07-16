from __future__ import annotations

from dataclasses import dataclass

from dmv.models.classification import ClassifiedDocument, ClassificationResult

# Forms present in test fixtures but produced as pipeline output, not real input.
DEBUG_OUTPUT_DOCUMENT_TYPES: frozenset[str] = frozenset(
    {
        "cover_letter",
        "universal_title_application",
        "vehicle_registration_application",
        "dealership_supplemental_title_form",
    }
)

DEBUG_OUTPUT_EXCLUSION_REASONS: dict[str, str] = {
    "cover_letter": (
        "test fixture — MV Express Cover Letter to NJ DMV is pipeline output, "
        "not real input"
    ),
    "universal_title_application": (
        "test fixture — NJMVC Universal Title Application is pipeline output, "
        "not real input"
    ),
    "vehicle_registration_application": (
        "test fixture — NJMVC Application for Vehicle Registration is pipeline "
        "output, not real input"
    ),
    "dealership_supplemental_title_form": (
        "test fixture — NJMVC New Car Dealership Supplemental Passenger Vehicle "
        "Title Form is pipeline output, not real input"
    ),
}


@dataclass(frozen=True)
class ExcludedDocument:
    id: str
    name: str
    type: str
    pages: list[int]
    reason: str


def exclusion_reason_for_type(document_type: str) -> str | None:
    if document_type not in DEBUG_OUTPUT_DOCUMENT_TYPES:
        return None
    return DEBUG_OUTPUT_EXCLUSION_REASONS.get(
        document_type,
        "test fixture — pipeline output form, not real input",
    )


def identify_debug_exclusions(
    classification: ClassificationResult,
) -> list[ExcludedDocument]:
    excluded: list[ExcludedDocument] = []
    for document in classification.documents:
        reason = exclusion_reason_for_type(document.type)
        if reason is None:
            continue
        excluded.append(
            ExcludedDocument(
                id=document.id,
                name=document.name,
                type=document.type,
                pages=document.pages,
                reason=reason,
            )
        )
    return excluded


def processable_documents(
    classification: ClassificationResult,
    *,
    debug_mode: bool,
) -> list[ClassifiedDocument]:
    if not debug_mode:
        return list(classification.documents)

    excluded_ids = {item.id for item in identify_debug_exclusions(classification)}
    return [
        document
        for document in classification.documents
        if document.id not in excluded_ids
    ]
