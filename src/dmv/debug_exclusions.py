from __future__ import annotations

from dataclasses import dataclass

from dmv.models.classification import ClassifiedDocument, ClassificationResult

# Forms present in test fixtures but produced as pipeline output, not real input.
# Still extracted in debug mode when listed only under APPEND (EPA etc. live here).
DEBUG_OUTPUT_DOCUMENT_TYPES: frozenset[str] = frozenset(
    {
        "cover_letter",
        "universal_title_application",
        "vehicle_registration_application",
        "dealership_supplemental_title_form",
    }
)

# Skip LLM extraction entirely — these are pure pipeline outputs with no unique
# input fields (EPA lives on the supplemental form, so that type is extracted).
DEBUG_SKIP_EXTRACTION_TYPES: frozenset[str] = frozenset(
    {
        "cover_letter",
        "universal_title_application",
        "vehicle_registration_application",
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
        "Title Form is pipeline output (still extracted for EPA MPG); not appended"
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
    for_append: bool = False,
) -> list[ClassifiedDocument]:
    """Documents to extract (default) or append into ``output.pdf``.

    In debug mode, cover / UTA / BA-49 are skipped for extraction. The
    dealership supplemental form is still extracted (EPA MPG) but never
    appended, since the pipeline regenerates that form.
    """
    if not debug_mode:
        return list(classification.documents)

    skip_types = (
        DEBUG_OUTPUT_DOCUMENT_TYPES if for_append else DEBUG_SKIP_EXTRACTION_TYPES
    )
    return [
        document
        for document in classification.documents
        if document.type not in skip_types
    ]
