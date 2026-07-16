import json
from pathlib import Path

from dmv.debug_exclusions import (
    DEBUG_OUTPUT_DOCUMENT_TYPES,
    identify_debug_exclusions,
    processable_documents,
)
from dmv.models.classification import ClassificationResult


def test_debug_output_document_types_match_fixture_forms() -> None:
    assert DEBUG_OUTPUT_DOCUMENT_TYPES == {
        "cover_letter",
        "universal_title_application",
        "vehicle_registration_application",
        "dealership_supplemental_title_form",
    }


def test_identify_debug_exclusions_on_fixture_examples() -> None:
    artifacts_dir = Path("artifacts")
    if not artifacts_dir.is_dir():
        return

    for classification_path in sorted(artifacts_dir.glob("*/doc_classification.json")):
        classification = ClassificationResult.from_dict(
            json.loads(classification_path.read_text(encoding="utf-8"))
        )
        excluded = identify_debug_exclusions(classification)
        assert len(excluded) == 4, classification_path
        excluded_types = {item.type for item in excluded}
        assert excluded_types == DEBUG_OUTPUT_DOCUMENT_TYPES
        for item in excluded:
            assert item.reason.startswith("test fixture —")


def test_processable_documents_respects_debug_mode(sample_classification) -> None:
    classification = ClassificationResult.from_dict(sample_classification)
    with_debug = processable_documents(classification, debug_mode=True)
    without_debug = processable_documents(classification, debug_mode=False)

    assert len(without_debug) == 2
    assert len(with_debug) == 2
    assert {doc.type for doc in with_debug} == {"driver_license", "insurance_card"}


def test_processable_documents_excludes_fixture_forms() -> None:
    classification = ClassificationResult.from_dict(
        {
            "documents": [
                {
                    "id": "doc-001",
                    "name": "MV Express Cover Letter to NJ DMV",
                    "type": "cover_letter",
                    "pages": [1],
                },
                {
                    "id": "doc-002",
                    "name": "Driver License Copy",
                    "type": "driver_license",
                    "pages": [11, 13],
                },
                {
                    "id": "doc-003",
                    "name": "Universal Title Application",
                    "type": "universal_title_application",
                    "pages": [5],
                },
            ],
            "empty_pages": [2, 3, 4, 6, 7, 8, 9, 10, 12],
        }
    )

    processable = processable_documents(classification, debug_mode=True)
    excluded = identify_debug_exclusions(classification)

    assert len(excluded) == 2
    assert len(processable) == 1
    assert processable[0].type == "driver_license"
