from dmv.models.classification import ClassificationResult
from dmv.validation import (
    validate_classification,
    validate_document_contiguity,
    validate_page_coverage,
)


def test_validate_page_coverage_complete(sample_classification) -> None:
    classification = ClassificationResult.from_dict(sample_classification)
    report = validate_page_coverage(classification, total_pages=4)

    assert report.is_complete is True
    assert report.missing_pages == []
    assert report.duplicate_pages == []
    assert report.invalid_pages == []


def test_validate_page_coverage_missing_pages(sample_classification) -> None:
    classification = ClassificationResult.from_dict(sample_classification)
    report = validate_page_coverage(classification, total_pages=5)

    assert report.is_complete is False
    assert report.missing_pages == [5]


def test_validate_page_coverage_duplicate_pages() -> None:
    classification = ClassificationResult.from_dict(
        {
            "documents": [
                {
                    "id": "doc-001",
                    "name": "A",
                    "type": "other",
                    "pages": [1, 2],
                },
                {
                    "id": "doc-002",
                    "name": "B",
                    "type": "other",
                    "pages": [2],
                },
            ],
            "empty_pages": [],
        }
    )
    report = validate_page_coverage(classification, total_pages=2)

    assert report.is_complete is False
    assert report.duplicate_pages == [2]


def test_validate_page_coverage_invalid_pages(sample_classification) -> None:
    classification = ClassificationResult.from_dict(sample_classification)
    classification.empty_pages.append(99)
    report = validate_page_coverage(classification, total_pages=4)

    assert report.is_complete is False
    assert report.invalid_pages == [99]


def test_validate_document_contiguity_allows_empty_pages_within_span() -> None:
    classification = ClassificationResult.from_dict(
        {
            "documents": [
                {
                    "id": "doc-012",
                    "name": "Motor Vehicle Lease Agreement",
                    "type": "lease_agreement",
                    "pages": [27, 29, 31, 33, 35, 37],
                }
            ],
            "empty_pages": [26, 28, 30, 32, 34, 36],
        }
    )

    report = validate_document_contiguity(classification)

    assert report.is_valid is True
    assert report.issues == []


def test_validate_document_contiguity_rejects_interleaved_documents() -> None:
    classification = ClassificationResult.from_dict(
        {
            "documents": [
                {
                    "id": "doc-008",
                    "name": "Limited Power of Attorney",
                    "type": "limited_power_of_attorney",
                    "pages": [15, 25],
                },
                {
                    "id": "doc-006",
                    "name": "Insurance Cards",
                    "type": "insurance_card",
                    "pages": [21, 22],
                },
            ],
            "empty_pages": [],
        }
    )

    report = validate_document_contiguity(classification)

    assert report.is_valid is False
    assert len(report.issues) == 1
    assert report.issues[0].document_id == "doc-008"
    assert report.issues[0].conflicting_pages == [21, 22]


def test_validate_classification_combines_coverage_and_contiguity() -> None:
    classification = ClassificationResult.from_dict(
        {
            "documents": [
                {
                    "id": "doc-001",
                    "name": "Lease Agreement",
                    "type": "lease_agreement",
                    "pages": [1, 3],
                }
            ],
            "empty_pages": [2],
        }
    )

    report = validate_classification(classification, total_pages=3)

    assert report.is_valid is True
