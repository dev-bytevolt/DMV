from dmv.models.classification import ClassificationResult, ClassifiedDocument


def test_classified_document_round_trip() -> None:
    payload = {
        "id": "doc-001",
        "name": "Insurance Card",
        "type": "insurance_card",
        "pages": [3, 1, 2],
    }
    document = ClassifiedDocument.from_dict(payload)

    assert document.pages == [1, 2, 3]
    assert document.to_dict() == {
        "id": "doc-001",
        "name": "Insurance Card",
        "type": "insurance_card",
        "pages": [1, 2, 3],
    }


def test_classification_result_round_trip(sample_classification) -> None:
    result = ClassificationResult.from_dict(sample_classification)
    assert result.to_dict() == sample_classification
