from dmv.models.classification import ClassificationResult
from dmv.classification_normalize import normalize_classification


def test_normalize_classification_moves_blank_documents_to_empty_pages() -> None:
    raw = ClassificationResult.from_dict(
        {
            "documents": [
                {
                    "id": "doc-001",
                    "name": "Driver License Copy",
                    "type": "driver_license",
                    "pages": [11],
                },
                {
                    "id": "doc-024",
                    "name": "Blank/Separator Page",
                    "type": "other",
                    "pages": [28, 30, 32],
                },
                {
                    "id": "doc-025",
                    "name": "Blank Separator Page 5",
                    "type": "other",
                    "pages": [5],
                },
            ],
            "empty_pages": [2, 4],
        }
    )

    normalized = normalize_classification(raw)

    assert [doc.name for doc in normalized.documents] == ["Driver License Copy"]
    assert normalized.empty_pages == [2, 4, 5, 28, 30, 32]


def test_normalize_classification_keeps_real_other_documents() -> None:
    raw = ClassificationResult.from_dict(
        {
            "documents": [
                {
                    "id": "doc-001",
                    "name": "Miscellaneous Supporting Document",
                    "type": "other",
                    "pages": [9],
                }
            ],
            "empty_pages": [],
        }
    )

    normalized = normalize_classification(raw)

    assert len(normalized.documents) == 1
    assert normalized.documents[0].name == "Miscellaneous Supporting Document"
