from dmv.models.classification import ClassificationResult
from dmv.pdf_splitter import classified_pdf_filename_for_document
from dmv.preprocess.classification_map import build_filename_to_document_type


def test_classified_pdf_filename_for_document_handles_duplicates() -> None:
    documents = ClassificationResult.from_dict(
        {
            "documents": [
                {
                    "id": "doc-001",
                    "name": "Insurance Card",
                    "type": "insurance_card",
                    "pages": [1],
                },
                {
                    "id": "doc-002",
                    "name": "Insurance Card",
                    "type": "insurance_card",
                    "pages": [2],
                },
            ],
            "empty_pages": [],
        }
    ).documents

    first = classified_pdf_filename_for_document(documents[0], documents)
    second = classified_pdf_filename_for_document(documents[1], documents)

    assert first == "Insurance_Card.pdf"
    assert second == "Insurance_Card_2.pdf"


def test_build_filename_to_document_type() -> None:
    classification = ClassificationResult.from_dict(
        {
            "documents": [
                {
                    "id": "doc-001",
                    "name": "Driver License",
                    "type": "driver_license",
                    "pages": [1],
                },
                {
                    "id": "doc-002",
                    "name": "Dealer Invoice",
                    "type": "dealer_invoice",
                    "pages": [2],
                },
            ],
            "empty_pages": [],
        }
    )

    mapping = build_filename_to_document_type(classification)

    assert mapping["Driver_License.pdf"] == "driver_license"
    assert mapping["Dealer_Invoice.pdf"] == "dealer_invoice"
