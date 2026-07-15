from pathlib import Path

from dmv.classification_recovery import recover_misclassified_empty_pages
from dmv.models.classification import ClassificationResult
from dmv.page_content import page_content_score


def test_recover_misclassified_empty_pages_attaches_to_previous_document() -> None:
    pdf_path = Path(
        "artifacts/HAJAL, ANGELA - NJ LEASE - INTERSTATE TOYOTA/original.pdf"
    )
    if not pdf_path.is_file():
        return

    classification = ClassificationResult.from_dict(
        {
            "documents": [
                {
                    "id": "doc-002",
                    "name": "Check payable to NJ DMV",
                    "type": "check_payment",
                    "pages": [3],
                }
            ],
            "empty_pages": [2, 4, 6],
        }
    )

    recovered = recover_misclassified_empty_pages(pdf_path, classification)

    check = recovered.documents[0]
    assert check.pages == [3, 4]
    assert 4 not in recovered.empty_pages
    assert 2 in recovered.empty_pages
    assert 6 in recovered.empty_pages


def test_page_content_score_distinguishes_blank_from_check_back() -> None:
    pdf_path = Path(
        "artifacts/HAJAL, ANGELA - NJ LEASE - INTERSTATE TOYOTA/original.pdf"
    )
    if not pdf_path.is_file():
        return

    assert page_content_score(pdf_path, 2) < 0.008
    assert page_content_score(pdf_path, 4) >= 0.008
