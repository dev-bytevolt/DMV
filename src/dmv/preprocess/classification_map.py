from __future__ import annotations

from dmv.models.classification import ClassificationResult
from dmv.pdf_splitter import classified_pdf_filename_for_document


def build_filename_to_document_type(
    classification: ClassificationResult,
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for document in classification.documents:
        filename = classified_pdf_filename_for_document(document, classification.documents)
        mapping[filename] = document.type
    return mapping
