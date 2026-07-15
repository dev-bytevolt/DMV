from __future__ import annotations

import logging
from pathlib import Path

from dmv.models.classification import ClassificationResult, ClassifiedDocument
from dmv.page_content import MIN_EMPTY_PAGE_CONTENT_SCORE, page_content_score

logger = logging.getLogger(__name__)


def recover_misclassified_empty_pages(
    pdf_path: Path,
    classification: ClassificationResult,
) -> ClassificationResult:
    if not classification.empty_pages:
        return classification

    recovered_pages: list[int] = []
    remaining_empty: list[int] = []
    for page_number in classification.empty_pages:
        if page_content_score(pdf_path, page_number) >= MIN_EMPTY_PAGE_CONTENT_SCORE:
            recovered_pages.append(page_number)
        else:
            remaining_empty.append(page_number)

    if not recovered_pages:
        return classification

    documents = [
        ClassifiedDocument(
            id=document.id,
            name=document.name,
            type=document.type,
            pages=list(document.pages),
        )
        for document in classification.documents
    ]

    attached_pages: list[int] = []
    for page_number in sorted(recovered_pages):
        target = _document_for_recovered_page(documents, page_number)
        if target is None:
            continue
        if page_number not in target.pages:
            target.pages.append(page_number)
            target.pages.sort()
        attached_pages.append(page_number)

    if not attached_pages:
        return classification

    logger.info(
        "Recovered %s misclassified empty page(s): %s",
        len(attached_pages),
        ", ".join(str(page) for page in attached_pages),
    )

    return ClassificationResult(
        documents=documents,
        empty_pages=sorted(set(remaining_empty) - set(attached_pages)),
    )


def _document_for_recovered_page(
    documents: list[ClassifiedDocument],
    page_number: int,
) -> ClassifiedDocument | None:
    for document in documents:
        if (page_number - 1) in document.pages:
            return document

    preceding = [
        document
        for document in documents
        if document.pages and max(document.pages) < page_number
    ]
    if preceding:
        return max(preceding, key=lambda document: max(document.pages))

    following = [
        document
        for document in documents
        if document.pages and min(document.pages) > page_number
    ]
    if following:
        return min(following, key=lambda document: min(document.pages))

    return documents[0] if documents else None
