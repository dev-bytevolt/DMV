from __future__ import annotations

import re

from dmv.models.classification import ClassificationResult, ClassifiedDocument

_BLANK_PAGE_NAME = re.compile(
    r"\b("
    r"blank|separator|seperator|empty(?:\s+page)?|divider|unused|"
    r"no\s+content|intentionally\s+blank"
    r")\b",
    re.IGNORECASE,
)


def is_blank_page_document(document: ClassifiedDocument) -> bool:
    return bool(_BLANK_PAGE_NAME.search(document.name))


def normalize_classification(
    classification: ClassificationResult,
) -> ClassificationResult:
    documents: list[ClassifiedDocument] = []
    empty_pages = set(classification.empty_pages)

    for document in classification.documents:
        if is_blank_page_document(document):
            empty_pages.update(document.pages)
        else:
            documents.append(document)

    return ClassificationResult(
        documents=documents,
        empty_pages=sorted(empty_pages),
    )
