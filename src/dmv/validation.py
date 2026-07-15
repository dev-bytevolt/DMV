from __future__ import annotations

from dataclasses import dataclass

from dmv.models.classification import ClassificationResult


@dataclass(frozen=True)
class PageCoverageReport:
    total_pages: int
    covered_pages: set[int]
    missing_pages: list[int]
    duplicate_pages: list[int]
    invalid_pages: list[int]
    is_complete: bool


@dataclass(frozen=True)
class DocumentContiguityIssue:
    document_id: str
    document_name: str
    document_pages: list[int]
    conflicting_pages: list[int]


@dataclass(frozen=True)
class DocumentContiguityReport:
    issues: list[DocumentContiguityIssue]
    is_valid: bool


@dataclass(frozen=True)
class ClassificationValidationReport:
    coverage: PageCoverageReport
    contiguity: DocumentContiguityReport

    @property
    def is_valid(self) -> bool:
        return self.coverage.is_complete and self.contiguity.is_valid


def validate_page_coverage(
    classification: ClassificationResult,
    total_pages: int,
) -> PageCoverageReport:
    if total_pages < 1:
        raise ValueError("total_pages must be at least 1")

    valid_range = set(range(1, total_pages + 1))
    seen: dict[int, int] = {}
    invalid_pages: list[int] = []

    for page in classification.empty_pages:
        if page in valid_range:
            seen[page] = seen.get(page, 0) + 1
        else:
            invalid_pages.append(page)

    for document in classification.documents:
        for page in document.pages:
            if page in valid_range:
                seen[page] = seen.get(page, 0) + 1
            else:
                invalid_pages.append(page)

    duplicate_pages = sorted(page for page, count in seen.items() if count > 1)
    covered_pages = set(seen.keys())
    missing_pages = sorted(valid_range - covered_pages)

    return PageCoverageReport(
        total_pages=total_pages,
        covered_pages=covered_pages,
        missing_pages=missing_pages,
        duplicate_pages=duplicate_pages,
        invalid_pages=sorted(set(invalid_pages)),
        is_complete=not missing_pages and not duplicate_pages and not invalid_pages,
    )


def validate_document_contiguity(
    classification: ClassificationResult,
) -> DocumentContiguityReport:
    empty_pages = set(classification.empty_pages)
    page_to_document: dict[int, tuple[str, str]] = {}

    for document in classification.documents:
        for page in document.pages:
            page_to_document[page] = (document.id, document.name)

    issues: list[DocumentContiguityIssue] = []

    for document in classification.documents:
        if len(document.pages) <= 1:
            continue

        document_pages = set(document.pages)
        span_start = min(document.pages)
        span_end = max(document.pages)
        conflicting_pages: list[int] = []

        for page in range(span_start, span_end + 1):
            if page in document_pages or page in empty_pages:
                continue
            if page in page_to_document:
                conflicting_pages.append(page)

        if conflicting_pages:
            issues.append(
                DocumentContiguityIssue(
                    document_id=document.id,
                    document_name=document.name,
                    document_pages=document.pages,
                    conflicting_pages=conflicting_pages,
                )
            )

    return DocumentContiguityReport(
        issues=issues,
        is_valid=not issues,
    )


def validate_classification(
    classification: ClassificationResult,
    total_pages: int,
) -> ClassificationValidationReport:
    return ClassificationValidationReport(
        coverage=validate_page_coverage(classification, total_pages),
        contiguity=validate_document_contiguity(classification),
    )
