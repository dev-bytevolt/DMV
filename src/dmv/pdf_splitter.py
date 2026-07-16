from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from pypdf import PdfReader, PdfWriter

from dmv.page_content import is_blank_page
from dmv.models.classification import ClassifiedDocument, ClassificationResult
from dmv.utils.filename import sanitize_filename

logger = logging.getLogger(__name__)


def get_pdf_page_count(pdf_path: Path) -> int:
    reader = PdfReader(str(pdf_path))
    return len(reader.pages)


def split_pdf_by_classification(
    source_pdf: Path,
    classification: ClassificationResult,
    output_dir: Path,
) -> dict[str, Path]:
    reader = PdfReader(str(source_pdf))
    total_pages = len(reader.pages)
    # Drop leftovers from prior runs (classification names change between packets).
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    used_names: set[str] = set()

    for document in classification.documents:
        writer = PdfWriter()
        for page_number in document.pages:
            if page_number < 1 or page_number > total_pages:
                raise ValueError(
                    f"Document {document.id!r} references invalid page {page_number} "
                    f"in a {total_pages}-page PDF"
                )
            if is_blank_page(source_pdf, page_number):
                logger.info(
                    "Skipping blank page %s for document %s (%s)",
                    page_number,
                    document.id,
                    document.name,
                )
                continue
            writer.add_page(reader.pages[page_number - 1])

        if len(writer.pages) == 0:
            logger.warning(
                "Skipping document %s (%s): all referenced pages are blank",
                document.id,
                document.name,
            )
            continue

        filename = classified_pdf_filename_for_document(document, classification.documents)
        used_names.add(filename)
        output_path = output_dir / filename
        with output_path.open("wb") as output_file:
            writer.write(output_file)
        written[document.id] = output_path

    return written


def classified_pdf_filename_for_document(
    document: ClassifiedDocument,
    documents: list[ClassifiedDocument],
) -> str:
    used_names: set[str] = set()
    for current in documents:
        base_name = sanitize_filename(current.name)
        filename = _unique_pdf_name(base_name, used_names)
        used_names.add(filename)
        if current.id == document.id:
            return filename
    raise ValueError(f"Document {document.id!r} not found in classification list")


def _unique_pdf_name(base_name: str, used_names: set[str]) -> str:
    candidate = f"{base_name}.pdf"
    if candidate not in used_names:
        return candidate

    suffix = 2
    while True:
        candidate = f"{base_name}_{suffix}.pdf"
        if candidate not in used_names:
            return candidate
        suffix += 1


def write_artifacts(
    source_pdf: Path,
    classification: ClassificationResult,
    artifacts_root: Path,
) -> Path:
    artifact_dir = artifacts_root / source_pdf.stem
    artifact_dir.mkdir(parents=True, exist_ok=True)

    original_copy = artifact_dir / "original.pdf"
    shutil.copy2(source_pdf, original_copy)

    classification_path = artifact_dir / "doc_classification.json"
    classification_path.write_text(
        json.dumps(classification.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    classified_dir = artifact_dir / "classified"
    split_pdf_by_classification(source_pdf, classification, classified_dir)
    return artifact_dir


def artifact_paths(artifact_dir: Path) -> tuple[Path, Path, Path]:
    return (
        artifact_dir / "classified",
        artifact_dir / "corrected",
        artifact_dir / "extracted",
    )
