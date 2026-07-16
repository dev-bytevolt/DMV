import json
from pathlib import Path

import fitz
import pytest

from dmv.models.classification import ClassificationResult
from dmv.pdf_splitter import (
    get_pdf_page_count,
    split_pdf_by_classification,
    write_artifacts,
)


def test_get_pdf_page_count(sample_pdf: Path) -> None:
    assert get_pdf_page_count(sample_pdf) == 4


def test_split_pdf_by_classification(sample_pdf: Path, sample_classification, tmp_path: Path) -> None:
    classification = ClassificationResult.from_dict(sample_classification)
    output_dir = tmp_path / "split"
    written = split_pdf_by_classification(sample_pdf, classification, output_dir)

    assert len(written) == 2
    assert (output_dir / "Driver_License_Copy.pdf").exists()
    assert (output_dir / "Insurance_Card.pdf").exists()
    assert get_pdf_page_count(output_dir / "Driver_License_Copy.pdf") == 2
    assert get_pdf_page_count(output_dir / "Insurance_Card.pdf") == 1


def test_split_pdf_by_classification_clears_stale_files(
    sample_pdf: Path, sample_classification, tmp_path: Path
) -> None:
    classification = ClassificationResult.from_dict(sample_classification)
    output_dir = tmp_path / "split"
    output_dir.mkdir()
    stale = output_dir / "Dealer_Invoice.pdf"
    stale.write_bytes(b"%PDF-stale")

    split_pdf_by_classification(sample_pdf, classification, output_dir)

    assert not stale.exists()
    assert (output_dir / "Driver_License_Copy.pdf").exists()
    assert (output_dir / "Insurance_Card.pdf").exists()


def test_split_pdf_rejects_invalid_page(sample_pdf: Path, tmp_path: Path) -> None:
    classification = ClassificationResult.from_dict(
        {
            "documents": [
                {
                    "id": "doc-001",
                    "name": "Bad",
                    "type": "other",
                    "pages": [99],
                }
            ],
            "empty_pages": [],
        }
    )

    with pytest.raises(ValueError, match="invalid page"):
        split_pdf_by_classification(sample_pdf, classification, tmp_path / "split")


def test_split_pdf_skips_blank_pages(tmp_path: Path) -> None:
    source_pdf = tmp_path / "mixed.pdf"
    doc = fitz.open()
    blank_page = doc.new_page(width=612, height=792)
    blank_page.insert_text((72, 72), " ", fontsize=1, color=(1, 1, 1))
    content_page = doc.new_page(width=612, height=792)
    content_page.draw_rect(fitz.Rect(72, 72, 540, 720), color=(0, 0, 0), fill=(0, 0, 0))
    content_page.insert_text((100, 120), "Driver License Copy", fontsize=18)
    doc.save(source_pdf)
    doc.close()

    classification = ClassificationResult.from_dict(
        {
            "documents": [
                {
                    "id": "doc-001",
                    "name": "Driver License Copy",
                    "type": "driver_license",
                    "pages": [1, 2],
                }
            ],
            "empty_pages": [],
        }
    )
    output_dir = tmp_path / "split"
    split_pdf_by_classification(source_pdf, classification, output_dir)

    output_pdf = output_dir / "Driver_License_Copy.pdf"
    assert output_pdf.exists()
    assert get_pdf_page_count(output_pdf) == 1


def test_split_pdf_skips_all_blank_document(tmp_path: Path) -> None:
    source_pdf = tmp_path / "source.pdf"
    doc = fitz.open()
    blank_page = doc.new_page(width=612, height=792)
    blank_page.insert_text((72, 72), " ", fontsize=1, color=(1, 1, 1))
    content_page = doc.new_page(width=612, height=792)
    content_page.draw_rect(fitz.Rect(72, 72, 540, 720), color=(0, 0, 0), fill=(0, 0, 0))
    content_page.insert_text((100, 120), "Driver License Copy", fontsize=18)
    doc.save(source_pdf)
    doc.close()

    classification = ClassificationResult.from_dict(
        {
            "documents": [
                {
                    "id": "doc-001",
                    "name": "Blank Separator",
                    "type": "other",
                    "pages": [1],
                },
                {
                    "id": "doc-002",
                    "name": "Driver License Copy",
                    "type": "driver_license",
                    "pages": [2],
                },
            ],
            "empty_pages": [],
        }
    )
    output_dir = tmp_path / "split"
    written = split_pdf_by_classification(source_pdf, classification, output_dir)

    assert set(written) == {"doc-002"}
    assert not (output_dir / "Blank_Separator.pdf").exists()
    assert get_pdf_page_count(output_dir / "Driver_License_Copy.pdf") == 1


def test_write_artifacts(sample_pdf: Path, sample_classification, tmp_path: Path) -> None:
    classification = ClassificationResult.from_dict(sample_classification)
    artifact_dir = write_artifacts(sample_pdf, classification, tmp_path / "artifacts")

    assert artifact_dir.name == "sample"
    assert (artifact_dir / "original.pdf").exists()
    assert (artifact_dir / "doc_classification.json").exists()
    assert (artifact_dir / "classified" / "Driver_License_Copy.pdf").exists()

    saved = json.loads((artifact_dir / "doc_classification.json").read_text())
    assert saved == sample_classification
