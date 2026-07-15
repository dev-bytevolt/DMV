from pathlib import Path

import fitz
import pytest

from dmv.preprocess.image_ops import PreprocessOptions
from dmv.preprocess.pdf import preprocess_pdf_file


def test_preprocess_pdf_file_writes_output(skewed_content_pdf: Path, tmp_path: Path) -> None:
    output_pdf = tmp_path / "corrected" / "skewed.pdf"
    result = preprocess_pdf_file(
        skewed_content_pdf,
        output_pdf,
        options=PreprocessOptions(dpi=120),
    )

    assert result.page_count == 1
    assert output_pdf.exists()
    with fitz.open(output_pdf) as doc:
        assert len(doc) == 1
        page = doc[0]
        assert page.rect.width == pytest.approx(8.5 * 72, rel=0.05)
        assert page.rect.height == pytest.approx(11.0 * 72, rel=0.05)


def test_preprocess_pdf_file_copies_unchanged_pages(tmp_path: Path) -> None:
    source_pdf = tmp_path / "source.pdf"
    output_pdf = tmp_path / "output.pdf"

    source_doc = fitz.open()
    page = source_doc.new_page(width=612, height=792)
    rect = fitz.Rect(72, 72, 540, 720)
    page.insert_textbox(rect, "unchanged page", fontsize=18)
    source_doc.save(source_pdf)
    source_doc.close()

    result = preprocess_pdf_file(
        source_pdf,
        output_pdf,
        options=PreprocessOptions(dpi=120),
        mode=__import__("dmv.preprocess.modes", fromlist=["PreprocessMode"]).PreprocessMode.FULL_PAGE_FORM,
    )

    assert result.page_count == 1
    with fitz.open(output_pdf) as output:
        assert len(output) == 1
        assert output[0].rect.width == pytest.approx(612, rel=0.01)
        assert output[0].get_images(full=True)


def test_preprocess_pdf_file_skips_blank_pages(tmp_path: Path) -> None:
    source_pdf = tmp_path / "source.pdf"
    output_pdf = tmp_path / "output.pdf"

    source_doc = fitz.open()
    blank_page = source_doc.new_page(width=612, height=792)
    blank_page.insert_text((72, 72), " ", fontsize=1, color=(1, 1, 1))
    content_page = source_doc.new_page(width=612, height=792)
    content_page.draw_rect(fitz.Rect(72, 72, 540, 720), color=(0, 0, 0), fill=(0, 0, 0))
    content_page.insert_text((100, 120), "Driver License Copy", fontsize=18)
    source_doc.save(source_pdf)
    source_doc.close()

    result = preprocess_pdf_file(
        source_pdf,
        output_pdf,
        options=PreprocessOptions(dpi=120),
    )

    assert result.page_count == 1
    with fitz.open(output_pdf) as output:
        assert len(output) == 1


def test_preprocess_pdf_file_handles_empty_source(tmp_path: Path) -> None:
    from pypdf import PdfWriter

    source_pdf = tmp_path / "empty.pdf"
    output_pdf = tmp_path / "output.pdf"

    writer = PdfWriter()
    with source_pdf.open("wb") as pdf_file:
        writer.write(pdf_file)

    result = preprocess_pdf_file(
        source_pdf,
        output_pdf,
        options=PreprocessOptions(dpi=120),
    )

    assert result.page_count == 0
    assert not output_pdf.exists()
