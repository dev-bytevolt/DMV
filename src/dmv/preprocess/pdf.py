from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import fitz
import numpy as np

from dmv.page_content import is_blank_scanned_page
from dmv.preprocess.image_ops import (
    PreprocessOptions,
    bgr_to_png_bytes,
    pixmap_to_bgr,
    preprocess_page_image,
)
from dmv.preprocess.modes import PreprocessMode

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PdfPreprocessResult:
    source_pdf: Path
    output_pdf: Path
    page_count: int
    document_type: str
    mode: PreprocessMode


def preprocess_pdf_file(
    source_pdf: Path,
    output_pdf: Path,
    *,
    options: PreprocessOptions,
    document_type: str = "other",
    mode: PreprocessMode = PreprocessMode.DEFAULT,
) -> PdfPreprocessResult:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    with fitz.open(source_pdf) as source_doc:
        if len(source_doc) == 0:
            logger.warning("Skipping empty source PDF: %s", source_pdf.name)
            return PdfPreprocessResult(
                source_pdf=source_pdf,
                output_pdf=output_pdf,
                page_count=0,
                document_type=document_type,
                mode=mode,
            )

        output_doc = fitz.open()
        try:
            page_count = _write_preprocessed_pages(
                source_doc,
                output_doc,
                options,
                mode=mode,
            )
            if page_count == 0:
                logger.warning(
                    "All pages in %s classified as blank during preprocessing; "
                    "retrying without blank-page filtering",
                    source_pdf.name,
                )
                page_count = _write_preprocessed_pages(
                    source_doc,
                    output_doc,
                    options,
                    mode=mode,
                    skip_blank_pages=False,
                )

            if page_count == 0:
                logger.warning(
                    "No pages produced for %s; copying source PDF unchanged",
                    source_pdf.name,
                )
                shutil.copy2(source_pdf, output_pdf)
                page_count = len(source_doc)
            else:
                output_doc.save(output_pdf)
        finally:
            output_doc.close()

    logger.info(
        "Preprocessed %s -> %s (%s page(s), type=%s, mode=%s)",
        source_pdf.name,
        output_pdf.name,
        page_count,
        document_type,
        mode.value,
    )
    return PdfPreprocessResult(
        source_pdf=source_pdf,
        output_pdf=output_pdf,
        page_count=page_count,
        document_type=document_type,
        mode=mode,
    )


def _write_preprocessed_pages(
    source_doc: fitz.Document,
    output_doc: fitz.Document,
    options: PreprocessOptions,
    *,
    mode: PreprocessMode,
    skip_blank_pages: bool = True,
) -> int:
    for page_number, page in enumerate(source_doc):
        pixmap = page.get_pixmap(dpi=options.dpi, alpha=False)
        image = pixmap_to_bgr(
            pixmap.samples,
            pixmap.width,
            pixmap.height,
            pixmap.n,
        )
        if skip_blank_pages:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if is_blank_scanned_page(gray):
                logger.info(
                    "Skipping blank page %s in %s",
                    page_number + 1,
                    source_doc.name,
                )
                continue

        processed = preprocess_page_image(image, options, mode=mode)
        if np.array_equal(image, processed) and _write_unchanged_page(
            output_doc,
            source_doc,
            page_number,
        ):
            continue

        page_width = processed.shape[1] * 72.0 / options.dpi
        page_height = processed.shape[0] * 72.0 / options.dpi
        page_rect = fitz.Rect(0, 0, page_width, page_height)
        new_page = output_doc.new_page(
            width=page_width,
            height=page_height,
        )
        new_page.insert_image(page_rect, stream=bgr_to_png_bytes(processed))

    return len(output_doc)


def _write_unchanged_page(
    output_doc: fitz.Document,
    source_doc: fitz.Document,
    page_number: int,
) -> bool:
    page = source_doc[page_number]
    images = page.get_images(full=True)
    if len(images) != 1:
        return False

    image_xref = images[0][0]
    extracted = source_doc.extract_image(image_xref)
    if extracted["ext"] not in {"jpeg", "jpg", "png"}:
        return False

    new_page = output_doc.new_page(width=page.rect.width, height=page.rect.height)
    new_page.insert_image(page.rect, stream=extracted["image"])
    return True
