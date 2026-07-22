from __future__ import annotations

import logging
import shutil
from pathlib import Path

import fitz
from pypdf import PdfReader, PdfWriter

logger = logging.getLogger(__name__)

# Blank AcroForms use ``/Helv 0 Tf`` (auto-size). Pin a uniform size before bake
# so VIN / short fields / long names all render at the same point size.
FORM_FIELD_FONTSIZE = 9.0


def fill_acroform_pdf(
    blank_path: Path,
    output_path: Path,
    field_values: dict[str, str],
) -> Path:
    """Copy ``blank_path``, write AcroForm values, then bake fields to plain text."""
    if not blank_path.is_file():
        raise FileNotFoundError(f"Blank form not found: {blank_path}")

    reader = PdfReader(str(blank_path))
    writer = PdfWriter()
    writer.append(reader)
    _strip_acrobat_prompts(writer)

    if field_values:
        known = set((reader.get_fields() or {}).keys())
        applied = {k: v for k, v in field_values.items() if k in known}
        skipped = sorted(set(field_values) - known)
        if skipped:
            logger.debug(
                "Skipping unknown AcroForm fields on %s: %s",
                blank_path.name,
                ", ".join(skipped),
            )
        if applied:
            for page in writer.pages:
                if "/Annots" not in page:
                    continue
                writer.update_page_form_field_values(page, applied)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        writer.write(handle)

    _bake_form_fields(output_path, fontsize=FORM_FIELD_FONTSIZE)

    logger.info(
        "Filled %s -> %s (%s field(s), flattened)",
        blank_path.name,
        output_path.name,
        len(field_values),
    )
    return output_path


def _strip_acrobat_prompts(writer: PdfWriter) -> None:
    """Remove blank-template JS alerts (e.g. BA-49 ONLINE FORM INSTRUCTIONS)."""
    root = writer._root_object
    if "/OpenAction" in root:
        del root["/OpenAction"]
    for page in writer.pages:
        if "/AA" in page:
            del page["/AA"]


def _pin_text_field_fontsize(doc: fitz.Document, fontsize: float) -> None:
    """Force every text widget to a fixed size (overrides auto-size DA ``0 Tf``)."""
    for page in doc:
        for widget in page.widgets() or []:
            if widget.field_type_string != "Text":
                continue
            widget.text_fontsize = fontsize
            widget.update()


def _bake_form_fields(pdf_path: Path, *, fontsize: float = FORM_FIELD_FONTSIZE) -> None:
    """Convert AcroForm widgets to permanent page content (non-editable)."""
    doc = fitz.open(pdf_path)
    try:
        _pin_text_field_fontsize(doc, fontsize)
        doc.bake(widgets=True, annots=False)
        baked = pdf_path.with_suffix(pdf_path.suffix + ".baked")
        doc.save(str(baked), garbage=3, deflate=True)
    finally:
        doc.close()
    baked.replace(pdf_path)


def copy_blank_if_empty(blank_path: Path, output_path: Path) -> Path:
    """Fallback when no values are available — still write a blank copy."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(blank_path, output_path)
    return output_path
