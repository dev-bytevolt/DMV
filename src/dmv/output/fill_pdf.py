from __future__ import annotations

import logging
import shutil
from pathlib import Path

from pypdf import PdfReader, PdfWriter

logger = logging.getLogger(__name__)


def fill_acroform_pdf(
    blank_path: Path,
    output_path: Path,
    field_values: dict[str, str],
) -> Path:
    """Copy ``blank_path`` and write AcroForm field values to ``output_path``."""
    if not blank_path.is_file():
        raise FileNotFoundError(f"Blank form not found: {blank_path}")

    reader = PdfReader(str(blank_path))
    writer = PdfWriter()
    writer.append(reader)

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
    logger.info(
        "Filled %s -> %s (%s field(s))",
        blank_path.name,
        output_path.name,
        len(field_values),
    )
    return output_path


def copy_blank_if_empty(blank_path: Path, output_path: Path) -> Path:
    """Fallback when no values are available — still write a blank copy."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(blank_path, output_path)
    return output_path
