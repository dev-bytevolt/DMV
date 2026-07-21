from __future__ import annotations

import logging
from pathlib import Path

from pypdf import PdfReader, PdfWriter

logger = logging.getLogger(__name__)


def merge_pdfs(paths: list[Path], output_path: Path) -> Path:
    """Concatenate PDFs in order into ``output_path``."""
    writer = PdfWriter()
    appended = 0
    for path in paths:
        if not path.is_file():
            logger.warning("Skipping missing PDF during assemble: %s", path)
            continue
        reader = PdfReader(str(path))
        if len(reader.pages) == 0:
            logger.warning("Skipping empty PDF during assemble: %s", path)
            continue
        writer.append(reader)
        appended += 1

    if appended == 0:
        raise ValueError("No PDF pages available to assemble output packet")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        writer.write(handle)
    logger.info("Assembled %s from %s PDF(s)", output_path.name, appended)
    return output_path
