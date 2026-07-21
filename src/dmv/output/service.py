from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dmv.debug_exclusions import processable_documents
from dmv.models.classification import ClassificationResult
from dmv.output.assemble import merge_pdfs
from dmv.output.cover_letter import build_cover_letter_pdf
from dmv.output.fill_pdf import fill_acroform_pdf
from dmv.output.mappings import (
    BLANK_BA49,
    BLANK_OWNERSHIP,
    BLANK_UTA,
    OUTPUT_BA49,
    OUTPUT_COVER,
    OUTPUT_OWNERSHIP,
    OUTPUT_PACKET_FILENAME,
    OUTPUT_UTA,
    build_ba49_fields,
    build_ownership_fields,
    build_uta_fields,
)
from dmv.output.tax_stamp import apply_uta_tax_stamp
from dmv.pdf_splitter import classified_pdf_filename_for_document

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OutputPacketResult:
    output_dir: Path
    cover_letter_pdf: Path
    uta_pdf: Path
    ba49_pdf: Path
    ownership_pdf: Path
    output_pdf: Path
    appended_document_count: int
    page_count: int


def load_consolidated_data(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected consolidated object in {path}")
    return payload


def build_output_packet(
    *,
    artifact_dir: Path,
    classified_dir: Path,
    consolidated_json: Path,
    classification: ClassificationResult,
    blanks_dir: Path,
    debug_mode: bool,
) -> OutputPacketResult:
    """Fill forms into ``output/`` and assemble root ``output.pdf``."""
    data = load_consolidated_data(consolidated_json)
    output_dir = artifact_dir / "output"
    if output_dir.exists():
        for stale in output_dir.glob("*.pdf"):
            stale.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    cover_path = output_dir / OUTPUT_COVER
    uta_path = output_dir / OUTPUT_UTA
    ba49_path = output_dir / OUTPUT_BA49
    ownership_path = output_dir / OUTPUT_OWNERSHIP

    build_cover_letter_pdf(data, cover_path)
    fill_acroform_pdf(blanks_dir / BLANK_UTA, uta_path, build_uta_fields(data))
    apply_uta_tax_stamp(uta_path, data)
    fill_acroform_pdf(blanks_dir / BLANK_BA49, ba49_path, build_ba49_fields(data))
    fill_acroform_pdf(
        blanks_dir / BLANK_OWNERSHIP,
        ownership_path,
        build_ownership_fields(data),
    )

    merge_paths: list[Path] = [cover_path, uta_path, ba49_path, ownership_path]
    docs = processable_documents(classification, debug_mode=debug_mode)
    appended = 0
    for document in docs:
        filename = classified_pdf_filename_for_document(
            document,
            classification.documents,
        )
        classified_pdf = classified_dir / filename
        if not classified_pdf.is_file():
            logger.warning(
                "Classified PDF missing for %s (%s): %s",
                document.id,
                document.name,
                classified_pdf.name,
            )
            continue
        merge_paths.append(classified_pdf)
        appended += 1

    output_pdf = artifact_dir / OUTPUT_PACKET_FILENAME
    merge_pdfs(merge_paths, output_pdf)

    from pypdf import PdfReader

    page_count = len(PdfReader(str(output_pdf)).pages)
    logger.info(
        "Output packet ready: %s (%s pages, %s classified doc(s))",
        output_pdf.name,
        page_count,
        appended,
    )
    return OutputPacketResult(
        output_dir=output_dir,
        cover_letter_pdf=cover_path,
        uta_pdf=uta_path,
        ba49_pdf=ba49_path,
        ownership_pdf=ownership_path,
        output_pdf=output_pdf,
        appended_document_count=appended,
        page_count=page_count,
    )
