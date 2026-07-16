from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from dmv.config import Settings
from dmv.debug_exclusions import processable_documents
from dmv.extraction.schemas import normalize_extraction_payload
from dmv.models.classification import ClassificationResult
from dmv.models.usage import TokenUsage
from dmv.pdf_splitter import classified_pdf_filename_for_document
from dmv.providers.base import ExtractionProvider

logger = logging.getLogger(__name__)

# Conservative limit vs provider 32MB upload cap.
EXTRACTION_UPLOAD_MAX_BYTES = 30 * 1024 * 1024


@dataclass(frozen=True)
class ExtractionOutput:
    source_pdf: Path
    document_type: str
    document_name: str
    output_json: Path
    usage: TokenUsage


@dataclass(frozen=True)
class ExtractionStats:
    elapsed_seconds: float
    documents_processed: int
    usage: TokenUsage


@dataclass(frozen=True)
class ExtractionResult:
    extracted_dir: Path
    outputs: list[ExtractionOutput]
    stats: ExtractionStats


@dataclass(frozen=True)
class _ExtractionTarget:
    pdf_path: Path
    document_type: str
    document_name: str
    document_id: str


class ExtractionService:
    def __init__(
        self,
        settings: Settings,
        provider: ExtractionProvider | None = None,
    ) -> None:
        self._settings = settings
        if provider is None:
            from dmv.providers.openai_provider import create_extraction_provider

            provider = create_extraction_provider(settings)
        self._provider = provider

    async def extract_directory(
        self,
        corrected_dir: Path,
        extracted_dir: Path,
        *,
        classification: ClassificationResult,
        debug_mode: bool,
    ) -> ExtractionResult:
        import asyncio
        import time

        if extracted_dir.exists():
            shutil.rmtree(extracted_dir)
        extracted_dir.mkdir(parents=True, exist_ok=True)
        targets = self._targets_for_extraction(
            corrected_dir,
            classification,
            debug_mode=debug_mode,
        )

        if not targets:
            return ExtractionResult(
                extracted_dir=extracted_dir,
                outputs=[],
                stats=ExtractionStats(
                    elapsed_seconds=0.0,
                    documents_processed=0,
                    usage=TokenUsage.empty(),
                ),
            )

        started_at = time.perf_counter()
        semaphore = asyncio.Semaphore(self._settings.worker_pool_size)

        async def _run(target: _ExtractionTarget) -> ExtractionOutput:
            async with semaphore:
                return await self._extract_one(target, extracted_dir)

        outputs = await asyncio.gather(*(_run(target) for target in targets))
        usage = TokenUsage.empty()
        for output in outputs:
            usage = usage.merge(output.usage)

        elapsed_seconds = time.perf_counter() - started_at
        logger.info(
            "Extracted %s document(s) in %.1fs",
            len(outputs),
            elapsed_seconds,
        )
        return ExtractionResult(
            extracted_dir=extracted_dir,
            outputs=list(outputs),
            stats=ExtractionStats(
                elapsed_seconds=elapsed_seconds,
                documents_processed=len(outputs),
                usage=usage,
            ),
        )

    async def _extract_one(
        self,
        target: _ExtractionTarget,
        extracted_dir: Path,
    ) -> ExtractionOutput:
        max_bytes = EXTRACTION_UPLOAD_MAX_BYTES
        output_json = extracted_dir / f"{target.pdf_path.stem}.json"

        if target.pdf_path.stat().st_size <= max_bytes:
            outcome = await self._provider.extract_pdf(
                target.pdf_path,
                document_type=target.document_type,
                document_name=target.document_name,
            )
            output_json.write_text(
                json.dumps(
                    normalize_extraction_payload(
                        outcome.result,
                        document_type=target.document_type,
                    ),
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            logger.info(
                "Extracted %s -> %s (type=%s)",
                target.pdf_path.name,
                output_json.name,
                target.document_type,
            )
            return ExtractionOutput(
                source_pdf=target.pdf_path,
                document_type=target.document_type,
                document_name=target.document_name,
                output_json=output_json,
                usage=outcome.usage,
            )

        # If the corrected PDF is too large to upload, split into smaller parts and
        # extract each part separately. The output JSON will be a list of per-part
        # raw AI payloads (still \"raw output\"), in scan order.
        part_paths = self._split_pdf_for_upload(
            target.pdf_path,
            extracted_dir=extracted_dir,
            max_bytes=max_bytes,
        )
        combined_usage = TokenUsage.empty()
        part_payloads: list[dict] = []
        for idx, part_path in enumerate(part_paths, start=1):
            outcome = await self._provider.extract_pdf(
                part_path,
                document_type=target.document_type,
                document_name=f"{target.document_name} (part {idx}/{len(part_paths)})",
            )
            combined_usage = combined_usage.merge(outcome.usage)
            part_payloads.append(outcome.result)

        output_json.write_text(
            json.dumps(
                [
                    normalize_extraction_payload(
                        item,
                        document_type=target.document_type,
                    )
                    for item in part_payloads
                ],
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        logger.info(
            "Extracted %s -> %s (type=%s)",
            target.pdf_path.name,
            output_json.name,
            target.document_type,
        )
        return ExtractionOutput(
            source_pdf=target.pdf_path,
            document_type=target.document_type,
            document_name=target.document_name,
            output_json=output_json,
            usage=combined_usage,
        )

    def _split_pdf_for_upload(
        self,
        pdf_path: Path,
        *,
        extracted_dir: Path,
        max_bytes: int,
    ) -> list[Path]:
        import io

        import fitz

        def _serialized_pdf_size(doc: fitz.Document) -> int:
            buffer = io.BytesIO()
            doc.save(buffer, garbage=4, deflate=True)
            return len(buffer.getvalue())

        chunks_dir = extracted_dir / "_chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)

        with fitz.open(pdf_path) as doc:
            if len(doc) == 0:
                return [pdf_path]

            part_paths: list[Path] = []
            part_idx = 1
            start_page = 0

            while start_page < len(doc):
                end_page = start_page
                last_good = start_page

                while end_page < len(doc):
                    with fitz.open() as trial:
                        trial.insert_pdf(doc, from_page=start_page, to_page=end_page)
                        if _serialized_pdf_size(trial) <= max_bytes:
                            last_good = end_page
                            end_page += 1
                            continue
                    break

                with fitz.open() as chunk:
                    chunk.insert_pdf(doc, from_page=start_page, to_page=last_good)
                    out_path = chunks_dir / f"{pdf_path.stem}.part{part_idx:02d}.pdf"
                    chunk.save(out_path, garbage=4, deflate=True)
                    part_paths.append(out_path)
                    part_idx += 1

                start_page = last_good + 1

            return part_paths or [pdf_path]

    def _targets_for_extraction(
        self,
        corrected_dir: Path,
        classification: ClassificationResult,
        *,
        debug_mode: bool,
    ) -> list[_ExtractionTarget]:
        documents = processable_documents(classification, debug_mode=debug_mode)
        targets: list[_ExtractionTarget] = []
        for document in documents:
            filename = classified_pdf_filename_for_document(
                document,
                classification.documents,
            )
            pdf_path = corrected_dir / filename
            if not pdf_path.is_file():
                logger.warning(
                    "Skipping extraction for %s (%s): corrected PDF not found at %s",
                    document.id,
                    document.name,
                    pdf_path,
                )
                continue
            targets.append(
                _ExtractionTarget(
                    pdf_path=pdf_path,
                    document_type=document.type,
                    document_name=document.name,
                    document_id=document.id,
                )
            )
        return targets
