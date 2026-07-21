from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from dmv.classification_normalize import is_blank_page_document, normalize_classification
from dmv.classification_recovery import recover_misclassified_empty_pages
from dmv.config import Settings
from dmv.cost import estimate_cost
from dmv.consolidation.service import ConsolidationResult, consolidate_extractions
from dmv.debug_exclusions import ExcludedDocument, identify_debug_exclusions
from dmv.extraction.service import ExtractionResult, ExtractionService
from dmv.models.classification import ClassificationResult
from dmv.models.usage import ProcessingStats, TokenUsage
from dmv.output.service import OutputPacketResult, build_output_packet
from dmv.pdf_splitter import artifact_paths, get_pdf_page_count, write_artifacts
from dmv.preprocess.service import PreprocessingResult, PreprocessingService
from dmv.providers import ClassificationProvider, create_classification_provider
from dmv.validation import (
    ClassificationValidationReport,
    validate_classification,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FileProcessingResult:
    source_pdf: Path
    classification: ClassificationResult
    validation: ClassificationValidationReport
    artifact_dir: Path
    classified_dir: Path
    corrected_dir: Path
    extracted_dir: Path
    stats: ProcessingStats
    preprocessing: PreprocessingResult
    extraction: ExtractionResult
    consolidation: ConsolidationResult
    output_packet: OutputPacketResult
    excluded_documents: list[ExcludedDocument]


@dataclass(frozen=True)
class RunSummary:
    results: list[FileProcessingResult]
    total_elapsed_seconds: float

    @property
    def total_usage(self) -> TokenUsage:
        usage = TokenUsage.empty()
        for result in self.results:
            usage = usage.merge(result.stats.usage)
            usage = usage.merge(result.extraction.stats.usage)
        return usage


class CategorizationService:
    def __init__(
        self,
        settings: Settings,
        provider: ClassificationProvider | None = None,
        preprocessing_service: PreprocessingService | None = None,
        extraction_service: ExtractionService | None = None,
    ) -> None:
        self._settings = settings
        self._provider = provider or create_classification_provider(settings)
        self._preprocessing_service = preprocessing_service or PreprocessingService(
            settings
        )
        self._extraction_service = extraction_service or ExtractionService(settings)

    async def process_files(self, pdf_paths: list[Path]) -> RunSummary:
        started_at = time.perf_counter()
        semaphore = asyncio.Semaphore(self._settings.worker_pool_size)

        async def _run(path: Path) -> FileProcessingResult:
            async with semaphore:
                return await self.process_file(path)

        results = await asyncio.gather(*(_run(path) for path in pdf_paths))
        return RunSummary(
            results=list(results),
            total_elapsed_seconds=time.perf_counter() - started_at,
        )

    async def process_file(self, pdf_path: Path) -> FileProcessingResult:
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {pdf_path}")

        started_at = time.perf_counter()
        outcome = await self._provider.classify_pdf(pdf_path)
        elapsed_seconds = time.perf_counter() - started_at

        classification = recover_misclassified_empty_pages(
            pdf_path,
            normalize_classification(outcome.result),
        )
        moved_pages = sorted(
            page
            for doc in outcome.result.documents
            if is_blank_page_document(doc)
            for page in doc.pages
        )
        if moved_pages:
            logger.info(
                "Moved %s blank/separator page(s) from document entries to empty_pages: %s",
                len(moved_pages),
                ", ".join(str(page) for page in moved_pages),
            )

        total_pages = get_pdf_page_count(pdf_path)
        validation = validate_classification(classification, total_pages)
        artifact_dir = write_artifacts(
            pdf_path,
            classification,
            self._settings.artifacts_dir,
        )
        classified_dir, corrected_dir, extracted_dir = artifact_paths(artifact_dir)
        preprocessing = await self._preprocessing_service.preprocess_directory(
            classified_dir,
            corrected_dir,
            classification=classification,
        )
        excluded_documents = (
            identify_debug_exclusions(classification)
            if self._settings.debug_mode
            else []
        )
        extraction = await self._extraction_service.extract_directory(
            corrected_dir,
            extracted_dir,
            classification=classification,
            debug_mode=self._settings.debug_mode,
        )
        consolidation = consolidate_extractions(extracted_dir, artifact_dir)
        output_packet = build_output_packet(
            artifact_dir=artifact_dir,
            classified_dir=classified_dir,
            consolidated_json=consolidation.output_json,
            classification=classification,
            blanks_dir=self._settings.blanks_dir,
            debug_mode=self._settings.debug_mode,
        )
        stats = ProcessingStats(
            elapsed_seconds=elapsed_seconds,
            usage=outcome.usage,
            cost=estimate_cost(outcome.usage, self._settings),
        )

        return FileProcessingResult(
            source_pdf=pdf_path,
            classification=classification,
            validation=validation,
            artifact_dir=artifact_dir,
            classified_dir=classified_dir,
            corrected_dir=corrected_dir,
            extracted_dir=extracted_dir,
            stats=stats,
            preprocessing=preprocessing,
            extraction=extraction,
            consolidation=consolidation,
            output_packet=output_packet,
            excluded_documents=excluded_documents,
        )
