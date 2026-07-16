from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from dmv.config import Settings
from dmv.models.classification import ClassificationResult
from dmv.preprocess.classification_map import build_filename_to_document_type
from dmv.preprocess.image_ops import PreprocessOptions
from dmv.preprocess.modes import preprocess_mode_for_document_type
from dmv.preprocess.pdf import PdfPreprocessResult, preprocess_pdf_file

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessingStats:
    elapsed_seconds: float
    documents_processed: int
    pages_processed: int


@dataclass(frozen=True)
class PreprocessingResult:
    corrected_dir: Path
    outputs: list[PdfPreprocessResult]
    stats: PreprocessingStats


class PreprocessingService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._options = PreprocessOptions(dpi=settings.preprocess_dpi)

    async def preprocess_directory(
        self,
        classified_dir: Path,
        corrected_dir: Path,
        *,
        classification: ClassificationResult | None = None,
    ) -> PreprocessingResult:
        classified_dir.mkdir(parents=True, exist_ok=True)
        if corrected_dir.exists():
            shutil.rmtree(corrected_dir)
        corrected_dir.mkdir(parents=True, exist_ok=True)

        pdf_files = sorted(classified_dir.glob("*.pdf"))
        if not pdf_files:
            return PreprocessingResult(
                corrected_dir=corrected_dir,
                outputs=[],
                stats=PreprocessingStats(
                    elapsed_seconds=0.0,
                    documents_processed=0,
                    pages_processed=0,
                ),
            )

        type_by_filename = (
            build_filename_to_document_type(classification)
            if classification is not None
            else {}
        )

        started_at = time.perf_counter()
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=self._settings.worker_pool_size) as executor:
            tasks = []
            for pdf_path in pdf_files:
                document_type = type_by_filename.get(pdf_path.name, "other")
                mode = preprocess_mode_for_document_type(document_type)
                tasks.append(
                    loop.run_in_executor(
                        executor,
                        partial(
                            preprocess_pdf_file,
                            pdf_path,
                            corrected_dir / pdf_path.name,
                            options=self._options,
                            document_type=document_type,
                            mode=mode,
                        ),
                    )
                )
            outputs = await asyncio.gather(*tasks)

        pages_processed = sum(result.page_count for result in outputs)
        elapsed_seconds = time.perf_counter() - started_at
        logger.info(
            "Preprocessed %s document(s), %s page(s), in %.1fs",
            len(outputs),
            pages_processed,
            elapsed_seconds,
        )

        return PreprocessingResult(
            corrected_dir=corrected_dir,
            outputs=list(outputs),
            stats=PreprocessingStats(
                elapsed_seconds=elapsed_seconds,
                documents_processed=len(outputs),
                pages_processed=pages_processed,
            ),
        )

    @staticmethod
    def load_classification(artifact_dir: Path) -> ClassificationResult | None:
        classification_path = artifact_dir / "doc_classification.json"
        if not classification_path.is_file():
            return None
        payload = json.loads(classification_path.read_text(encoding="utf-8"))
        return ClassificationResult.from_dict(payload)
