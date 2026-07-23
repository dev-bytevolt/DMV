from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from dmv.categorization import (
    CategorizationService,
    FileProcessingError,
    FileProcessingResult,
    RunSummary,
    is_already_processed,
)
from dmv.config import Settings, load_settings
from dmv.cost import estimate_cost
from dmv.debug_exclusions import ExcludedDocument
from dmv.extraction.service import ExtractionResult
from dmv.logging_config import configure_logging, format_error
from dmv.models.usage import ProcessingStats, TokenUsage
from dmv.preprocess.service import PreprocessingResult
from dmv.validation import DocumentContiguityReport, PageCoverageReport

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify pages in DMV PDF scans and split them into documents.",
    )
    parser.add_argument(
        "pdf_files",
        nargs="+",
        type=Path,
        help="One or more PDF files to classify and split",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional path to a .env file",
    )
    parser.add_argument(
        "--skip-processed",
        action="store_true",
        help=(
            "Skip PDFs that already have a completed artifacts/<name>/output.pdf "
            "(process only inputs whose artifact folders are missing or incomplete)"
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging for DMV internals (not HTTP libraries)",
    )
    return parser


def print_results(summary: RunSummary, settings: Settings) -> int:
    exit_code = 0

    if summary.skipped:
        print(f"\nSkipped already processed ({len(summary.skipped)}):")
        for path in summary.skipped:
            print(f"  - {path.name}")

    for result in summary.results:
        print(f"\nFile: {result.source_pdf}")
        print(f"Artifacts: {result.artifact_dir}")
        print(f"Classified: {result.classified_dir}")
        print(f"Corrected: {result.corrected_dir}")
        print(f"Extracted: {result.extracted_dir}")
        print(f"Consolidated: {result.consolidation.output_json}")
        print(f"Output forms: {result.output_packet.output_dir}")
        print(
            f"Output packet: {result.output_packet.output_pdf} "
            f"({result.output_packet.page_count} pages, "
            f"{result.output_packet.appended_document_count} classified docs)"
        )
        if not result.classification.documents:
            print("Documents found: none")
        else:
            print("Documents found:")
            for document in result.classification.documents:
                pages = ", ".join(str(page) for page in document.pages)
                print(
                    f"  - [{document.id}] {document.name} "
                    f"(type={document.type}, pages={pages})"
                )

        if result.classification.empty_pages:
            empty = ", ".join(str(page) for page in result.classification.empty_pages)
            print(f"Empty pages: {empty}")

        print_coverage_report(result.validation.coverage)
        print_contiguity_report(result.validation.contiguity)
        print_debug_exclusions(result.excluded_documents)
        print_preprocessing_stats(result.preprocessing)
        print_extraction_stats(result.extraction)
        print_consolidation_stats(result.consolidation)
        print_output_packet_stats(result.output_packet)
        print_processing_stats(_file_total_stats(result, settings), settings)
        if not result.validation.is_valid:
            exit_code = 1

    if summary.failures:
        exit_code = 1
        print(f"\nFailed ({len(summary.failures)}):")
        for failure in summary.failures:
            print(f"  - {failure.source_pdf.name}: {failure.error}")

    if len(summary.results) > 1 or summary.failures or summary.skipped:
        print("\nRun totals:")
        print(
            f"  Succeeded: {len(summary.results)}, "
            f"failed: {len(summary.failures)}, "
            f"skipped: {len(summary.skipped)}"
        )
        print_processing_stats(
            ProcessingStats(
                elapsed_seconds=summary.total_elapsed_seconds,
                usage=summary.total_usage,
                cost=estimate_cost(summary.total_usage, settings),
            ),
            settings,
            prefix="  ",
        )

    return exit_code


def print_coverage_report(coverage: PageCoverageReport) -> None:
    print(
        f"Page coverage: {len(coverage.covered_pages)}/{coverage.total_pages} pages "
        f"accounted for"
    )
    if coverage.missing_pages:
        missing = ", ".join(str(page) for page in coverage.missing_pages)
        print(f"  Missing pages: {missing}")
    if coverage.duplicate_pages:
        duplicate = ", ".join(str(page) for page in coverage.duplicate_pages)
        print(f"  Duplicate pages: {duplicate}")
    if coverage.invalid_pages:
        invalid = ", ".join(str(page) for page in coverage.invalid_pages)
        print(f"  Invalid pages: {invalid}")
    if coverage.is_complete:
        print("  All pages are uniquely covered.")


def print_contiguity_report(contiguity: DocumentContiguityReport) -> None:
    if contiguity.is_valid:
        print("Document contiguity: all documents have valid page grouping.")
        return

    print("Document contiguity issues:")
    for issue in contiguity.issues:
        document_pages = ", ".join(str(page) for page in issue.document_pages)
        conflicting = ", ".join(str(page) for page in issue.conflicting_pages)
        print(
            f"  - [{issue.document_id}] {issue.document_name} "
            f"(pages={document_pages}) spans other documents on pages: {conflicting}"
        )


def print_debug_exclusions(excluded_documents: list[ExcludedDocument]) -> None:
    if not excluded_documents:
        return

    print(
        f"Debug exclusions ({len(excluded_documents)}): "
        "classified and preprocessed, excluded from further processing"
    )
    for item in excluded_documents:
        pages = ", ".join(str(page) for page in item.pages)
        print(
            f"  - [{item.id}] {item.name} "
            f"(type={item.type}, pages={pages}): {item.reason}"
        )


def print_preprocessing_stats(preprocessing: PreprocessingResult, *, prefix: str = "") -> None:
    stats = preprocessing.stats
    print(f"{prefix}Preprocessing:")
    print(f"{prefix}  Time: {stats.elapsed_seconds:.1f}s")
    print(
        f"{prefix}  Documents: {stats.documents_processed}, "
        f"pages: {stats.pages_processed}"
    )


def print_extraction_stats(extraction: ExtractionResult, *, prefix: str = "") -> None:
    stats = extraction.stats
    print(f"{prefix}Extraction:")
    print(f"{prefix}  Time: {stats.elapsed_seconds:.1f}s")
    print(f"{prefix}  Documents: {stats.documents_processed}")
    if extraction.outputs:
        print(f"{prefix}  JSON files:")
        for output in extraction.outputs:
            print(
                f"{prefix}    - {output.output_json.name} "
                f"(type={output.document_type})"
            )


def print_consolidation_stats(consolidation, *, prefix: str = "") -> None:
    print(f"{prefix}Consolidation:")
    print(f"{prefix}  Fields: {consolidation.field_count}")
    if consolidation.field_count > 0:
        print(
            f"{prefix}  Review not required: "
            f"{consolidation.fields_without_review}/{consolidation.field_count} "
            f"({consolidation.review_pass_percent:.1f}%)"
        )
    else:
        print(f"{prefix}  Review not required: n/a")


def print_output_packet_stats(output_packet, *, prefix: str = "") -> None:
    print(f"{prefix}Output packet:")
    print(f"{prefix}  Forms dir: {output_packet.output_dir}")
    print(f"{prefix}  Packet PDF: {output_packet.output_pdf.name}")
    print(f"{prefix}  Pages: {output_packet.page_count}")
    print(f"{prefix}  Classified docs appended: {output_packet.appended_document_count}")


def print_processing_stats(
    stats: ProcessingStats,
    settings: Settings,
    *,
    prefix: str = "",
) -> None:
    print(f"{prefix}Statistics:")
    print(f"{prefix}  Processing time: {stats.elapsed_seconds:.1f}s")
    print(f"{prefix}  Tokens: {_format_token_usage(stats.usage)}")
    print(f"{prefix}  Estimated cost: {_format_cost(stats, settings)}")


def _file_total_stats(
    result: FileProcessingResult,
    settings: Settings,
) -> ProcessingStats:
    usage = result.stats.usage.merge(result.extraction.stats.usage)
    elapsed_seconds = (
        result.stats.elapsed_seconds
        + result.preprocessing.stats.elapsed_seconds
        + result.extraction.stats.elapsed_seconds
    )
    return ProcessingStats(
        elapsed_seconds=elapsed_seconds,
        usage=usage,
        cost=estimate_cost(usage, settings),
    )


def _format_token_usage(usage: TokenUsage) -> str:
    parts = [
        f"{usage.input_tokens:,} input",
        f"{usage.output_tokens:,} output",
        f"{usage.total_tokens:,} total",
    ]
    if usage.cached_input_tokens:
        parts.append(f"{usage.cached_input_tokens:,} cached input")
    if usage.model:
        parts.append(f"model={usage.model}")
    return ", ".join(parts)


def _format_cost(stats: ProcessingStats, settings: Settings) -> str:
    if stats.cost is None:
        model = stats.usage.model or settings.active_model
        return f"unknown for {model} (no built-in pricing for this model)"

    return (
        f"${stats.cost.amount_usd:.4f} USD "
        f"({stats.cost.pricing_label})"
    )


async def run_async(
    pdf_files: list[Path],
    env_file: Path | None,
    *,
    skip_processed: bool = False,
) -> int:
    settings = load_settings(env_file)
    if skip_processed:
        pending = [
            path
            for path in pdf_files
            if not is_already_processed(path, settings.artifacts_dir)
        ]
        if not pending:
            print(
                f"All {len(pdf_files)} input file(s) already processed "
                f"under {settings.artifacts_dir}/ — nothing to do."
            )
            return 0
    service = CategorizationService(settings)
    summary = await service.process_files(
        pdf_files,
        skip_processed=skip_processed,
    )
    return print_results(summary, settings)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)

    try:
        return asyncio.run(
            run_async(
                args.pdf_files,
                args.env_file,
                skip_processed=args.skip_processed,
            )
        )
    except KeyboardInterrupt:
        logger.error("Interrupted")
        return 130
    except FileProcessingError as exc:
        logger.error(
            "Processing failed for %s: %s",
            exc.source_pdf.name,
            format_error(exc.cause),
        )
        if args.verbose:
            logger.exception("Full traceback:")
        return 1
    except Exception as exc:
        logger.error("Processing failed: %s", format_error(exc))
        if args.verbose:
            logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
