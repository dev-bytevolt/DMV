from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from dmv.categorization import CategorizationService, FileProcessingResult, RunSummary
from dmv.config import Settings, load_settings
from dmv.cost import estimate_cost
from dmv.logging_config import configure_logging, format_error
from dmv.models.usage import ProcessingStats, TokenUsage
from dmv.preprocess.service import PreprocessingResult, PreprocessingStats
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
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging for DMV internals (not HTTP libraries)",
    )
    return parser


def print_results(summary: RunSummary, settings: Settings) -> int:
    exit_code = 0

    for result in summary.results:
        print(f"\nFile: {result.source_pdf}")
        print(f"Artifacts: {result.artifact_dir}")
        print(f"Classified: {result.classified_dir}")
        print(f"Corrected: {result.corrected_dir}")

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
        print_processing_stats(result.stats, settings)
        print_preprocessing_stats(result.preprocessing)
        if not result.validation.is_valid:
            exit_code = 1

    if len(summary.results) > 1:
        print("\nRun totals:")
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


def print_preprocessing_stats(preprocessing: PreprocessingResult, *, prefix: str = "") -> None:
    stats = preprocessing.stats
    print(f"{prefix}Preprocessing:")
    print(f"{prefix}  Time: {stats.elapsed_seconds:.1f}s")
    print(
        f"{prefix}  Documents: {stats.documents_processed}, "
        f"pages: {stats.pages_processed}"
    )


def print_processing_stats(
    stats: ProcessingStats,
    settings: Settings,
    *,
    prefix: str = "",
) -> None:
    print(f"{prefix}Statistics:")
    print(f"{prefix}  Processing time: {stats.elapsed_seconds:.1f}s")
    print(f"{prefix}  OpenAI tokens: {_format_token_usage(stats.usage)}")
    print(f"{prefix}  Estimated cost: {_format_cost(stats, settings)}")


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
        model = stats.usage.model or settings.openai_model
        return (
            f"unknown for {model} — set OPENAI_INPUT_PRICE_PER_MILLION and "
            "OPENAI_OUTPUT_PRICE_PER_MILLION in .env"
        )

    return (
        f"${stats.cost.amount_usd:.4f} USD "
        f"({stats.cost.pricing_label})"
    )


async def run_async(pdf_files: list[Path], env_file: Path | None) -> int:
    settings = load_settings(env_file)
    service = CategorizationService(settings)
    summary = await service.process_files(pdf_files)
    return print_results(summary, settings)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)

    try:
        return asyncio.run(run_async(args.pdf_files, args.env_file))
    except KeyboardInterrupt:
        logger.error("Interrupted")
        return 130
    except Exception as exc:
        logger.error("Processing failed: %s", format_error(exc))
        if args.verbose:
            logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
