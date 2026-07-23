from pathlib import Path

import pytest

from dmv.categorization import FileProcessingFailure, FileProcessingResult, RunSummary
from dmv.cli import main, print_results
from dmv.config import Settings
from dmv.consolidation.service import ConsolidationResult, CONSOLIDATED_DATA_FILENAME
from dmv.cost import estimate_cost
from dmv.debug_exclusions import ExcludedDocument
from dmv.extraction.service import ExtractionResult, ExtractionStats
from dmv.models.classification import ClassificationResult
from dmv.models.usage import ProcessingStats, TokenUsage
from dmv.output.service import OutputPacketResult
from dmv.preprocess.service import PreprocessingResult, PreprocessingStats
from dmv.validation import (
    ClassificationValidationReport,
    DocumentContiguityIssue,
    DocumentContiguityReport,
    PageCoverageReport,
    validate_classification,
)


@pytest.fixture
def cli_settings() -> Settings:
    return Settings(
        ai_provider="openai",
        openai_api_key="test-key",
        openai_model="gpt-4o",
        worker_pool_size=1,
        max_ai_retries=1,
        ai_retry_base_delay_seconds=0.01,
        artifacts_dir=Path("artifacts"),
        openai_input_price_per_million=None,
        openai_output_price_per_million=None,
        openai_cached_input_price_per_million=None,
        preprocess_dpi=200,
        debug_mode=False,
    )


def _empty_consolidation(tmp_path: Path) -> ConsolidationResult:
    artifact_dir = tmp_path / "artifacts" / "sample"
    output_json = artifact_dir / CONSOLIDATED_DATA_FILENAME
    return ConsolidationResult(
        artifact_dir=artifact_dir,
        output_json=output_json,
        field_count=0,
        fields_without_review=0,
        extra_document_count=0,
    )


def _empty_output_packet(tmp_path: Path) -> OutputPacketResult:
    artifact_dir = tmp_path / "artifacts" / "sample"
    output_dir = artifact_dir / "output"
    return OutputPacketResult(
        output_dir=output_dir,
        cover_letter_pdf=output_dir / "Cover_Letter.pdf",
        uta_pdf=output_dir / "Universal_Title_Application.pdf",
        ba49_pdf=output_dir / "Application_for_Vehicle_Registration.pdf",
        ownership_pdf=output_dir / "New_Car_Ownership.pdf",
        output_pdf=artifact_dir / "output.pdf",
        appended_document_count=0,
        page_count=0,
    )


def _empty_extraction(tmp_path: Path) -> ExtractionResult:
    artifact_dir = tmp_path / "artifacts" / "sample"
    return ExtractionResult(
        extracted_dir=artifact_dir / "extracted",
        outputs=[],
        stats=ExtractionStats(
            elapsed_seconds=0.0,
            documents_processed=0,
            usage=TokenUsage.empty(),
        ),
    )


def _empty_preprocessing(tmp_path: Path) -> PreprocessingResult:
    artifact_dir = tmp_path / "artifacts" / "sample"
    return PreprocessingResult(
        corrected_dir=artifact_dir / "corrected",
        outputs=[],
        stats=PreprocessingStats(
            elapsed_seconds=0.0,
            documents_processed=0,
            pages_processed=0,
        ),
    )


def _make_result(
    sample_pdf: Path,
    sample_classification,
    tmp_path: Path,
    *,
    validation: ClassificationValidationReport | None = None,
    usage: TokenUsage | None = None,
    elapsed_seconds: float = 1.5,
) -> FileProcessingResult:
    classification = ClassificationResult.from_dict(sample_classification)
    if validation is None:
        validation = validate_classification(classification, total_pages=4)
    token_usage = usage or TokenUsage(
        input_tokens=1000,
        output_tokens=200,
        total_tokens=1200,
        model="gpt-4o",
    )
    settings = Settings(
        ai_provider="openai",
        openai_api_key="test-key",
        openai_model="gpt-4o",
        worker_pool_size=1,
        max_ai_retries=1,
        ai_retry_base_delay_seconds=0.01,
        artifacts_dir=tmp_path / "artifacts",
        openai_input_price_per_million=None,
        openai_output_price_per_million=None,
        openai_cached_input_price_per_million=None,
        preprocess_dpi=200,
        debug_mode=False,
    )
    artifact_dir = tmp_path / "artifacts" / "sample"
    return FileProcessingResult(
        source_pdf=sample_pdf,
        classification=classification,
        validation=validation,
        artifact_dir=artifact_dir,
        classified_dir=artifact_dir / "classified",
        corrected_dir=artifact_dir / "corrected",
        extracted_dir=artifact_dir / "extracted",
        stats=ProcessingStats(
            elapsed_seconds=elapsed_seconds,
            usage=token_usage,
            cost=estimate_cost(token_usage, settings),
        ),
        preprocessing=_empty_preprocessing(tmp_path),
        extraction=_empty_extraction(tmp_path),
        consolidation=_empty_consolidation(tmp_path),
        output_packet=_empty_output_packet(tmp_path),
        excluded_documents=[],
    )


def test_print_results_reports_complete_coverage(
    sample_pdf: Path,
    sample_classification,
    tmp_path: Path,
    capsys,
    cli_settings: Settings,
) -> None:
    summary = RunSummary(
        results=[_make_result(sample_pdf, sample_classification, tmp_path)],
        failures=[],
        total_elapsed_seconds=1.5,
    )

    exit_code = print_results(summary, cli_settings)
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Driver License Copy" in output
    assert "All pages are uniquely covered." in output
    assert "Document contiguity: all documents have valid page grouping." in output
    assert "Preprocessing:" in output
    assert "Extraction:" in output
    assert "Statistics:" in output
    assert "Processing time: 1.5s" in output
    assert "Tokens:" in output
    assert "Estimated cost:" in output
    preprocessing_pos = output.index("Preprocessing:")
    extraction_pos = output.index("Extraction:")
    stats_pos = output.index("Statistics:")
    assert preprocessing_pos < extraction_pos < stats_pos


def test_print_results_reports_total_stats_after_all_steps(
    sample_pdf: Path,
    sample_classification,
    tmp_path: Path,
    capsys,
    cli_settings: Settings,
) -> None:
    preprocessing = PreprocessingResult(
        corrected_dir=tmp_path / "artifacts" / "sample" / "corrected",
        outputs=[],
        stats=PreprocessingStats(
            elapsed_seconds=2.0,
            documents_processed=2,
            pages_processed=3,
        ),
    )
    extraction = ExtractionResult(
        extracted_dir=tmp_path / "artifacts" / "sample" / "extracted",
        outputs=[],
        stats=ExtractionStats(
            elapsed_seconds=3.0,
            documents_processed=2,
            usage=TokenUsage(
                input_tokens=5000,
                output_tokens=50,
                total_tokens=5050,
                model="gpt-4o",
            ),
        ),
    )
    result = _make_result(
        sample_pdf,
        sample_classification,
        tmp_path,
        elapsed_seconds=1.0,
        usage=TokenUsage(
            input_tokens=1000,
            output_tokens=200,
            total_tokens=1200,
            model="gpt-4o",
        ),
    )
    result = FileProcessingResult(
        source_pdf=result.source_pdf,
        classification=result.classification,
        validation=result.validation,
        artifact_dir=result.artifact_dir,
        classified_dir=result.classified_dir,
        corrected_dir=result.corrected_dir,
        extracted_dir=result.extracted_dir,
        stats=result.stats,
        preprocessing=preprocessing,
        extraction=extraction,
        consolidation=_empty_consolidation(tmp_path),
        output_packet=_empty_output_packet(tmp_path),
        excluded_documents=[],
    )
    summary = RunSummary(results=[result], failures=[], total_elapsed_seconds=6.0)

    print_results(summary, cli_settings)
    output = capsys.readouterr().out

    assert "Processing time: 6.0s" in output
    assert "6,000 input" in output
    assert "250 output" in output
    assert "6,250 total" in output


def test_print_results_reports_debug_exclusions(
    sample_pdf: Path,
    tmp_path: Path,
    capsys,
) -> None:
    classification = {
        "documents": [
            {
                "id": "doc-001",
                "name": "MV Express Cover Letter to NJ DMV",
                "type": "cover_letter",
                "pages": [1],
            },
            {
                "id": "doc-002",
                "name": "Driver License Copy",
                "type": "driver_license",
                "pages": [2],
            },
        ],
        "empty_pages": [],
    }
    result = _make_result(sample_pdf, classification, tmp_path)
    result = FileProcessingResult(
        source_pdf=result.source_pdf,
        classification=result.classification,
        validation=result.validation,
        artifact_dir=result.artifact_dir,
        classified_dir=result.classified_dir,
        corrected_dir=result.corrected_dir,
        extracted_dir=result.extracted_dir,
        stats=result.stats,
        preprocessing=result.preprocessing,
        extraction=result.extraction,
        consolidation=result.consolidation,
        output_packet=result.output_packet,
        excluded_documents=[
            ExcludedDocument(
                id="doc-001",
                name="MV Express Cover Letter to NJ DMV",
                type="cover_letter",
                pages=[1],
                reason="test fixture — MV Express Cover Letter to NJ DMV is pipeline output, not real input",
            )
        ],
    )
    settings = Settings(
        ai_provider="openai",
        openai_api_key="test-key",
        openai_model="gpt-4o",
        worker_pool_size=1,
        max_ai_retries=1,
        ai_retry_base_delay_seconds=0.01,
        artifacts_dir=tmp_path / "artifacts",
        openai_input_price_per_million=None,
        openai_output_price_per_million=None,
        openai_cached_input_price_per_million=None,
        preprocess_dpi=200,
        debug_mode=True,
    )
    summary = RunSummary(results=[result], failures=[], total_elapsed_seconds=1.0)

    print_results(summary, settings)
    output = capsys.readouterr().out

    assert "Debug exclusions (1):" in output
    assert "excluded from further processing" in output
    assert "MV Express Cover Letter to NJ DMV" in output
    assert "test fixture —" in output


def test_print_results_returns_error_for_incomplete_coverage(
    sample_pdf: Path,
    sample_classification,
    tmp_path: Path,
    cli_settings: Settings,
) -> None:
    validation = ClassificationValidationReport(
        coverage=PageCoverageReport(
            total_pages=5,
            covered_pages={1, 2, 3, 4},
            missing_pages=[5],
            duplicate_pages=[],
            invalid_pages=[],
            is_complete=False,
        ),
        contiguity=DocumentContiguityReport(issues=[], is_valid=True),
    )
    summary = RunSummary(
        results=[
            _make_result(
                sample_pdf,
                sample_classification,
                tmp_path,
                validation=validation,
            )
        ],
        failures=[],
        total_elapsed_seconds=1.5,
    )

    assert print_results(summary, cli_settings) == 1


def test_print_results_returns_error_for_contiguity_issues(
    sample_pdf: Path,
    sample_classification,
    tmp_path: Path,
    capsys,
    cli_settings: Settings,
) -> None:
    validation = ClassificationValidationReport(
        coverage=PageCoverageReport(
            total_pages=4,
            covered_pages={1, 2, 3, 4},
            missing_pages=[],
            duplicate_pages=[],
            invalid_pages=[],
            is_complete=True,
        ),
        contiguity=DocumentContiguityReport(
            issues=[
                DocumentContiguityIssue(
                    document_id="doc-008",
                    document_name="Limited Power of Attorney",
                    document_pages=[15, 25],
                    conflicting_pages=[21, 22],
                )
            ],
            is_valid=False,
        ),
    )
    summary = RunSummary(
        results=[
            _make_result(
                sample_pdf,
                sample_classification,
                tmp_path,
                validation=validation,
            )
        ],
        failures=[],
        total_elapsed_seconds=1.5,
    )

    exit_code = print_results(summary, cli_settings)
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "Document contiguity issues:" in output
    assert "spans other documents on pages: 21, 22" in output


def test_print_results_lists_failed_input_files(
    sample_pdf: Path,
    sample_classification,
    tmp_path: Path,
    capsys,
    cli_settings: Settings,
) -> None:
    failed = tmp_path / "ROMANO, GINA M - NJ LEASE - LEGEND NISSAN.pdf"
    failed.write_bytes(b"%PDF")
    summary = RunSummary(
        results=[_make_result(sample_pdf, sample_classification, tmp_path)],
        failures=[
            FileProcessingFailure(
                source_pdf=failed,
                error="RemoteProtocolError: Server disconnected",
            )
        ],
        total_elapsed_seconds=2.0,
        skipped=(),
    )

    exit_code = print_results(summary, cli_settings)
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "Failed (1):" in output
    assert "ROMANO, GINA M - NJ LEASE - LEGEND NISSAN.pdf" in output
    assert "Server disconnected" in output
    assert "failed: 1" in output


def test_build_parser_accepts_skip_processed() -> None:
    from dmv.cli import build_parser

    args = build_parser().parse_args(["--skip-processed", "a.pdf"])
    assert args.skip_processed is True
    assert args.pdf_files[0].name == "a.pdf"


def test_cli_main_integration(monkeypatch, sample_pdf: Path, tmp_path: Path) -> None:
    async def fake_run_async(pdf_files, env_file, *, skip_processed: bool = False):
        from dmv.categorization import CategorizationService
        from dmv.cli import print_results
        from dmv.config import Settings
        from dmv.models.classification import ClassificationResult
        from dmv.extraction.service import ExtractionService
        from tests.fakes import FakeExtractionProvider, FakeProvider

        settings = Settings(
            ai_provider="openai",
            openai_api_key="test-key",
            openai_model="gpt-4o",
            worker_pool_size=1,
            max_ai_retries=1,
            ai_retry_base_delay_seconds=0.01,
            artifacts_dir=tmp_path / "artifacts",
            openai_input_price_per_million=None,
            openai_output_price_per_million=None,
            openai_cached_input_price_per_million=None,
            preprocess_dpi=200,
            debug_mode=False,
        )
        service = CategorizationService(
            settings,
            provider=FakeProvider(
                ClassificationResult.from_dict(
                    {
                        "documents": [
                            {
                                "id": "doc-001",
                                "name": "Driver License Copy",
                                "type": "driver_license",
                                "pages": [1, 2],
                            }
                        ],
                        "empty_pages": [3, 4],
                    }
                )
            ),
            extraction_service=ExtractionService(
                settings,
                provider=FakeExtractionProvider(),
            ),
        )
        summary = await service.process_files(
            pdf_files,
            skip_processed=skip_processed,
        )
        return print_results(summary, settings)

    monkeypatch.setattr("dmv.cli.run_async", fake_run_async)

    exit_code = main([str(sample_pdf)])

    assert exit_code == 0
    assert (tmp_path / "artifacts" / "sample" / "original.pdf").exists()
    assert (tmp_path / "artifacts" / "sample" / "classified").exists()
    assert (tmp_path / "artifacts" / "sample" / "corrected").exists()
    assert (tmp_path / "artifacts" / "sample" / "extracted").exists()
    assert (tmp_path / "artifacts" / "sample" / "output").exists()
    assert (tmp_path / "artifacts" / "sample" / "output.pdf").exists()
