from pathlib import Path

import pytest

from dmv.categorization import FileProcessingResult, RunSummary
from dmv.cli import main, print_results
from dmv.config import Settings
from dmv.cost import estimate_cost
from dmv.models.classification import ClassificationResult
from dmv.models.usage import ProcessingStats, TokenUsage
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
    )
    artifact_dir = tmp_path / "artifacts" / "sample"
    return FileProcessingResult(
        source_pdf=sample_pdf,
        classification=classification,
        validation=validation,
        artifact_dir=artifact_dir,
        classified_dir=artifact_dir / "classified",
        corrected_dir=artifact_dir / "corrected",
        stats=ProcessingStats(
            elapsed_seconds=elapsed_seconds,
            usage=token_usage,
            cost=estimate_cost(token_usage, settings),
        ),
        preprocessing=_empty_preprocessing(tmp_path),
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
        total_elapsed_seconds=1.5,
    )

    exit_code = print_results(summary, cli_settings)
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Driver License Copy" in output
    assert "All pages are uniquely covered." in output
    assert "Document contiguity: all documents have valid page grouping." in output
    assert "Statistics:" in output
    assert "Processing time: 1.5s" in output
    assert "OpenAI tokens:" in output
    assert "Estimated cost:" in output
    assert "Preprocessing:" in output


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
        total_elapsed_seconds=1.5,
    )

    exit_code = print_results(summary, cli_settings)
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "Document contiguity issues:" in output
    assert "spans other documents on pages: 21, 22" in output


def test_cli_main_integration(monkeypatch, sample_pdf: Path, tmp_path: Path) -> None:
    async def fake_run_async(pdf_files, env_file):
        from dmv.categorization import CategorizationService
        from dmv.cli import print_results
        from dmv.config import Settings
        from dmv.models.classification import ClassificationResult
        from tests.fakes import FakeProvider

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
        )
        summary = await service.process_files(pdf_files)
        return print_results(summary, settings)

    monkeypatch.setattr("dmv.cli.run_async", fake_run_async)

    exit_code = main([str(sample_pdf)])

    assert exit_code == 0
    assert (tmp_path / "artifacts" / "sample" / "original.pdf").exists()
    assert (tmp_path / "artifacts" / "sample" / "classified").exists()
    assert (tmp_path / "artifacts" / "sample" / "corrected").exists()
