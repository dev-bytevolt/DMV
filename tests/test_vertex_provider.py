from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from dmv.config import Settings
from dmv.extraction.schemas import build_extraction_json_schema
from dmv.providers.google_schema import openai_schema_to_google
from dmv.providers.openai_provider import (
    create_classification_provider,
    create_extraction_provider,
)
from dmv.providers.usage import parse_vertex_usage
from dmv.providers.vertex_provider import (
    VertexClassificationProvider,
    VertexExtractionProvider,
)


@dataclass
class FakeUsageMetadata:
    prompt_token_count: int = 1000
    candidates_token_count: int = 150
    total_token_count: int = 1150
    cached_content_token_count: int = 0


@dataclass
class FakeVertexResponse:
    text: str
    usage_metadata: FakeUsageMetadata


@pytest.fixture
def vertex_settings(tmp_path: Path) -> Settings:
    sa_path = tmp_path / "service-account.json"
    sa_path.write_text(
        '{"type":"service_account","project_id":"demo-project","client_email":"x@y.z"}',
        encoding="utf-8",
    )
    return Settings(
        ai_provider="vertex",
        openai_api_key="",
        openai_model="gpt-4o",
        worker_pool_size=2,
        max_ai_retries=2,
        ai_retry_base_delay_seconds=0.01,
        artifacts_dir=tmp_path / "artifacts",
        openai_input_price_per_million=None,
        openai_output_price_per_million=None,
        openai_cached_input_price_per_million=None,
        preprocess_dpi=200,
        debug_mode=False,
        vertex_project="demo-project",
        vertex_location="us-central1",
        vertex_model="gemini-3.1-pro-preview",
        vertex_service_account_json=sa_path,
    )


def test_parse_vertex_usage() -> None:
    usage = parse_vertex_usage(FakeUsageMetadata(), model="gemini-3.1-pro-preview")
    assert usage.input_tokens == 1000
    assert usage.output_tokens == 150
    assert usage.total_tokens == 1150
    assert usage.model == "gemini-3.1-pro-preview"


@pytest.mark.asyncio
async def test_vertex_classification_provider(
    vertex_settings: Settings,
    sample_pdf: Path,
) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_generate_content(**kwargs: Any) -> FakeVertexResponse:
        calls.append(kwargs)
        return FakeVertexResponse(
            text=(
                '{"documents": [{"id": "doc-001", "name": "Driver License Copy", '
                '"type": "driver_license", "pages": [1, 2]}], "empty_pages": [3, 4]}'
            ),
            usage_metadata=FakeUsageMetadata(),
        )

    provider = VertexClassificationProvider(
        vertex_settings,
        client=object(),
        generate_content=fake_generate_content,
    )
    outcome = await provider.classify_pdf(sample_pdf)

    assert outcome.result.documents[0].type == "driver_license"
    assert outcome.usage.total_tokens == 1150
    assert calls
    assert calls[0]["model"] == "gemini-3.1-pro-preview"
    schema = calls[0]["config"].response_json_schema
    assert "additionalProperties" not in schema
    assert schema["propertyOrdering"] == ["documents", "empty_pages"]


@pytest.mark.asyncio
async def test_vertex_extraction_provider(
    vertex_settings: Settings,
    sample_pdf: Path,
) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_generate_content(**kwargs: Any) -> FakeVertexResponse:
        calls.append(kwargs)
        return FakeVertexResponse(
            text=(
                '{"document_type":"driver_license","source_document_name":"sample.pdf",'
                '"driver_full_name":{"value":"Jane Doe","confidence":0.97}}'
            ),
            usage_metadata=FakeUsageMetadata(),
        )

    provider = VertexExtractionProvider(
        vertex_settings,
        client=object(),
        generate_content=fake_generate_content,
    )
    outcome = await provider.extract_pdf(
        sample_pdf,
        document_type="driver_license",
        document_name="Driver License Copy",
    )

    assert outcome.result["driver_full_name"]["value"] == "Jane Doe"
    expected = openai_schema_to_google(build_extraction_json_schema("driver_license"))
    assert calls[0]["config"].response_json_schema == expected


def test_create_providers_for_vertex(vertex_settings: Settings) -> None:
    classification = create_classification_provider(vertex_settings)
    extraction = create_extraction_provider(vertex_settings)
    assert isinstance(classification, VertexClassificationProvider)
    assert isinstance(extraction, VertexExtractionProvider)
