from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from dmv.config import Settings
from dmv.models.classification import ClassificationResult
from dmv.providers.base import ClassificationOutcome
from dmv.providers.openai_provider import (
    OpenAIClassificationProvider,
    create_classification_provider,
    openai_responses_create_with_retry,
)
from dmv.providers.usage import parse_openai_usage


@dataclass
class FakeUploadedFile:
    id: str


class FakeUsage:
    input_tokens = 1000
    output_tokens = 150
    total_tokens = 1150
    input_tokens_details = None


class FakeAsyncOpenAI:
    def __init__(self) -> None:
        self.files = self._Files()
        self.responses = self._Responses()

    class _Files:
        def __init__(self) -> None:
            self.created: list[tuple[str, str]] = []
            self.deleted: list[str] = []

        async def create(self, *, file, purpose: str) -> FakeUploadedFile:
            file.read()
            file_id = f"file-{len(self.created) + 1}"
            self.created.append((file_id, purpose))
            return FakeUploadedFile(id=file_id)

        async def delete(self, file_id: str) -> None:
            self.deleted.append(file_id)

    class _Responses:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def create(self, **kwargs: Any):
            self.calls.append(kwargs)

            class Response:
                output_text = (
                    '{"documents": [{"id": "doc-001", "name": "Driver License Copy", '
                    '"type": "driver_license", "pages": [1, 2]}], "empty_pages": [3, 4]}'
                )
                usage = FakeUsage()

            return Response()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        ai_provider="openai",
        openai_api_key="test-key",
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
    )


@pytest.mark.asyncio
async def test_openai_provider_uploads_classifies_and_deletes(
    settings: Settings,
    sample_pdf: Path,
) -> None:
    client = FakeAsyncOpenAI()
    provider = OpenAIClassificationProvider(settings, client=client)

    outcome = await provider.classify_pdf(sample_pdf)

    assert isinstance(outcome, ClassificationOutcome)
    assert len(outcome.result.documents) == 1
    assert outcome.result.documents[0].type == "driver_license"
    assert outcome.usage.total_tokens == 1150
    assert client.files.deleted == ["file-1"]


@pytest.mark.asyncio
async def test_openai_provider_deletes_file_when_response_json_is_invalid(
    settings: Settings,
    sample_pdf: Path,
) -> None:
    client = FakeAsyncOpenAI()

    async def invalid_json_create(**kwargs: Any):
        class Response:
            output_text = "not-json"
            usage = FakeUsage()

        return Response()

    client.responses.create = invalid_json_create
    provider = OpenAIClassificationProvider(settings, client=client)

    with pytest.raises(ValueError, match="not valid JSON"):
        await provider.classify_pdf(sample_pdf)

    assert client.files.deleted == ["file-1"]


@pytest.mark.asyncio
async def test_openai_provider_deletes_file_even_when_classification_fails(
    settings: Settings,
    sample_pdf: Path,
) -> None:
    client = FakeAsyncOpenAI()

    async def failing_create(**kwargs: Any):
        raise RuntimeError("classification failed")

    client.responses.create = failing_create
    provider = OpenAIClassificationProvider(settings, client=client)

    with pytest.raises(RuntimeError, match="classification failed"):
        await provider.classify_pdf(sample_pdf)

    assert client.files.deleted == ["file-1"]


@pytest.mark.asyncio
async def test_openai_responses_create_with_retry(settings: Settings) -> None:
    client = FakeAsyncOpenAI()
    attempts = {"count": 0}
    original_create = client.responses.create

    async def flaky_create(**kwargs: Any):
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise RuntimeError("temporary")
        return await original_create(**kwargs)

    client.responses.create = flaky_create

    response = await openai_responses_create_with_retry(
        client,
        model="gpt-4o",
        input_messages=[{"role": "user", "content": []}],
        text_format={"format": {"type": "json_schema"}},
        max_retries=2,
        base_delay_seconds=0.01,
    )

    assert "documents" in response.output_text
    assert response.usage.total_tokens == 1150
    assert attempts["count"] == 2


def test_parse_openai_usage_reads_cached_tokens() -> None:
    class Details:
        cached_tokens = 42

    class Usage:
        input_tokens = 100
        output_tokens = 20
        total_tokens = 120
        input_tokens_details = Details()

    usage = parse_openai_usage(Usage(), model="gpt-4o")

    assert usage.input_tokens == 100
    assert usage.cached_input_tokens == 42
    assert usage.model == "gpt-4o"


def test_create_classification_provider(settings: Settings) -> None:
    provider = create_classification_provider(settings)
    assert provider.__class__.__name__ == "OpenAIClassificationProvider"


def test_create_classification_provider_rejects_unknown_provider(settings: Settings) -> None:
    unknown = Settings(
        ai_provider="unknown",
        openai_api_key="x",
        openai_model="gpt-4o",
        worker_pool_size=1,
        max_ai_retries=1,
        ai_retry_base_delay_seconds=0.01,
        artifacts_dir=settings.artifacts_dir,
        openai_input_price_per_million=None,
        openai_output_price_per_million=None,
        openai_cached_input_price_per_million=None,
        preprocess_dpi=200,
        debug_mode=False,
    )

    with pytest.raises(ValueError, match="Unsupported AI provider"):
        create_classification_provider(unknown)
