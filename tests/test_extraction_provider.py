from pathlib import Path

import pytest

from dmv.extraction.prompt import build_extraction_prompt
from dmv.extraction.schemas import build_extraction_json_schema
from dmv.providers.openai_provider import OpenAIExtractionProvider
from tests.test_openai_provider import FakeAsyncOpenAI, FakeUsage


def test_build_extraction_prompt_lists_canonical_fields() -> None:
    prompt = build_extraction_prompt(
        document_type="driver_license",
        document_name="Driver License Copy",
        source_filename="Driver_License_Copy.pdf",
    )
    assert "driver_license" in prompt
    assert "driver_first_name" in prompt
    assert "omit them entirely" in prompt
    assert "Never output empty strings" in prompt
    assert "confidence" in prompt
    assert "owner_full_name" not in prompt


@pytest.mark.asyncio
async def test_openai_extraction_provider_uploads_extracts_and_deletes(
    settings,
    sample_pdf: Path,
) -> None:
    client = FakeAsyncOpenAI()

    async def extract_create(**kwargs):
        class Response:
            output_text = (
                '{"document_type":"driver_license","source_document_name":"sample.pdf",'
                '"driver_full_name":{"value":"Jane Doe","confidence":0.97}}'
            )
            usage = FakeUsage()

        client.responses.calls.append(kwargs)
        return Response()

    client.responses.create = extract_create
    provider = OpenAIExtractionProvider(settings, client=client)

    outcome = await provider.extract_pdf(
        sample_pdf,
        document_type="driver_license",
        document_name="Driver License Copy",
    )

    assert outcome.result["driver_full_name"] == {
        "value": "Jane Doe",
        "confidence": 0.97,
    }
    assert client.files.deleted == ["file-1"]
    assert client.responses.calls
    schema = client.responses.calls[0]["text"]["format"]["schema"]
    assert schema == build_extraction_json_schema("driver_license")
    assert "owner_full_name" not in schema["properties"]
    assert client.responses.calls[0]["text"]["format"]["strict"] is False
    assert client.responses.calls[0]["text"]["format"]["name"] == "document_extraction"
