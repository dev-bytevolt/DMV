from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from dmv.config import Settings
from dmv.document_types import build_classification_prompt
from dmv.logging_config import format_error
from dmv.models.classification import ClassificationResult
from dmv.models.usage import TokenUsage
from dmv.providers.base import ClassificationOutcome, ClassificationProvider
from dmv.providers.schemas import CLASSIFICATION_JSON_SCHEMA
from dmv.providers.usage import parse_openai_usage
from dmv.utils.retry import parse_json_response, retry_async

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenAIResponseResult:
    output_text: str
    usage: TokenUsage


async def openai_responses_create_with_retry(
    client: AsyncOpenAI,
    *,
    model: str,
    input_messages: list[dict[str, Any]],
    text_format: dict[str, Any],
    max_retries: int,
    base_delay_seconds: float,
) -> OpenAIResponseResult:
    async def _call() -> OpenAIResponseResult:
        response = await client.responses.create(
            model=model,
            input=input_messages,
            text=text_format,
        )
        return OpenAIResponseResult(
            output_text=response.output_text,
            usage=parse_openai_usage(response.usage, model=model),
        )

    return await retry_async(
        _call,
        max_retries=max_retries,
        base_delay_seconds=base_delay_seconds,
        operation_name="OpenAI POST /v1/responses",
    )


class OpenAIClassificationProvider(ClassificationProvider):
    def __init__(
        self,
        settings: Settings,
        *,
        client: AsyncOpenAI | None = None,
        upload_file: Callable[..., Any] | None = None,
        delete_file: Callable[..., Any] | None = None,
    ) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when AI_PROVIDER=openai")

        self._settings = settings
        self._client = client or AsyncOpenAI(api_key=settings.openai_api_key)
        self._upload_file = upload_file or self._client.files.create
        self._delete_file = delete_file or self._client.files.delete

    async def classify_pdf(self, pdf_path: Path) -> ClassificationOutcome:
        logger.info("Classifying %s", pdf_path.name)
        file_id: str | None = None
        try:
            file_id = await self._upload_pdf(pdf_path)
            return await self._classify_uploaded_file(file_id, pdf_path.name)
        finally:
            if file_id is not None:
                await self._delete_remote_file(file_id, pdf_name=pdf_path.name)

    async def _upload_pdf(self, pdf_path: Path) -> str:
        logger.info("OpenAI POST /v1/files — upload %s", pdf_path.name)

        async def _upload() -> str:
            with pdf_path.open("rb") as pdf_file:
                uploaded = await self._upload_file(
                    file=pdf_file,
                    purpose="user_data",
                )
            logger.info("OpenAI POST /v1/files — uploaded as %s", uploaded.id)
            return uploaded.id

        return await retry_async(
            _upload,
            max_retries=self._settings.max_ai_retries,
            base_delay_seconds=self._settings.ai_retry_base_delay_seconds,
            operation_name=f"OpenAI POST /v1/files (upload {pdf_path.name})",
        )

    async def _classify_uploaded_file(
        self,
        file_id: str,
        pdf_name: str,
    ) -> ClassificationOutcome:
        logger.info(
            "OpenAI POST /v1/responses — classify %s (model=%s, file_id=%s)",
            pdf_name,
            self._settings.openai_model,
            file_id,
        )
        input_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": file_id},
                    {
                        "type": "input_text",
                        "text": build_classification_prompt(),
                    },
                ],
            }
        ]
        text_format = {
            "format": {
                "type": "json_schema",
                "name": "doc_classification",
                "schema": CLASSIFICATION_JSON_SCHEMA,
                "strict": True,
            }
        }

        response = await openai_responses_create_with_retry(
            self._client,
            model=self._settings.openai_model,
            input_messages=input_messages,
            text_format=text_format,
            max_retries=self._settings.max_ai_retries,
            base_delay_seconds=self._settings.ai_retry_base_delay_seconds,
        )
        payload = parse_json_response(response.output_text)
        logger.info(
            "OpenAI POST /v1/responses — classified %s into %s document(s); "
            "tokens: %s input, %s output, %s total",
            pdf_name,
            len(payload.get("documents", [])),
            response.usage.input_tokens,
            response.usage.output_tokens,
            response.usage.total_tokens,
        )
        return ClassificationOutcome(
            result=ClassificationResult.from_dict(payload),
            usage=response.usage,
        )

    async def _delete_remote_file(self, file_id: str, *, pdf_name: str) -> None:
        logger.info("OpenAI DELETE /v1/files/%s — cleanup for %s", file_id, pdf_name)

        async def _delete() -> None:
            await self._delete_file(file_id)
            logger.info("OpenAI DELETE /v1/files/%s — deleted", file_id)

        try:
            await retry_async(
                _delete,
                max_retries=self._settings.max_ai_retries,
                base_delay_seconds=self._settings.ai_retry_base_delay_seconds,
                operation_name=f"OpenAI DELETE /v1/files/{file_id}",
            )
        except Exception as exc:
            logger.error(
                "OpenAI DELETE /v1/files/%s failed after retries for %s: %s",
                file_id,
                pdf_name,
                format_error(exc),
            )
            logger.error(
                "Sensitive data may remain on the provider for file_id=%s",
                file_id,
            )


def create_classification_provider(settings: Settings) -> ClassificationProvider:
    if settings.ai_provider == "openai":
        return OpenAIClassificationProvider(settings)

    raise ValueError(
        f"Unsupported AI provider: {settings.ai_provider!r}. "
        "Supported providers: openai"
    )
