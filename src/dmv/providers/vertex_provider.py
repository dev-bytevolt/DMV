from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from dmv.config import Settings
from dmv.document_types import build_classification_prompt
from dmv.extraction.prompt import build_extraction_prompt
from dmv.extraction.schemas import build_extraction_json_schema
from dmv.models.classification import ClassificationResult
from dmv.models.usage import TokenUsage
from dmv.providers.base import (
    ClassificationOutcome,
    ClassificationProvider,
    ExtractionOutcome,
    ExtractionProvider,
)
from dmv.providers.google_schema import openai_schema_to_google
from dmv.providers.schemas import CLASSIFICATION_JSON_SCHEMA
from dmv.providers.usage import parse_vertex_usage
from dmv.utils.retry import parse_json_response, retry_async

logger = logging.getLogger(__name__)

_VERTEX_SCOPES = (
    "https://www.googleapis.com/auth/cloud-platform",
)

GenerateContentFn = Callable[..., Awaitable[Any]]


def _load_service_account_credentials(path: Path) -> Any:
    from google.oauth2 import service_account

    if not path.is_file():
        raise ValueError(f"VERTEX_SERVICE_ACCOUNT_JSON not found: {path}")
    return service_account.Credentials.from_service_account_file(
        str(path),
        scopes=list(_VERTEX_SCOPES),
    )


def _project_id_from_service_account(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    project_id = str(payload.get("project_id", "")).strip()
    if not project_id:
        raise ValueError(
            f"Service account JSON at {path} has no project_id; "
            "set VERTEX_PROJECT explicitly"
        )
    return project_id


def build_vertex_client(settings: Settings) -> Any:
    """Build a google-genai Client configured for Vertex AI."""
    from google import genai

    if settings.vertex_service_account_json is None:
        raise ValueError(
            "VERTEX_SERVICE_ACCOUNT_JSON is required when AI_PROVIDER=vertex"
        )

    credentials = _load_service_account_credentials(settings.vertex_service_account_json)
    project = settings.vertex_project.strip()
    if not project:
        project = _project_id_from_service_account(settings.vertex_service_account_json)

    return genai.Client(
        vertexai=True,
        project=project,
        location=settings.vertex_location,
        credentials=credentials,
    )


def _response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text
    raise ValueError("Vertex response did not include text output")


class VertexClassificationProvider(ClassificationProvider):
    def __init__(
        self,
        settings: Settings,
        *,
        client: Any | None = None,
        generate_content: GenerateContentFn | None = None,
    ) -> None:
        if settings.vertex_service_account_json is None and client is None:
            raise ValueError(
                "VERTEX_SERVICE_ACCOUNT_JSON is required when AI_PROVIDER=vertex"
            )

        self._settings = settings
        self._client = client
        self._generate_content = generate_content

    def _ensure_client(self) -> Any:
        if self._client is None:
            self._client = build_vertex_client(self._settings)
        return self._client

    async def classify_pdf(self, pdf_path: Path) -> ClassificationOutcome:
        logger.info("Classifying %s via Vertex", pdf_path.name)
        pdf_bytes = pdf_path.read_bytes()
        schema = openai_schema_to_google(CLASSIFICATION_JSON_SCHEMA)
        prompt = build_classification_prompt()

        async def _call() -> Any:
            return await self._generate(
                pdf_bytes=pdf_bytes,
                prompt=prompt,
                schema=schema,
                operation=f"classify {pdf_path.name}",
            )

        response = await retry_async(
            _call,
            max_retries=self._settings.max_ai_retries,
            base_delay_seconds=self._settings.ai_retry_base_delay_seconds,
            operation_name=f"Vertex generate_content (classify {pdf_path.name})",
        )
        payload = parse_json_response(_response_text(response))
        usage = parse_vertex_usage(
            getattr(response, "usage_metadata", None),
            model=self._settings.vertex_model,
        )
        logger.info(
            "Vertex classified %s into %s document(s); tokens: %s input, %s output, %s total",
            pdf_path.name,
            len(payload.get("documents", [])),
            usage.input_tokens,
            usage.output_tokens,
            usage.total_tokens,
        )
        return ClassificationOutcome(
            result=ClassificationResult.from_dict(payload),
            usage=usage,
        )

    async def _generate(
        self,
        *,
        pdf_bytes: bytes,
        prompt: str,
        schema: dict[str, Any],
        operation: str,
    ) -> Any:
        from google.genai import types

        logger.info(
            "Vertex generate_content — %s (model=%s)",
            operation,
            self._settings.vertex_model,
        )
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                    types.Part.from_text(text=prompt),
                ],
            )
        ]
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=schema,
        )
        if self._generate_content is not None:
            return await self._generate_content(
                model=self._settings.vertex_model,
                contents=contents,
                config=config,
            )

        client = self._ensure_client()
        return await client.aio.models.generate_content(
            model=self._settings.vertex_model,
            contents=contents,
            config=config,
        )


class VertexExtractionProvider(ExtractionProvider):
    def __init__(
        self,
        settings: Settings,
        *,
        client: Any | None = None,
        generate_content: GenerateContentFn | None = None,
    ) -> None:
        if settings.vertex_service_account_json is None and client is None:
            raise ValueError(
                "VERTEX_SERVICE_ACCOUNT_JSON is required when AI_PROVIDER=vertex"
            )

        self._settings = settings
        self._client = client
        self._generate_content = generate_content

    def _ensure_client(self) -> Any:
        if self._client is None:
            self._client = build_vertex_client(self._settings)
        return self._client

    async def extract_pdf(
        self,
        pdf_path: Path,
        *,
        document_type: str,
        document_name: str,
    ) -> ExtractionOutcome:
        logger.info("Extracting %s (%s) via Vertex", pdf_path.name, document_type)
        pdf_bytes = pdf_path.read_bytes()
        openai_schema = build_extraction_json_schema(document_type)
        schema = openai_schema_to_google(openai_schema)
        prompt = build_extraction_prompt(
            document_type=document_type,
            document_name=document_name,
            source_filename=pdf_path.name,
        )

        async def _call() -> Any:
            return await self._generate(
                pdf_bytes=pdf_bytes,
                prompt=prompt,
                schema=schema,
                operation=f"extract {pdf_path.name}",
            )

        response = await retry_async(
            _call,
            max_retries=self._settings.max_ai_retries,
            base_delay_seconds=self._settings.ai_retry_base_delay_seconds,
            operation_name=f"Vertex generate_content (extract {pdf_path.name})",
        )
        payload = parse_json_response(_response_text(response))
        usage = parse_vertex_usage(
            getattr(response, "usage_metadata", None),
            model=self._settings.vertex_model,
        )
        logger.info(
            "Vertex extracted %s; tokens: %s input, %s output, %s total",
            pdf_path.name,
            usage.input_tokens,
            usage.output_tokens,
            usage.total_tokens,
        )
        return ExtractionOutcome(result=payload, usage=usage)

    async def _generate(
        self,
        *,
        pdf_bytes: bytes,
        prompt: str,
        schema: dict[str, Any],
        operation: str,
    ) -> Any:
        from google.genai import types

        logger.info(
            "Vertex generate_content — %s (model=%s)",
            operation,
            self._settings.vertex_model,
        )
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                    types.Part.from_text(text=prompt),
                ],
            )
        ]
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=schema,
        )
        if self._generate_content is not None:
            return await self._generate_content(
                model=self._settings.vertex_model,
                contents=contents,
                config=config,
            )

        client = self._ensure_client()
        return await client.aio.models.generate_content(
            model=self._settings.vertex_model,
            contents=contents,
            config=config,
        )
