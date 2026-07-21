from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _optional_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    return float(raw)


def _optional_path(name: str) -> Path | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    return Path(raw.strip()).expanduser()


def _debug_mode() -> bool:
    raw = os.getenv("DEBUG_MODE", "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    ai_provider: str
    openai_api_key: str
    openai_model: str
    worker_pool_size: int
    max_ai_retries: int
    ai_retry_base_delay_seconds: float
    artifacts_dir: Path
    openai_input_price_per_million: float | None
    openai_output_price_per_million: float | None
    openai_cached_input_price_per_million: float | None
    preprocess_dpi: int
    debug_mode: bool
    vertex_project: str = ""
    vertex_location: str = "us-central1"
    vertex_model: str = "gemini-3.1-pro-preview"
    vertex_service_account_json: Path | None = None
    blanks_dir: Path = Path("artifacts/blanks")

    @property
    def active_model(self) -> str:
        if self.ai_provider == "vertex":
            return self.vertex_model
        return self.openai_model


def _preprocess_dpi() -> int:
    raw = int(os.getenv("PREPROCESS_DPI", "200"))
    return max(150, min(200, raw))


def load_settings(env_path: Path | None = None) -> Settings:
    if env_path is not None:
        load_dotenv(env_path, override=True)
    else:
        load_dotenv()

    return Settings(
        ai_provider=os.getenv("AI_PROVIDER", "openai").lower(),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        worker_pool_size=int(os.getenv("WORKER_POOL_SIZE", "5")),
        max_ai_retries=int(os.getenv("MAX_AI_RETRIES", "3")),
        ai_retry_base_delay_seconds=float(
            os.getenv("AI_RETRY_BASE_DELAY_SECONDS", "1.0")
        ),
        artifacts_dir=Path(os.getenv("ARTIFACTS_DIR", "artifacts")),
        openai_input_price_per_million=_optional_float("OPENAI_INPUT_PRICE_PER_MILLION"),
        openai_output_price_per_million=_optional_float(
            "OPENAI_OUTPUT_PRICE_PER_MILLION"
        ),
        openai_cached_input_price_per_million=_optional_float(
            "OPENAI_CACHED_INPUT_PRICE_PER_MILLION"
        ),
        preprocess_dpi=_preprocess_dpi(),
        debug_mode=_debug_mode(),
        vertex_project=os.getenv("VERTEX_PROJECT", "").strip(),
        vertex_location=os.getenv("VERTEX_LOCATION", "us-central1").strip()
        or "us-central1",
        vertex_model=os.getenv("VERTEX_MODEL", "gemini-3.1-pro-preview").strip()
        or "gemini-3.1-pro-preview",
        vertex_service_account_json=_optional_path("VERTEX_SERVICE_ACCOUNT_JSON"),
        blanks_dir=Path(os.getenv("BLANKS_DIR", "artifacts/blanks")),
    )
