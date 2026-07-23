from pathlib import Path

from dmv.config import Settings, load_settings


def test_load_settings_from_env(monkeypatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=test-key",
                "AI_PROVIDER=openai",
                "OPENAI_MODEL=gpt-4.1",
                "WORKER_POOL_SIZE=2",
                "MAX_AI_RETRIES=4",
                "AI_RETRY_BASE_DELAY_SECONDS=0.5",
                "ARTIFACTS_DIR=output",
            ]
        )
    )

    settings = load_settings(env_file)

    assert settings == Settings(
        ai_provider="openai",
        openai_api_key="test-key",
        openai_model="gpt-4.1",
        worker_pool_size=2,
        max_ai_retries=4,
        ai_retry_base_delay_seconds=0.5,
        artifacts_dir=Path("output"),
        openai_input_price_per_million=None,
        openai_output_price_per_million=None,
        openai_cached_input_price_per_million=None,
        preprocess_dpi=200,
        debug_mode=False,
        ai_max_concurrency=2,
        vertex_http_timeout_ms=600_000,
    )


def test_load_settings_defaults(monkeypatch, tmp_path: Path) -> None:
    for key in (
        "OPENAI_API_KEY",
        "AI_PROVIDER",
        "OPENAI_MODEL",
        "WORKER_POOL_SIZE",
        "MAX_AI_RETRIES",
        "AI_RETRY_BASE_DELAY_SECONDS",
        "ARTIFACTS_DIR",
        "OPENAI_INPUT_PRICE_PER_MILLION",
        "OPENAI_OUTPUT_PRICE_PER_MILLION",
        "OPENAI_CACHED_INPUT_PRICE_PER_MILLION",
        "PREPROCESS_DPI",
        "VERTEX_PROJECT",
        "VERTEX_LOCATION",
        "VERTEX_MODEL",
        "VERTEX_SERVICE_ACCOUNT_JSON",
    ):
        monkeypatch.delenv(key, raising=False)

    empty_env = tmp_path / "empty.env"
    empty_env.write_text("")

    settings = load_settings(empty_env)

    assert settings.ai_provider == "openai"
    assert settings.openai_model == "gpt-4o"
    assert settings.worker_pool_size == 5
    assert settings.max_ai_retries == 3
    assert settings.ai_retry_base_delay_seconds == 1.0
    assert settings.artifacts_dir == Path("artifacts")
    assert settings.preprocess_dpi == 200
    assert settings.debug_mode is False
    assert settings.vertex_project == ""
    assert settings.vertex_location == "us-central1"
    assert settings.vertex_model == "gemini-3.1-pro-preview"
    assert settings.vertex_service_account_json is None
    assert settings.active_model == "gpt-4o"
    assert settings.ai_max_concurrency == 2
    assert settings.vertex_http_timeout_ms == 600_000


def test_load_settings_vertex(tmp_path: Path) -> None:
    sa_path = tmp_path / "sa.json"
    sa_path.write_text("{}", encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "AI_PROVIDER=vertex",
                "VERTEX_PROJECT=my-project",
                "VERTEX_LOCATION=europe-west1",
                "VERTEX_MODEL=gemini-3.1-pro-preview",
                f"VERTEX_SERVICE_ACCOUNT_JSON={sa_path}",
            ]
        )
    )

    settings = load_settings(env_file)

    assert settings.ai_provider == "vertex"
    assert settings.vertex_project == "my-project"
    assert settings.vertex_location == "europe-west1"
    assert settings.vertex_model == "gemini-3.1-pro-preview"
    assert settings.vertex_service_account_json == sa_path
    assert settings.active_model == "gemini-3.1-pro-preview"


def test_load_settings_debug_mode(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("DEBUG_MODE=true\n")

    settings = load_settings(env_file)

    assert settings.debug_mode is True
    env_file = tmp_path / ".env"
    env_file.write_text("PREPROCESS_DPI=400\n")

    settings = load_settings(env_file)

    assert settings.preprocess_dpi == 200


def test_preprocess_dpi_minimum(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("PREPROCESS_DPI=100\n")

    settings = load_settings(env_file)

    assert settings.preprocess_dpi == 150
