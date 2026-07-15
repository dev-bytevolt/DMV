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


def test_preprocess_dpi_is_clamped(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("PREPROCESS_DPI=400\n")

    settings = load_settings(env_file)

    assert settings.preprocess_dpi == 200


def test_preprocess_dpi_minimum(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("PREPROCESS_DPI=100\n")

    settings = load_settings(env_file)

    assert settings.preprocess_dpi == 150
