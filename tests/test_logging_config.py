import logging

from dmv.logging_config import NOISY_LOGGER_NAMES, configure_logging, format_error


def test_configure_logging_suppresses_noisy_loggers() -> None:
    configure_logging(verbose=True)

    for logger_name in NOISY_LOGGER_NAMES:
        assert logging.getLogger(logger_name).level == logging.WARNING

    assert logging.getLogger("dmv").level == logging.DEBUG


def test_configure_logging_default_is_info_for_dmv() -> None:
    configure_logging(verbose=False)

    assert logging.getLogger("dmv").level == logging.INFO


def test_format_error_includes_exception_message() -> None:
    error = RuntimeError("something went wrong")
    assert "RuntimeError: something went wrong" in format_error(error)


def test_format_error_includes_status_and_body() -> None:
    class FakeAPIError(Exception):
        status_code = 429
        code = "rate_limit_exceeded"
        body = {"error": {"message": "Rate limit reached"}}

    formatted = format_error(FakeAPIError("Too Many Requests"))
    assert "status=429" in formatted
    assert "code=rate_limit_exceeded" in formatted
    assert "Rate limit reached" in formatted
