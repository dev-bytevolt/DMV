import asyncio

import pytest

from dmv.utils.retry import parse_json_response, retry_async


@pytest.mark.asyncio
async def test_retry_async_succeeds_after_transient_failure() -> None:
    attempts = {"count": 0}

    async def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("temporary")
        return "ok"

    result = await retry_async(
        flaky,
        max_retries=3,
        base_delay_seconds=0.01,
        operation_name="flaky",
    )

    assert result == "ok"
    assert attempts["count"] == 3


@pytest.mark.asyncio
async def test_retry_async_raises_after_exhausting_retries() -> None:
    async def always_fail() -> None:
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError, match="nope"):
        await retry_async(
            always_fail,
            max_retries=2,
            base_delay_seconds=0.01,
            operation_name="always_fail",
        )


@pytest.mark.asyncio
async def test_retry_async_uses_longer_backoff_for_disconnects(monkeypatch) -> None:
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("dmv.utils.retry.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("dmv.utils.retry.random.uniform", lambda _a, _b: 0.0)

    class RemoteProtocolError(Exception):
        pass

    async def always_fail() -> None:
        raise RemoteProtocolError("Server disconnected without sending a response.")

    with pytest.raises(RemoteProtocolError):
        await retry_async(
            always_fail,
            max_retries=2,
            base_delay_seconds=1.0,
            operation_name="disconnect",
        )

    # base * 2^attempt * 3 for transient disconnects
    assert sleeps == [3.0, 6.0]


def test_parse_json_response() -> None:
    assert parse_json_response('{"documents": [], "empty_pages": []}') == {
        "documents": [],
        "empty_pages": [],
    }


def test_parse_json_response_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_json_response("not-json")
