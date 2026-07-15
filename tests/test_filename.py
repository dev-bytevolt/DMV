import pytest

from dmv.utils.filename import sanitize_filename


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Driver License Copy", "Driver_License_Copy"),
        ("Check #123 / Payment!", "Check_123_Payment"),
        ("   ", "document"),
        ("Universal Title Application (page 1)", "Universal_Title_Application_(page_1)"),
    ],
)
def test_sanitize_filename(raw: str, expected: str) -> None:
    assert sanitize_filename(raw) == expected
