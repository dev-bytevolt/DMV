from __future__ import annotations

import re
import unicodedata


def sanitize_filename(name: str, *, max_length: int = 120) -> str:
    normalized = unicodedata.normalize("NFKD", name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^\w\s\-().]", "", ascii_name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.replace(" ", "_")
    if not cleaned:
        cleaned = "document"
    return cleaned[:max_length]
