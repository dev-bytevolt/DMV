from __future__ import annotations

from typing import Any


def get_consolidated_value(data: dict[str, Any], path: str) -> str | None:
    """Resolve a dotted path into a consolidated field ``value``.

    Supports nested entity groups (``owner.full_name``, ``lienholder.address.street``)
    and flat top-level fields (``vehicle_vin``).
    """
    if not path:
        return None
    node: Any = data
    for part in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
        if node is None:
            return None
    return _leaf_value(node)


def _leaf_value(node: Any) -> str | None:
    if isinstance(node, dict) and "value" in node:
        value = str(node.get("value", "")).strip()
        return value or None
    if isinstance(node, str):
        value = node.strip()
        return value or None
    return None


def first_value(data: dict[str, Any], *paths: str) -> str | None:
    """Return the first non-empty value among ``paths``."""
    for path in paths:
        value = get_consolidated_value(data, path)
        if value:
            return value
    return None


def truthy_flag(raw: str | None) -> bool | None:
    """Interpret YES/NO style consolidated flags. ``None`` if unknown."""
    if raw is None:
        return None
    normalized = raw.strip().lower()
    if normalized in {"yes", "y", "true", "1", "x", "checked"}:
        return True
    if normalized in {"no", "n", "false", "0", "unchecked"}:
        return False
    return None
