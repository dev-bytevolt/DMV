from __future__ import annotations

from datetime import date

from dmv.consolidation.priority import normalize_vin

# Common MV retail-certificate color codes → cover-letter style names.
_COLOR_CODES: dict[str, str] = {
    "GY": "GRAY",
    "GR": "GREEN",
    "BK": "BLACK",
    "BL": "BLUE",
    "WH": "WHITE",
    "RD": "RED",
    "SL": "SILVER",
    "SV": "SILVER",
    "GL": "GOLD",
    "BR": "BROWN",
    "TN": "TAN",
    "YL": "YELLOW",
    "OR": "ORANGE",
    "PU": "PURPLE",
    "PK": "PINK",
    "BG": "BEIGE",
    "CH": "CHARCOAL",
}

__all__ = [
    "cover_vin",
    "expand_color",
    "format_currency_field",
    "format_money",
    "format_odometer_reading",
    "last_name_token",
    "normalize_vin",
    "today_form_date",
]


def today_form_date() -> str:
    """Date used on cover letter / ownership signature / tax stamp."""
    return date.today().strftime("%m/%d/%Y")


def expand_color(raw: str | None) -> str:
    if not raw:
        return ""
    cleaned = " ".join(raw.strip().split())
    code = cleaned.upper()
    if code in _COLOR_CODES:
        return _COLOR_CODES[code]
    return cleaned


def format_money(amount: float) -> str:
    return f"{amount:.2f}"


def format_currency_field(amount: float | str | None) -> str | None:
    """Ownership Step 4 style amounts: ``$ 45,863.54``."""
    if amount is None:
        return None
    if isinstance(amount, str):
        cleaned = amount.strip()
        if not cleaned:
            return None
        if cleaned.startswith("$"):
            return cleaned
        try:
            value = float(cleaned.replace(",", "").replace("$", "").strip())
        except ValueError:
            return cleaned
    else:
        value = float(amount)
    return f"$ {value:,.2f}"


def format_odometer_reading(raw: str | None) -> str | None:
    """Strip leading zeros so ``0000001`` becomes ``1``."""
    if not raw:
        return None
    cleaned = raw.strip().replace(",", "")
    if not cleaned:
        return None
    if cleaned.isdigit():
        return str(int(cleaned))
    return cleaned


def cover_vin(raw: str | None) -> str:
    """MV Express cover letters print the last 8 of the VIN, not the full VIN."""
    vin = normalize_vin(raw) or (raw or "").strip().upper()
    if not vin:
        return ""
    return vin[-8:] if len(vin) >= 8 else vin


def last_name_token(full_name: str) -> str:
    """Best-effort last name for cover-letter LESSEE short form."""
    cleaned = " ".join(full_name.replace(",", " ").split())
    if not cleaned:
        return ""
    parts = cleaned.split()
    if len(parts) == 1:
        return parts[0].upper()
    suffixes = {"JR", "SR", "II", "III", "IV", "ESQ"}
    if parts[-1].upper().rstrip(".") in suffixes and len(parts) >= 2:
        return parts[-2].upper()
    return parts[-1].upper()
