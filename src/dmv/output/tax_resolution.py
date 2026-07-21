from __future__ import annotations

import re
from typing import Any

from dmv.output.values import first_value, get_consolidated_value, truthy_flag

# Markers that indicate sales tax was already remitted / satisfied by the dealer.
_PAID_MARKERS = (
    "satisfied",
    "tax paid",
    "dealer paid",
    "already paid",
    "tax included",
    "paid by dealer",
    "remitted",
)

_NUMERIC_RE = re.compile(r"[\d]+(?:[.,]\d+)?")


def money_text(raw: str | None) -> str:
    """Strip currency formatting; preserve non-numeric paid markers."""
    if not raw:
        return ""
    return raw.strip().replace("$", "").replace(",", "").strip()


def parse_money_amount(raw: str | None) -> float | None:
    """Return a positive float when ``raw`` is a numeric money amount."""
    cleaned = money_text(raw)
    if not cleaned:
        return None
    lower = cleaned.lower()
    if any(marker in lower for marker in _PAID_MARKERS):
        return None
    # Reject pure words / LEASE etc.
    match = _NUMERIC_RE.search(cleaned.replace(" ", ""))
    if not match:
        return None
    try:
        value = float(match.group(0).replace(",", ""))
    except ValueError:
        return None
    if value <= 0:
        return None
    return value


def _sales_tax_raw(data: dict[str, Any]) -> str | None:
    return first_value(data, "sales_tax", "sales_tax_amount")


def _looks_paid_text(raw: str | None) -> bool:
    if not raw:
        return False
    lower = raw.strip().lower()
    if not lower:
        return False
    # Bare "paid" is weak; require paid-context markers.
    return any(marker in lower for marker in _PAID_MARKERS)


def taxes_already_paid(data: dict[str, Any]) -> bool:
    """True when non-excluded packet evidence shows sales tax already remitted.

    Rules (conservative — unpaid/ambiguous → False):
    1. Explicit ``sales_tax_paid`` flag from extraction.
    2. ``sales_tax`` / ``sales_tax_amount`` text contains paid/satisfied markers.
    3. Check to a tax authority whose amount matches the invoice sales-tax line
       (within $0.02) — treated as remittance evidence.
    """
    flagged = truthy_flag(get_consolidated_value(data, "sales_tax_paid"))
    if flagged is True:
        return True
    if flagged is False:
        return False

    tax_raw = _sales_tax_raw(data)
    if _looks_paid_text(tax_raw):
        return True

    tax_amount = parse_money_amount(tax_raw)
    if tax_amount is None:
        # No numeric tax and no paid marker → treat as not prepaid.
        return False

    check_amount = parse_money_amount(first_value(data, "check_amount"))
    payee = (first_value(data, "payee_name") or "").lower()
    memo = (first_value(data, "check_memo") or "").lower()
    if check_amount is not None and abs(check_amount - tax_amount) <= 0.02:
        if any(token in payee for token in ("tax", "taxation", "treasury", "revenue")):
            return True
        if "sales tax" in memo or "sales/use" in memo:
            return True

    return False


def resolve_collect_taxes(data: dict[str, Any]) -> bool:
    """True = NJ DMV / customer should collect taxes; False = dealer already paid.

    Prefer an explicit ``collect_taxes`` flag when present; otherwise invert
    ``taxes_already_paid``. Stamp must always get a definite answer.
    """
    flagged = truthy_flag(get_consolidated_value(data, "collect_taxes"))
    if flagged is not None:
        return flagged
    return not taxes_already_paid(data)


def resolve_collect_lfis(data: dict[str, Any]) -> bool:
    """True = NJ DMV should collect LFIS; False otherwise.

    Prefer explicit ``collect_lfis``. Else YES when a positive unpaid LFIS /
    surcharge amount exists on non-excluded docs; default NO when absent.
    """
    flagged = truthy_flag(get_consolidated_value(data, "collect_lfis"))
    if flagged is not None:
        return flagged

    lfis_raw = first_value(data, "lfis_amount", "surcharge_amount")
    if _looks_paid_text(lfis_raw):
        return False
    amount = parse_money_amount(lfis_raw)
    return amount is not None


def tax_stamp_fields(data: dict[str, Any]) -> dict[str, str]:
    """Amounts/date/id for the UTA tax stamp (non-excluded consolidated data)."""
    purchase = money_text(
        first_value(data, "purchase_price", "gross_sales_lease_price")
    )
    # Prefer numeric display; if paid marker only, leave amount blank for stamp.
    tax_raw = _sales_tax_raw(data)
    tax_amount = parse_money_amount(tax_raw)
    sales_tax = f"{tax_amount:.2f}" if tax_amount is not None else ""
    net_sales = purchase  # blank forms usually mirror purchase / net sales amt
    stamp_date = (
        first_value(data, "sale_date", "document_date", "lease_signed_date") or ""
    )
    mv_ident = (
        first_value(
            data,
            "dealership.entity_id",
            "dealership_entity_id",
            "dealer_entity_id",
            "certificate_number",
        )
        or ""
    )
    return {
        "mv_ident": mv_ident,
        "purchase_price": purchase,
        "net_sales": net_sales,
        "sales_tax": sales_tax,
        "date": stamp_date,
        "ex_code": first_value(data, "sales_tax_exemption_code", "ex_code") or "",
    }
