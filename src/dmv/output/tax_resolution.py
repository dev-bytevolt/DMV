from __future__ import annotations

import re
from typing import Any

from dmv.output.formatting import format_money, today_form_date
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

# NY dealer invoices in these packets compute tax at 6.625%. Cover-letter
# "Purchase Price" is the taxable basis (= sales_tax / rate), not cash total.
_NY_SALES_TAX_RATE = 0.06625
# NJ LFIS / fuel-inefficient surcharge used on the ownership forms in samples.
_LFIS_RATE = 0.004


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
    """True = NJ DMV / customer should collect taxes; False = dealer already paid."""
    flagged = truthy_flag(get_consolidated_value(data, "collect_taxes"))
    if flagged is not None:
        return flagged
    return not taxes_already_paid(data)


def gross_vehicle_price(data: dict[str, Any]) -> float | None:
    """Vehicle retail/gross used for LFIS and ownership Gross.

    Collect numeric candidates from purchase/gross fields (including variants)
    and prefer the lowest plausible vehicle-price amount — ownership forms use
    NEW CAR RETAIL / agreed value, not fee-inflated totals.
    """
    amounts: list[float] = []
    for key in ("purchase_price", "gross_sales_lease_price"):
        node = data.get(key)
        candidates: list[str] = []
        if isinstance(node, dict):
            if node.get("value"):
                candidates.append(str(node["value"]))
            for variant in node.get("variants") or []:
                if isinstance(variant, dict) and variant.get("value"):
                    candidates.append(str(variant["value"]))
        else:
            raw = first_value(data, key)
            if raw:
                candidates.append(raw)
        for candidate in candidates:
            amount = parse_money_amount(candidate)
            if amount is not None:
                amounts.append(amount)
    if not amounts:
        return None
    return min(amounts)


def compute_lfis_amount(data: dict[str, Any]) -> float | None:
    """LFIS/surcharge from extracted fields, or 0.4% of vehicle gross when absent."""
    explicit = parse_money_amount(first_value(data, "lfis_amount", "surcharge_amount"))
    if explicit is not None:
        return explicit
    gross = gross_vehicle_price(data)
    if gross is None:
        return None
    return round(gross * _LFIS_RATE, 2)


def epa_rating(data: dict[str, Any]) -> float | None:
    raw = first_value(data, "vehicle_epa_mpg_rating")
    if not raw:
        return None
    match = _NUMERIC_RE.search(raw.replace(" ", ""))
    if not match:
        return None
    try:
        value = float(match.group(0).replace(",", ""))
    except ValueError:
        return None
    if value > 150:
        return None
    return value


def _epa_rating(data: dict[str, Any]) -> float | None:
    """Deprecated alias — prefer ``epa_rating``."""
    return epa_rating(data)


def resolve_collect_lfis(data: dict[str, Any]) -> bool:
    """True = NJ DMV should collect LFIS; False otherwise.

    Prefer explicit ``collect_lfis``. Else use NJ-style thresholds when EPA is
    known: EPA < 19, or EPA < 40 with gross >= $45,000. Without EPA, YES only
    when an explicit LFIS/surcharge amount was extracted (not merely computed).
    """
    flagged = truthy_flag(get_consolidated_value(data, "collect_lfis"))
    if flagged is not None:
        return flagged
    lfis_raw = first_value(data, "lfis_amount", "surcharge_amount")
    if _looks_paid_text(lfis_raw):
        return False

    epa = epa_rating(data)
    gross = gross_vehicle_price(data)
    if epa is not None:
        if epa < 19:
            return True
        if epa < 40 and gross is not None and gross >= 45000:
            return True
        return False

    # Without EPA: luxury/gross threshold used on ownership forms in samples.
    if gross is not None and gross >= 45000:
        return True
    return parse_money_amount(lfis_raw) is not None


def cover_taxable_purchase_price(data: dict[str, Any]) -> str:
    """Cover-letter purchase price: taxable basis from sales tax when computable."""
    tax_raw = _sales_tax_raw(data)
    if _looks_paid_text(tax_raw):
        # Dealer already remitted — keep any explicit purchase price if present.
        return money_text(first_value(data, "purchase_price", "gross_sales_lease_price"))

    tax_amount = parse_money_amount(tax_raw)
    if tax_amount is not None:
        return format_money(tax_amount / _NY_SALES_TAX_RATE)

    return money_text(first_value(data, "purchase_price", "gross_sales_lease_price"))


def sales_tax_display(data: dict[str, Any]) -> str:
    """Cover / stamp sales-tax text (amount or TAX SATISFIED)."""
    tax_raw = _sales_tax_raw(data)
    if _looks_paid_text(tax_raw):
        return "TAX SATISFIED"
    amount = parse_money_amount(tax_raw)
    if amount is not None:
        return format_money(amount)
    return money_text(tax_raw)


def tax_stamp_fields(data: dict[str, Any]) -> dict[str, str]:
    """Amounts/date/id for the UTA tax stamp (non-excluded consolidated data)."""
    purchase = cover_taxable_purchase_price(data)
    tax_raw = _sales_tax_raw(data)
    tax_amount = parse_money_amount(tax_raw)
    sales_tax = format_money(tax_amount) if tax_amount is not None else ""
    net_sales = purchase
    stamp_date = today_form_date()
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
