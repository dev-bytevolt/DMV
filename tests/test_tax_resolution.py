from __future__ import annotations

from dmv.output.formatting import today_form_date
from dmv.output.tax_resolution import (
    parse_money_amount,
    resolve_collect_lfis,
    resolve_collect_taxes,
    tax_stamp_fields,
    taxes_already_paid,
)


def test_taxes_already_paid_from_satisfied_text() -> None:
    data = {"sales_tax": {"value": "TAX SATISFIED", "confidence": 1.0}}
    assert taxes_already_paid(data) is True
    assert resolve_collect_taxes(data) is False


def test_taxes_unpaid_numeric_means_customer_collects() -> None:
    data = {"sales_tax": {"value": "1873.53", "confidence": 1.0}}
    assert taxes_already_paid(data) is False
    assert resolve_collect_taxes(data) is True


def test_explicit_collect_taxes_overrides_amounts() -> None:
    data = {
        "collect_taxes": {"value": "no", "confidence": 1.0},
        "sales_tax": {"value": "100.00", "confidence": 1.0},
    }
    assert resolve_collect_taxes(data) is False


def test_sales_tax_paid_flag() -> None:
    data = {
        "sales_tax_paid": {"value": "yes", "confidence": 1.0},
        "sales_tax": {"value": "500.00", "confidence": 1.0},
    }
    assert taxes_already_paid(data) is True
    assert resolve_collect_taxes(data) is False


def test_check_matching_sales_tax_to_tax_authority() -> None:
    data = {
        "sales_tax": {"value": "1042.26", "confidence": 1.0},
        "check_amount": {"value": "1042.26", "confidence": 1.0},
        "payee_name": {"value": "NJ Division of Taxation", "confidence": 1.0},
    }
    assert taxes_already_paid(data) is True


def test_nj_dmv_check_does_not_count_as_tax_paid() -> None:
    data = {
        "sales_tax": {"value": "1042.26", "confidence": 1.0},
        "check_amount": {"value": "1042.26", "confidence": 1.0},
        "payee_name": {"value": "NJ DMV", "confidence": 1.0},
    }
    assert taxes_already_paid(data) is False
    assert resolve_collect_taxes(data) is True


def test_resolve_collect_lfis_from_amount() -> None:
    assert resolve_collect_lfis({"lfis_amount": {"value": "783.45"}}) is True
    assert resolve_collect_lfis({}) is False
    assert resolve_collect_lfis({"lfis_amount": {"value": "PAID"}}) is False


def test_parse_money_amount() -> None:
    assert parse_money_amount("$1,873.53") == 1873.53
    assert parse_money_amount("TAX SATISFIED") is None
    assert parse_money_amount("LEASE") is None


def test_tax_stamp_fields() -> None:
    data = {
        "purchase_price": {"value": "28279.70", "confidence": 1.0},
        "sales_tax": {"value": "1873.53", "confidence": 1.0},
        "sale_date": {"value": "06/20/2026", "confidence": 1.0},
        "dealership_entity_id": {"value": "7086161", "confidence": 1.0},
    }
    fields = tax_stamp_fields(data)
    assert fields["purchase_price"] == "28279.70"
    assert fields["sales_tax"] == "1873.53"
    assert fields["date"] == today_form_date()
    assert fields["mv_ident"] == "7086161"
