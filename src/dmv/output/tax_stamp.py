from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz

from dmv.output.tax_resolution import resolve_collect_taxes, tax_stamp_fields

BLACK = (0, 0, 0)
WHITE = (1, 1, 1)

# Stamp between form title (~385) and state seal (~551).
# Equal white pad around a black frame that stays clear of the seal.
_WHITE_BORDER = 7.0
_BLACK_LEFT = 397.0
_BLACK_RIGHT = 526.0
_STAMP_LEFT = _BLACK_LEFT - _WHITE_BORDER
_STAMP_RIGHT = _BLACK_RIGHT + _WHITE_BORDER
_STAMP_TOP = 15.0


def apply_uta_tax_stamp(
    uta_path: Path,
    data: dict[str, Any],
    *,
    force_dealer_paid: bool | None = None,
) -> Path:
    """Draw the tax stamp on page 1 of a filled (baked) UTA PDF.

    ``force_dealer_paid`` overrides deduction (useful for visual QA of both variants).
    """
    if force_dealer_paid is None:
        dealer_paid = not resolve_collect_taxes(data)
    else:
        dealer_paid = force_dealer_paid

    fields = tax_stamp_fields(data)
    doc = fitz.open(uta_path)
    try:
        page = doc[0]
        if dealer_paid:
            _draw_dealer_paid_stamp(page, fields)
        else:
            _draw_customer_pays_stamp(page, fields)
        tmp = uta_path.with_suffix(uta_path.suffix + ".stamped")
        doc.save(str(tmp), garbage=3, deflate=True)
    finally:
        doc.close()
    tmp.replace(uta_path)
    return uta_path


def _draw_box(page: fitz.Page, outer: fitz.Rect) -> fitz.Rect:
    """White outer margin with an inset black border (equal on all sides)."""
    # Opaque white pad. Light gray hairline marks the outer edge so the white
    # border is visible even on a white page background.
    shape = page.new_shape()
    shape.draw_rect(outer)
    shape.finish(color=(0.75, 0.75, 0.75), fill=WHITE, width=0.6)
    shape.commit()
    inner = fitz.Rect(
        outer.x0 + _WHITE_BORDER,
        outer.y0 + _WHITE_BORDER,
        outer.x1 - _WHITE_BORDER,
        outer.y1 - _WHITE_BORDER,
    )
    shape = page.new_shape()
    shape.draw_rect(inner)
    shape.finish(color=BLACK, fill=WHITE, width=1.25)
    shape.commit()
    return inner


def _hline(page: fitz.Page, *, x0: float, x1: float, y: float) -> None:
    page.draw_line(fitz.Point(x0, y), fitz.Point(x1, y), color=BLACK, width=0.55)


def _row(
    page: fitz.Page,
    *,
    left: float,
    right: float,
    baseline: float,
    label: str,
    value: str,
    fontsize: float,
) -> None:
    page.insert_text(
        fitz.Point(left, baseline),
        label,
        fontsize=fontsize,
        fontname="helv",
        color=BLACK,
    )
    label_w = fitz.get_text_length(label, fontname="helv", fontsize=fontsize)
    line_x0 = left + label_w + 2.5
    _hline(page, x0=line_x0, x1=right, y=baseline + 0.8)
    if value:
        page.insert_text(
            fitz.Point(line_x0 + 2, baseline),
            value,
            fontsize=fontsize,
            fontname="helv",
            color=BLACK,
        )


def _outer_from_content_bottom(content_bottom: float) -> fitz.Rect:
    """Build outer rect so black frame ends just below the last content."""
    # content_bottom is page Y of last baseline; add descent + pad inside black,
    # then white border outside.
    inner_bottom = content_bottom + 4.0
    outer_bottom = inner_bottom + _WHITE_BORDER
    return fitz.Rect(_STAMP_LEFT, _STAMP_TOP, _STAMP_RIGHT, outer_bottom)


def _draw_dealer_paid_stamp(page: fitz.Page, fields: dict[str, str]) -> None:
    fontsize = 6.3
    header_size = 6.8
    row_gap = 11.5
    y0 = _STAMP_TOP + _WHITE_BORDER
    header_baseline = y0 + 9
    first_row = header_baseline + 10
    # 6 content rows (ident, purchase, net, tax/ex, date, signature)
    signature_baseline = first_row + 5 * row_gap
    outer = _outer_from_content_bottom(signature_baseline)
    box = _draw_box(page, outer)
    left = box.x0 + 4
    right = box.x1 - 4

    header = "N.J. SALES TAX SATISFIED"
    header_w = fitz.get_text_length(header, fontname="hebo", fontsize=header_size)
    page.insert_text(
        fitz.Point((box.x0 + box.x1 - header_w) / 2, header_baseline),
        header,
        fontsize=header_size,
        fontname="hebo",
        color=BLACK,
    )

    y = first_row
    for label, key in (
        ("M.V. Ident No. ", "mv_ident"),
        ("Purchase Price $ ", "purchase_price"),
        ("Net Sales Amt. $ ", "net_sales"),
    ):
        _row(
            page,
            left=left,
            right=right,
            baseline=y,
            label=label,
            value=fields.get(key, ""),
            fontsize=fontsize,
        )
        y += row_gap

    tax_right = left + 90
    _row(
        page,
        left=left,
        right=tax_right,
        baseline=y,
        label="Sales Tax Paid $ ",
        value=fields.get("sales_tax", ""),
        fontsize=fontsize,
    )
    ex_x = tax_right + 3
    ex_label = "Ex.Code "
    page.insert_text(
        fitz.Point(ex_x, y),
        ex_label,
        fontsize=fontsize,
        fontname="helv",
        color=BLACK,
    )
    ex_w = fitz.get_text_length(ex_label, fontname="helv", fontsize=fontsize)
    _hline(page, x0=ex_x + ex_w + 2, x1=right, y=y + 0.8)
    if fields.get("ex_code"):
        page.insert_text(
            fitz.Point(ex_x + ex_w + 3, y),
            fields["ex_code"],
            fontsize=fontsize,
            fontname="helv",
            color=BLACK,
        )
    y += row_gap

    _row(
        page,
        left=left,
        right=right,
        baseline=y,
        label="Date ",
        value=fields.get("date", ""),
        fontsize=fontsize,
    )
    y += row_gap
    _row(
        page,
        left=left,
        right=right,
        baseline=y,
        label="Dealer's Signature ",
        value="",
        fontsize=fontsize,
    )


def _draw_customer_pays_stamp(page: fitz.Page, fields: dict[str, str]) -> None:
    fontsize = 7.0
    row_gap = 12.5
    y0 = _STAMP_TOP + _WHITE_BORDER
    # top pad + 2 rows + gap to divider + initials baseline
    first_baseline = y0 + 9
    second_baseline = first_baseline + row_gap
    divider_y = second_baseline + 4
    initials_baseline = divider_y + 9
    outer = _outer_from_content_bottom(initials_baseline)
    box = _draw_box(page, outer)
    left = box.x0 + 4
    right = box.x1 - 4

    y = first_baseline
    _row(
        page,
        left=left,
        right=right,
        baseline=y,
        label="Purchase Price $ ",
        value=fields.get("purchase_price", ""),
        fontsize=fontsize,
    )
    y = second_baseline
    _row(
        page,
        left=left,
        right=right,
        baseline=y,
        label="Sales/Use Tax $ ",
        value=fields.get("sales_tax", ""),
        fontsize=fontsize,
    )

    _hline(page, x0=box.x0 + 1.0, x1=box.x1 - 1.0, y=divider_y)

    y = initials_baseline
    mid = (left + right) / 2
    page.draw_line(
        fitz.Point(mid, divider_y),
        fitz.Point(mid, y + 3),
        color=BLACK,
        width=0.55,
    )

    bottom_size = 6.8
    for x0, x1, label in (
        (left, mid - 3, "Ex. Code "),
        (mid + 3, right, "Initials "),
    ):
        page.insert_text(
            fitz.Point(x0, y),
            label,
            fontsize=bottom_size,
            fontname="helv",
            color=BLACK,
        )
        lw = fitz.get_text_length(label, fontname="helv", fontsize=bottom_size)
        _hline(page, x0=x0 + lw + 2, x1=x1, y=y + 0.8)
