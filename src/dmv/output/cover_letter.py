from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import fitz

from dmv.output.values import first_value, get_consolidated_value, truthy_flag

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
COVER_LETTERHEAD_PATH = _ASSETS_DIR / "cover_logo.jpeg"
COVER_LOGO_PATH = COVER_LETTERHEAD_PATH

BLACK = (0, 0, 0)

# US Letter — matches the Word cover-sheet blank and filled samples.
_LETTER = fitz.paper_rect("letter")
PAGE_WIDTH = float(_LETTER.width)
PAGE_HEIGHT = float(_LETTER.height)
# Fraction of letterhead width from each side to the shared content edge
# (logo / body / footer), measured on the normalized letterhead asset.
_LETTERHEAD_PAD_FRAC = 94.0 / 1700.0
# Full-bleed content inset was ~34pt; use 1.5× that for left/right.
_CONTENT_SIDE_PAD = PAGE_WIDTH * _LETTERHEAD_PAD_FRAC * 1.5
# Outer letterhead margins: restore top/bottom; derive L/R so the visible
# content edge (after the asset's own pad) lands on _CONTENT_SIDE_PAD.
MARGIN_TOP = 36.0
MARGIN_BOTTOM = 36.0
MARGIN_LEFT = (_CONTENT_SIDE_PAD - _LETTERHEAD_PAD_FRAC * PAGE_WIDTH) / (
    1.0 - 2.0 * _LETTERHEAD_PAD_FRAC
)
MARGIN_RIGHT = MARGIN_LEFT

# Prefer real Times New Roman (matches Word blank); fall back to built-in Times.
_TIMES_DIR = Path("/System/Library/Fonts/Supplemental")
_TIMES_REGULAR = _TIMES_DIR / "Times New Roman.ttf"
_TIMES_ITALIC = _TIMES_DIR / "Times New Roman Italic.ttf"
_TIMES_BOLD_ITALIC = _TIMES_DIR / "Times New Roman Bold Italic.ttf"

FONT_BODY = "cover-times"
FONT_ITALIC = "cover-times-it"
FONT_EMPHASIS = "cover-times-bi"


def _money(raw: str | None) -> str:
    if not raw:
        return ""
    return raw.strip().replace("$", "").replace(",", "")


def cover_letter_fields(data: dict[str, Any]) -> dict[str, str]:
    cover_date = first_value(data, "cover_date") or date.today().strftime("%m/%d/%Y")
    name = first_value(
        data,
        "customer_name",
        "driver.full_name",
        "lessee.name",
        "lessee_name",
        "buyer_name",
    )
    vin = first_value(data, "vehicle_vin") or ""
    lien = first_value(data, "lien_holder", "lienholder.name", "lienholder_name") or "N/A"
    plates = first_value(data, "plate_type", "plate_number") or ""
    color = first_value(data, "vehicle_color") or ""
    purchase = _money(first_value(data, "purchase_price", "gross_sales_lease_price"))
    sales_tax = _money(first_value(data, "sales_tax", "sales_tax_amount"))
    lfis = _money(first_value(data, "lfis_amount"))
    return {
        "date": cover_date,
        "name": name or "",
        "vin": vin,
        "lien": lien,
        "plates": plates,
        "color": color,
        "purchase_price": purchase,
        "sales_tax": sales_tax,
        "lfis_amount": lfis,
    }


def resolve_collect_taxes(data: dict[str, Any]) -> bool | None:
    """Prefer explicit flag; if missing, YES when a sales-tax amount is present."""
    flagged = truthy_flag(get_consolidated_value(data, "collect_taxes"))
    if flagged is not None:
        return flagged
    sales_tax = _money(first_value(data, "sales_tax", "sales_tax_amount"))
    if sales_tax:
        return True
    return None


def resolve_collect_lfis(data: dict[str, Any]) -> bool | None:
    """Prefer explicit flag; if missing, YES when an LFIS amount is present, else NO."""
    flagged = truthy_flag(get_consolidated_value(data, "collect_lfis"))
    if flagged is not None:
        return flagged
    lfis = _money(first_value(data, "lfis_amount"))
    if lfis:
        return True
    return False


def _register_fonts(
    page: fitz.Page,
) -> tuple[str, str, str, fitz.Font | None]:
    """Return (body_font, italic_font, emphasis_font, emphasis_metrics)."""
    body = "tiro"
    italic = "tiit"
    emphasis = "tibi"
    emphasis_metrics: fitz.Font | None = None
    if _TIMES_REGULAR.is_file():
        page.insert_font(fontname=FONT_BODY, fontfile=str(_TIMES_REGULAR))
        body = FONT_BODY
    if _TIMES_ITALIC.is_file():
        page.insert_font(fontname=FONT_ITALIC, fontfile=str(_TIMES_ITALIC))
        italic = FONT_ITALIC
    if _TIMES_BOLD_ITALIC.is_file():
        page.insert_font(fontname=FONT_EMPHASIS, fontfile=str(_TIMES_BOLD_ITALIC))
        emphasis = FONT_EMPHASIS
        emphasis_metrics = fitz.Font(fontfile=str(_TIMES_BOLD_ITALIC))
    return body, italic, emphasis, emphasis_metrics


def _insert(
    page: fitz.Page,
    point: fitz.Point,
    text: str,
    *,
    fontsize: float,
    fontname: str,
    color: tuple[float, float, float] = BLACK,
) -> float:
    page.insert_text(
        point,
        text,
        fontsize=fontsize,
        fontname=fontname,
        color=color,
    )
    return point.y


def _text_width(
    text: str,
    *,
    fontname: str,
    fontsize: float,
    metrics: fitz.Font | None = None,
) -> float:
    if metrics is not None:
        return metrics.text_length(text, fontsize=fontsize)
    try:
        return fitz.get_text_length(text, fontname=fontname, fontsize=fontsize)
    except Exception:
        fallback = "tibi" if fontname == FONT_EMPHASIS else "tiro"
        return fitz.get_text_length(text, fontname=fallback, fontsize=fontsize)


def _circle_choice(
    page: fitz.Page,
    *,
    x: float,
    baseline_y: float,
    width: float,
    fontsize: float,
) -> None:
    pad_x = 8.0
    pad_top = fontsize * 0.78
    pad_bottom = 5.0
    oval = fitz.Rect(
        x - pad_x,
        baseline_y - pad_top,
        x + width + pad_x,
        baseline_y + pad_bottom,
    )
    page.draw_oval(oval, color=BLACK, width=1.35)


def _draw_yes_no(
    page: fitz.Page,
    origin: fitz.Point,
    label: str,
    flag: bool | None,
    *,
    fontsize: float,
    fontname: str,
    metrics: fitz.Font | None = None,
) -> None:
    prefix = f"{label}:   "
    _insert(page, origin, prefix, fontsize=fontsize, fontname=fontname)
    prefix_width = _text_width(
        prefix, fontname=fontname, fontsize=fontsize, metrics=metrics
    )
    yes = "YES"
    gap = "    "
    no = "NO"
    x = origin.x + prefix_width
    y = origin.y
    yes_width = _text_width(yes, fontname=fontname, fontsize=fontsize, metrics=metrics)
    gap_width = _text_width(gap, fontname=fontname, fontsize=fontsize, metrics=metrics)
    no_width = _text_width(no, fontname=fontname, fontsize=fontsize, metrics=metrics)

    _insert(page, fitz.Point(x, y), yes, fontsize=fontsize, fontname=fontname)
    no_x = x + yes_width + gap_width
    _insert(page, fitz.Point(no_x, y), no, fontsize=fontsize, fontname=fontname)

    if flag is True:
        _circle_choice(page, x=x, baseline_y=y, width=yes_width, fontsize=fontsize)
    elif flag is False:
        _circle_choice(page, x=no_x, baseline_y=y, width=no_width, fontsize=fontsize)


def build_cover_letter_pdf(
    data: dict[str, Any],
    output_path: Path,
    *,
    logo_path: Path | None = None,
) -> Path:
    """Build a US Letter cover PDF matching the MV Express cover-sheet blank."""
    fields = cover_letter_fields(data)
    collect_taxes = resolve_collect_taxes(data)
    collect_lfis = resolve_collect_lfis(data)
    letterhead = logo_path or COVER_LETTERHEAD_PATH

    doc = fitz.open()
    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    body_font, italic_font, emphasis_font, emphasis_metrics = _register_fonts(page)

    letterhead_rect = fitz.Rect(
        MARGIN_LEFT,
        MARGIN_TOP,
        PAGE_WIDTH - MARGIN_RIGHT,
        PAGE_HEIGHT - MARGIN_BOTTOM,
    )
    if letterhead.is_file():
        page.insert_image(letterhead_rect, filename=str(letterhead), keep_proportion=False)

    # Shared vertical edge: logo left = body left = footer left.
    content_pad = letterhead_rect.width * _LETTERHEAD_PAD_FRAC
    left = letterhead_rect.x0 + content_pad
    right = letterhead_rect.x1 - content_pad
    y = letterhead_rect.y0 + 96

    # Font sizes from COVER SHEET DMV.docx:
    # - body / signature: 12.5pt Times regular
    # - money + YES/NO: 18pt Times bold italic
    # - PLEASE SEND + postage: 12.5pt (bold italic / italic) — same size as body
    body_size = 12.5
    body_line_gap = 18.0
    gap_after_name = 20.0
    gap_after_intro = 30.0
    gap_before_money = 38.0
    emphasis_size = 18.0
    emphasis_line_gap = 36.0
    gap_before_send = 34.0
    closing_size = 12.5
    closing_line_gap = 18.0
    gap_before_signature = 22.0
    signature_line_gap = 16.0

    def body_line(text: str) -> None:
        nonlocal y
        _insert(page, fitz.Point(left, y), text, fontsize=body_size, fontname=body_font)
        y += body_line_gap

    def money_line(text: str) -> None:
        nonlocal y
        _insert(
            page,
            fitz.Point(left, y),
            text,
            fontsize=emphasis_size,
            fontname=emphasis_font,
        )
        y += emphasis_line_gap

    body_line("ATTN: NJ DMV")
    body_line(f"DATE: {fields['date']}")
    body_line(f"NAME: {fields['name']}")
    y += gap_after_name

    body_line("Please see the attached paperwork for Registration, Plates & Titling")
    y += gap_after_intro

    body_line(f"VIN: {fields['vin']}")
    body_line(f"LIEN: {fields['lien']}")
    body_line(f"PLATES: {fields['plates']}")
    body_line(f"COLOR: {fields['color']}")
    y += gap_before_money

    money_line(f"Purchase Price: $ {fields['purchase_price']}".rstrip())
    money_line(f"Sales Tax: $ {fields['sales_tax']}".rstrip())
    money_line(f"LFIS: $ {fields['lfis_amount']}".rstrip())
    _draw_yes_no(
        page,
        fitz.Point(left, y),
        "NJ DMV SHOULD COLLECT TAXES",
        collect_taxes,
        fontsize=emphasis_size,
        fontname=emphasis_font,
        metrics=emphasis_metrics,
    )
    y += emphasis_line_gap
    _draw_yes_no(
        page,
        fitz.Point(left, y),
        "NJ DMV SHOULD COLLECT LFIS",
        collect_lfis,
        fontsize=emphasis_size,
        fontname=emphasis_font,
        metrics=emphasis_metrics,
    )
    y += gap_before_send

    send_line = (
        "PLEASE SEND PLATES, REGISTRATION, AND ALL RETURN DOCUMENTS TO MV EXPRESS!"
    )
    # Keep PLEASE SEND + postage at the same size; nudge down only if needed to
    # stay on one line inside the content column.
    max_width = right - left
    while closing_size > 11.0 and _text_width(
        send_line,
        fontname=emphasis_font,
        fontsize=closing_size,
        metrics=emphasis_metrics,
    ) > max_width:
        closing_size -= 0.25
    _insert(
        page,
        fitz.Point(left, y),
        send_line,
        fontsize=closing_size,
        fontname=emphasis_font,
    )
    y += closing_line_gap

    def postage_line(text: str) -> None:
        nonlocal y
        # Filled sample: italic only (same point size as PLEASE SEND), not bold.
        _insert(
            page,
            fitz.Point(left, y),
            text,
            fontsize=closing_size,
            fontname=italic_font,
        )
        y += closing_line_gap

    postage_line("A postage-paid return envelope has been enclosed.")
    postage_line("If any additional information is needed, please contact me.")
    y += gap_before_signature

    def signature_line(text: str) -> None:
        nonlocal y
        _insert(page, fitz.Point(left, y), text, fontsize=body_size, fontname=body_font)
        y += signature_line_gap

    signature_line("Thank you")
    signature_line("Shloime")
    signature_line("shloime@getplatesfast.com")
    signature_line("718-687-1860 EXT 302")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    doc.close()
    return output_path
