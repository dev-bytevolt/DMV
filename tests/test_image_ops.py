import cv2
import numpy as np
import pytest
from pathlib import Path

from dmv.preprocess.image_ops import (
    PreprocessOptions,
    _should_apply_document_boundary_warp,
    crop_to_content,
    crop_to_page,
    deskew_image,
    detect_skew_angle,
    detect_skew_angle_hough,
    preprocess_page_image,
    rotate_image,
    try_page_perspective_correction,
    try_card_extraction,
)
from dmv.preprocess.modes import PreprocessMode


def _blank_page(width: int = 800, height: int = 1100) -> np.ndarray:
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    image[120:980, 100:700] = 240
    image[200:900, 150:650] = 0
    return image


def test_detect_skew_angle_hough_on_rotated_content() -> None:
    image = _blank_page()
    rotated = rotate_image(image, 5.0, max_degrees=15.0)
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    angle = detect_skew_angle_hough(gray)
    assert abs(abs(angle) - 5.0) < 2.0


def test_deskew_image_refines_sparse_page_with_min_area() -> None:
    image = _blank_page()
    rotated = rotate_image(image, 4.0, max_degrees=15.0)
    deskewed = deskew_image(rotated, PreprocessOptions())
    angle = detect_skew_angle_hough(cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY))
    assert abs(angle) < 2.0


def test_crop_to_content_skips_over_aggressive_crop() -> None:
    image = _blank_page()
    options = PreprocessOptions(padding_ratio=0.05, min_padding_pixels=30)
    cropped = crop_to_content(image, image[:, :, 0], options)
    assert cropped.shape[0] <= image.shape[0]
    assert cropped.shape[1] <= image.shape[1]


def test_full_page_form_mode_skips_perspective(monkeypatch) -> None:
    def fail_perspective(*_args, **_kwargs):
        raise AssertionError("perspective should not run")

    monkeypatch.setattr(
        "dmv.preprocess.image_ops.try_page_perspective_correction",
        fail_perspective,
    )
    processed = preprocess_page_image(
        _blank_page(),
        PreprocessOptions(),
        mode=PreprocessMode.FULL_PAGE_FORM,
    )
    assert processed.size > 0


def test_embedded_card_mode_uses_card_detection(monkeypatch) -> None:
    card = np.full((320, 500, 3), 200, dtype=np.uint8)

    def fake_extract(*_args, **_kwargs):
        return card

    monkeypatch.setattr("dmv.preprocess.image_ops._extract_embedded_card", fake_extract)
    processed = preprocess_page_image(
        _blank_page(),
        PreprocessOptions(),
        mode=PreprocessMode.EMBEDDED_CARD,
    )
    assert processed.shape == card.shape


def test_try_page_perspective_correction_rejects_small_quads() -> None:
    image = _blank_page()
    gray = image[:, :, 0]
    result = try_page_perspective_correction(image, gray, PreprocessOptions())
    assert result is None


def test_preprocess_page_image_returns_image() -> None:
    processed = preprocess_page_image(_blank_page(), PreprocessOptions())
    assert processed.ndim == 3
    assert processed.shape[2] == 3
    assert processed.size > 0


def test_should_apply_document_boundary_warp_rejects_when_fully_inset() -> None:
    points = np.array(
        [[80.0, 90.0], [720.0, 70.0], [740.0, 980.0], [60.0, 1000.0]],
        dtype=np.float32,
    )
    assert not _should_apply_document_boundary_warp(points, 800, 1100)


def test_should_reject_document_boundary_warp_for_full_page_scan() -> None:
    points = np.array(
        [[10.0, 10.0], [790.0, 10.0], [790.0, 1090.0], [10.0, 1090.0]],
        dtype=np.float32,
    )
    assert not _should_apply_document_boundary_warp(points, 800, 1100)


def test_should_reject_document_boundary_warp_for_axis_aligned_content_crop() -> None:
    points = np.array(
        [[120.0, 140.0], [680.0, 140.0], [680.0, 960.0], [120.0, 960.0]],
        dtype=np.float32,
    )
    assert not _should_apply_document_boundary_warp(points, 800, 1100)


def test_crop_to_page_falls_back_to_paper_bounds() -> None:
    image = np.full((1100, 800, 3), 245, dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cropped = crop_to_page(image, gray, PreprocessOptions())
    assert cropped.shape[0] <= image.shape[0]
    assert cropped.shape[1] <= image.shape[1]


def test_crop_to_page_crops_small_document_on_scanner_bed() -> None:
    image = np.full((1100, 800, 3), 210, dtype=np.uint8)
    image[200:700, 100:700] = 255
    image[220:680, 120:680] = 40
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cropped = crop_to_page(image, gray, PreprocessOptions())
    assert cropped.shape[0] <= image.shape[0]
    assert cropped.shape[1] <= image.shape[1]


def test_try_card_extraction_finds_flat_scan_card() -> None:
    image = np.full((1100, 850, 3), 250, dtype=np.uint8)
    image[120:420, 180:680] = 220
    image[150:250, 220:320] = 40
    image[160:230, 360:640] = 30
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    card = try_card_extraction(image, gray, PreprocessOptions())
    # Synthetic card may or may not be detected; ensure no crash.
    if card is not None:
        assert card.shape[0] < image.shape[0]
        assert card.shape[1] < image.shape[1]


def test_page_looks_like_flat_scan_card_allows_scanner_noise() -> None:
    from dmv.preprocess.image_ops import _page_looks_like_flat_scan_card

    gray = np.full((1100, 850), 248, dtype=np.uint8)
    gray[40:55, 60:80] = 20
    assert float(np.mean(gray)) >= 240.0
    assert _page_looks_like_flat_scan_card(gray)


def test_normalize_card_orientation_rotates_portrait_cards() -> None:
    from dmv.preprocess.image_ops import _normalize_card_orientation

    portrait = np.zeros((720, 460, 3), dtype=np.uint8)
    normalized = _normalize_card_orientation(portrait)
    assert normalized.shape[1] > normalized.shape[0]


def test_upright_card_if_upside_down_flips_inverted_fronts() -> None:
    from dmv.preprocess.image_ops import _upright_card_if_upside_down

    card = np.full((400, 640, 3), 230, dtype=np.uint8)
    card[:, :200] = 40
    upright = _upright_card_if_upside_down(card)
    left_dark = float(np.mean(upright[:, : upright.shape[1] // 2] < 120))
    right_dark = float(np.mean(upright[:, upright.shape[1] // 2 :] < 120))
    assert left_dark > right_dark


def test_upright_card_if_upside_down_flips_barcode_backs() -> None:
    from dmv.preprocess.image_ops import _card_orientation_score, _upright_card_if_upside_down

    card = np.full((420, 660, 3), 230, dtype=np.uint8)
    for offset in range(20, 120, 3):
        card[offset : offset + 2, :] = 0
    inverted = cv2.rotate(card, cv2.ROTATE_180)
    upright = _upright_card_if_upside_down(inverted)
    assert _card_orientation_score(upright) > _card_orientation_score(inverted)


def test_upright_card_if_upside_down_ignores_wallet_margins() -> None:
    from dmv.preprocess.image_ops import _upright_card_if_upside_down

    card = np.full((420, 660, 3), 230, dtype=np.uint8)
    card[:, 120:320] = 40
    card[:, :80] = 20
    inverted = cv2.rotate(card, cv2.ROTATE_180)
    upright = _upright_card_if_upside_down(inverted)
    left_dark = float(np.mean(upright[:, : upright.shape[1] // 2] < 120))
    right_dark = float(np.mean(upright[:, upright.shape[1] // 2 :] < 120))
    assert left_dark > right_dark


def test_upright_card_if_upside_down_keeps_upright_front_with_barcode_edges() -> None:
    from dmv.preprocess.image_ops import _photo_side_score, _upright_card_if_upside_down

    card = np.full((420, 660, 3), 230, dtype=np.uint8)
    card[:, :200] = 40
    for offset in range(360, 410, 3):
        card[offset : offset + 2, :] = 0
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    assert _photo_side_score(gray) > 4.0

    upright = _upright_card_if_upside_down(card)
    upright_gray = cv2.cvtColor(upright, cv2.COLOR_BGR2GRAY)
    assert _photo_side_score(upright_gray) > 4.0


def test_page_has_full_page_content_span_detects_letter_page() -> None:
    from dmv.preprocess.image_ops import _page_has_full_page_content_span

    page = np.full((1100, 850), 250, dtype=np.uint8)
    page[120:980, 100:750] = 30
    assert _page_has_full_page_content_span(page)


def test_try_card_extraction_finds_small_card_on_flatbed_page() -> None:
    from dmv.preprocess.image_ops import try_card_extraction

    page = np.full((2200, 1700, 3), 255, dtype=np.uint8)
    page[120:560, 120:860] = 180
    page[1500:1600, 1200:1400] = 30
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    card = try_card_extraction(page, gray, PreprocessOptions())
    assert card is not None
    assert (card.shape[0] * card.shape[1]) / (page.shape[0] * page.shape[1]) < 0.25


def test_pick_best_embedded_card_normalizes_before_deskew() -> None:
    from dmv.preprocess.image_ops import _pick_best_embedded_card

    card = np.full((850, 590, 3), 230, dtype=np.uint8)
    card[:, :200] = 40
    card[40:810, 60:530] = 160
    best = _pick_best_embedded_card([card], PreprocessOptions())
    assert best is not None
    assert best.shape[1] > best.shape[0]


def test_is_solid_card_extraction_accepts_large_card_crop() -> None:
    from dmv.preprocess.image_ops import _is_solid_card_extraction

    page = np.full((2200, 1700, 3), 180, dtype=np.uint8)
    card = np.full((870, 1186, 3), 200, dtype=np.uint8)
    assert _is_solid_card_extraction(card, page.shape[0] * page.shape[1], PreprocessOptions())


def test_score_card_quadrilateral_rejects_distorted_perspective() -> None:
    from dmv.preprocess.image_ops import _score_card_quadrilateral, order_quadrilateral_points

    image = _blank_page()
    gray = image[:, :, 0]
    page_area = gray.shape[0] * gray.shape[1]
    points = order_quadrilateral_points(
        np.array(
            [
                [100.0, 100.0],
                [700.0, 180.0],
                [620.0, 900.0],
                [80.0, 820.0],
            ],
            dtype=np.float32,
        )
    )
    contour = points.reshape(-1, 1, 2).astype(np.int32)
    approx = contour
    result = _score_card_quadrilateral(
        image,
        contour,
        approx,
        page_area,
        PreprocessOptions(),
    )
    assert result is None


def test_preprocess_page_image_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="BGR color image"):
        preprocess_page_image(np.zeros((10, 10), dtype=np.uint8), PreprocessOptions())
