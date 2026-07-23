from __future__ import annotations

import cv2
import numpy as np

from dmv.preprocess.modes import PreprocessMode
from dmv.preprocess.verification import (
    border_content_ratio,
    verify_corrected_page,
)


def _card_like_image(*, overcut: bool = False) -> np.ndarray:
    image = np.full((400, 640, 3), 245, dtype=np.uint8)
    image[40:360, 50:590] = 230
    image[80:300, 70:220] = 40  # photo block with margin
    image[60:100, 240:560] = 30  # header text band
    if overcut:
        # Push content into every border to simulate a clipped crop.
        image[:12, :] = 20
        image[-12:, :] = 20
        image[:, :12] = 20
        image[:, -12:] = 20
        image[0:200, 0:180] = 40
    return image


def test_verify_accepts_safe_card_crop() -> None:
    original = np.full((1100, 850, 3), 30, dtype=np.uint8)
    card = _card_like_image()
    original[80:480, 100:740] = card
    result = verify_corrected_page(
        original, card, mode=PreprocessMode.EMBEDDED_CARD
    )
    assert result.ok


def test_verify_rejects_overcut_card() -> None:
    original = np.full((1100, 850, 3), 30, dtype=np.uint8)
    card = _card_like_image(overcut=True)
    result = verify_corrected_page(
        original, card, mode=PreprocessMode.EMBEDDED_CARD
    )
    assert not result.ok
    assert "overcut" in result.reasons


def test_verify_rejects_sideways_page() -> None:
    page = np.full((1100, 850, 3), 255, dtype=np.uint8)
    for x in range(80, 780, 24):
        cv2.line(page, (x, 60), (x, 1040), (0, 0, 0), 2)
    result = verify_corrected_page(page, page, mode=PreprocessMode.DEFAULT)
    # Unchanged page short-circuits as ok even if sideways.
    assert result.ok

    upright = np.full((850, 1100, 3), 255, dtype=np.uint8)
    for y in range(60, 790, 24):
        cv2.line(upright, (40, y), (1060, y), (0, 0, 0), 2)
    sideways = cv2.rotate(upright, cv2.ROTATE_90_CLOCKWISE)
    result = verify_corrected_page(upright, sideways, mode=PreprocessMode.DEFAULT)
    assert not result.ok
    assert "sideways" in result.reasons


def test_border_content_ratio_detects_edge_ink() -> None:
    clean = _card_like_image()
    dirty = _card_like_image(overcut=True)
    assert border_content_ratio(dirty) > border_content_ratio(clean)
