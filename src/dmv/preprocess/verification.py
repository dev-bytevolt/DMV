from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from dmv.preprocess.modes import PreprocessMode


@dataclass(frozen=True)
class CorrectionVerification:
    ok: bool
    reasons: tuple[str, ...] = ()

    @property
    def summary(self) -> str:
        if self.ok:
            return "ok"
        return ", ".join(self.reasons) or "failed"


def verify_corrected_page(
    original: np.ndarray,
    corrected: np.ndarray,
    *,
    mode: PreprocessMode = PreprocessMode.DEFAULT,
) -> CorrectionVerification:
    """Reject corrections that destroy readable content.

    Prefer keeping the original page over an overcut or sideways result.
    """
    if corrected is original or corrected.shape == original.shape and np.array_equal(
        original, corrected
    ):
        return CorrectionVerification(ok=True)

    reasons: list[str] = []

    if _looks_sideways(corrected):
        reasons.append("sideways")

    if _looks_overcut(corrected, mode=mode):
        reasons.append("overcut")

    if mode is PreprocessMode.EMBEDDED_CARD and _looks_card_fragment(
        original, corrected
    ):
        reasons.append("card_fragment")

    if mode is PreprocessMode.FULL_PAGE_FORM and _looks_warp_damaged(
        original, corrected
    ):
        reasons.append("warp_damage")

    return CorrectionVerification(ok=not reasons, reasons=tuple(reasons))


def border_content_ratio(image: np.ndarray, *, band_ratio: float = 0.025) -> float:
    return float(np.mean(_border_content_ratios(image, band_ratio=band_ratio)))


def border_edge_density(image: np.ndarray, *, band_ratio: float = 0.025) -> float:
    return float(np.mean(_border_edge_densities(image, band_ratio=band_ratio)))


def _border_content_ratios(
    image: np.ndarray, *, band_ratio: float = 0.025
) -> tuple[float, float, float, float]:
    gray = _as_gray(image)
    height, width = gray.shape[:2]
    band = max(2, int(min(height, width) * band_ratio))
    ink = gray < 200
    strips = (
        ink[:band, :],
        ink[-band:, :],
        ink[:, :band],
        ink[:, -band:],
    )
    return tuple(float(strip.mean()) for strip in strips)


def _border_edge_densities(
    image: np.ndarray, *, band_ratio: float = 0.025
) -> tuple[float, float, float, float]:
    gray = _as_gray(image)
    height, width = gray.shape[:2]
    band = max(2, int(min(height, width) * band_ratio))
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
    strips = (
        edges[:band, :],
        edges[-band:, :],
        edges[:, :band],
        edges[:, -band:],
    )
    return tuple(float(strip.mean()) for strip in strips)


def _looks_overcut(image: np.ndarray, *, mode: PreprocessMode) -> bool:
    ink_ratios = _border_content_ratios(image)
    edge_densities = _border_edge_densities(image)
    mean_ink = float(np.mean(ink_ratios))
    mean_edge = float(np.mean(edge_densities))
    hot_borders = sum(1 for ratio in ink_ratios if ratio >= 0.18)
    max_ink = max(ink_ratios)

    # Cards/IDs: a portrait may touch one side; clipping usually heats multiple
    # borders or drives one border extremely high (cut through text/photo).
    if mode is PreprocessMode.EMBEDDED_CARD:
        return (
            hot_borders >= 2
            or max_ink >= 0.22
            or mean_ink >= 0.14
            or mean_edge >= 20.0
        )

    # Forms: allow a little scanner noise near edges; reject hard clips.
    if mode is PreprocessMode.FULL_PAGE_FORM:
        return mean_ink >= 0.18 or mean_edge >= 35.0 or hot_borders >= 3

    return mean_ink >= 0.14 or mean_edge >= 28.0 or hot_borders >= 2


def _looks_sideways(image: np.ndarray) -> bool:
    gray = _as_gray(image)
    horizontal, vertical = _axis_lengths(gray)
    if horizontal + vertical < 800.0:
        return False
    return vertical > horizontal * 1.5


def _looks_card_fragment(original: np.ndarray, corrected: np.ndarray) -> bool:
    original_area = max(original.shape[0] * original.shape[1], 1)
    corrected_area = corrected.shape[0] * corrected.shape[1]
    area_ratio = corrected_area / original_area
    # Real ID cards on letter pages are often ~4-12% of page area.
    if area_ratio < 0.025:
        return True

    aspect = max(corrected.shape[:2]) / max(min(corrected.shape[:2]), 1)
    # Near-square or extremely thin strips are usually failed card crops.
    if aspect < 1.05 or aspect > 3.2:
        return True
    return False


def _looks_warp_damaged(original: np.ndarray, corrected: np.ndarray) -> bool:
    """Detect perspective warps that bow form lines worse than the source."""
    original_score = _line_straightness(_as_gray(original))
    corrected_score = _line_straightness(_as_gray(corrected))
    if original_score <= 0.0:
        return False
    # Corrected should not get dramatically less straight.
    return corrected_score < original_score * 0.55 and border_content_ratio(corrected) >= 0.12


def _line_straightness(gray: np.ndarray) -> float:
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=80,
        minLineLength=max(min(gray.shape) // 10, 40),
        maxLineGap=16,
    )
    if lines is None:
        return 0.0

    score = 0.0
    for raw_line in lines:
        x1, y1, x2, y2 = (int(v) for v in np.asarray(raw_line).reshape(-1)[:4])
        length = float(np.hypot(x2 - x1, y2 - y1))
        angle = abs(float(np.degrees(np.arctan2(y2 - y1, x2 - x1))))
        angle = min(angle, 180.0 - angle)
        # Reward near-axis-aligned segments.
        axis_alignment = max(0.0, 1.0 - min(angle, 90.0 - angle) / 20.0)
        score += length * axis_alignment
    return score


def _axis_lengths(gray: np.ndarray) -> tuple[float, float]:
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=60,
        minLineLength=max(min(gray.shape) // 12, 30),
        maxLineGap=20,
    )
    horizontal = 0.0
    vertical = 0.0
    if lines is None:
        return horizontal, vertical

    for raw_line in lines:
        coords = np.asarray(raw_line).reshape(-1)
        if coords.size < 4:
            continue
        x1, y1, x2, y2 = coords[:4]
        angle = abs(float(np.degrees(np.arctan2(y2 - y1, x2 - x1))))
        angle = min(angle, 180.0 - angle)
        length = float(np.hypot(x2 - x1, y2 - y1))
        if angle < 20.0:
            horizontal += length
        elif angle > 70.0:
            vertical += length
    return horizontal, vertical


def _as_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
