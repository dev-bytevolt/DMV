from __future__ import annotations

from pathlib import Path

import cv2
import fitz
import numpy as np

RECOVERY_DPI = 150
MIN_EMPTY_PAGE_CONTENT_SCORE = 0.008


def _content_binary(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)


def _pixmap_to_gray(pixmap: fitz.Pixmap) -> np.ndarray:
    array = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height,
        pixmap.width,
        pixmap.n,
    )
    if pixmap.n == 4:
        bgr = cv2.cvtColor(array, cv2.COLOR_RGBA2BGR)
    elif pixmap.n == 3:
        bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    else:
        bgr = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def gray_content_score(gray: np.ndarray) -> float:
    binary = _content_binary(gray)
    return float(np.count_nonzero(binary) / binary.size)


def page_content_score(pdf_path: Path, page_number: int) -> float:
    with fitz.open(pdf_path) as document:
        page = document[page_number - 1]
        pixmap = page.get_pixmap(dpi=RECOVERY_DPI, alpha=False)
    return gray_content_score(_pixmap_to_gray(pixmap))


def is_blank_page(pdf_path: Path, page_number: int) -> bool:
    return page_content_score(pdf_path, page_number) < MIN_EMPTY_PAGE_CONTENT_SCORE


def is_blank_scanned_page(gray: np.ndarray) -> bool:
    if gray_content_score(gray) >= MIN_EMPTY_PAGE_CONTENT_SCORE:
        return False
    dark_ratio = float(np.mean(gray < 200))
    return dark_ratio < 0.001
