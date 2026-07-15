from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from dmv.preprocess.modes import PreprocessMode


@dataclass(frozen=True)
class PreprocessOptions:
    dpi: int = 200
    padding_ratio: float = 0.03
    min_padding_pixels: int = 24
    max_skew_degrees: float = 15.0
    min_crop_margin_ratio: float = 0.02
    min_retained_area_ratio: float = 0.85
    min_page_quad_area_ratio: float = 0.88
    max_page_quad_area_ratio: float = 0.995
    min_card_area_ratio: float = 0.05
    max_card_area_ratio: float = 0.45
    min_card_aspect_ratio: float = 1.15
    max_card_aspect_ratio: float = 2.1
    max_card_perspective_distortion: float = 0.12
    card_quad_padding_ratio: float = 0.20
    min_document_boundary_area_ratio: float = 0.30
    max_document_boundary_area_ratio: float = 0.92
    max_deskew_passes: int = 4


def preprocess_page_image(
    image: np.ndarray,
    options: PreprocessOptions,
    *,
    mode: PreprocessMode = PreprocessMode.DEFAULT,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a BGR color image")

    if mode is PreprocessMode.EMBEDDED_CARD:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        card = _extract_embedded_card(image, gray, options)
        if card is not None:
            return card

        working = deskew_image(image, options)
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        return crop_to_content(working, gray, options)

    working = deskew_image(image, options)
    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)

    if mode is PreprocessMode.FULL_PAGE_FORM:
        boundary = try_document_boundary_warp(working, gray, options)
        if boundary is not None:
            working = boundary
            gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        return crop_to_page(working, gray, options)

    perspective = try_page_perspective_correction(working, gray, options)
    if perspective is not None:
        working = perspective
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)

    return crop_to_content(working, gray, options)


def _card_deskew_options(options: PreprocessOptions) -> PreprocessOptions:
    return PreprocessOptions(
        dpi=options.dpi,
        max_deskew_passes=2,
        max_skew_degrees=options.max_skew_degrees,
    )


def _extract_embedded_card(
    image: np.ndarray,
    gray: np.ndarray,
    options: PreprocessOptions,
) -> np.ndarray | None:
    candidates: list[np.ndarray] = []
    page_area = gray.shape[0] * gray.shape[1]

    raw_card = try_card_extraction(image, gray, options)
    if raw_card is not None:
        candidates.append(raw_card)

    if raw_card is None or not _is_solid_card_extraction(raw_card, page_area, options):
        deskewed_page = deskew_image(image, options)
        deskewed_gray = cv2.cvtColor(deskewed_page, cv2.COLOR_BGR2GRAY)
        deskewed_card = try_card_extraction(deskewed_page, deskewed_gray, options)
        if deskewed_card is not None:
            candidates.append(deskewed_card)

    if not candidates:
        return None
    return _pick_best_embedded_card(candidates, options)


def _pick_best_embedded_card(
    candidates: list[np.ndarray],
    options: PreprocessOptions,
) -> np.ndarray | None:
    best_score = float("-inf")
    best_card: np.ndarray | None = None
    card_options = _card_deskew_options(options)
    scored: list[tuple[float, np.ndarray]] = []

    for candidate in candidates:
        candidate_gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(candidate_gray))
        if mean_brightness < 100.0:
            continue

        deskewed = _upright_card_if_upside_down(
            deskew_image(_normalize_card_orientation(candidate), card_options)
        )
        deskewed = _tighten_wallet_card_crop(deskewed, options)
        deskewed_gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)
        if not _is_balanced_card_crop(deskewed_gray):
            continue
        residual_skew = abs(detect_skew_angle_hough(deskewed_gray))
        aspect_ratio = _card_aspect_ratio(deskewed)
        if not options.min_card_aspect_ratio <= aspect_ratio <= options.max_card_aspect_ratio:
            continue

        score = (
            -residual_skew * 10.0
            - abs(aspect_ratio - 1.58)
            + mean_brightness * 0.01
        )
        scored.append((score, deskewed))

    if not scored:
        return None

    max_area = max(item[1].shape[0] * item[1].shape[1] for item in scored)
    for score, deskewed in scored:
        area_ratio = (deskewed.shape[0] * deskewed.shape[1]) / max(max_area, 1)
        adjusted = score
        if area_ratio < 0.65:
            adjusted -= 12.0
        if adjusted <= best_score:
            continue
        best_score = adjusted
        best_card = deskewed

    return best_card


def deskew_image(image: np.ndarray, options: PreprocessOptions) -> np.ndarray:
    working = image.copy()
    first_pass_used_min_area = False
    for pass_index in range(options.max_deskew_passes):
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        if pass_index == 0:
            angle = _skew_correction_for_pass(gray, pass_index)
            hough_angle = detect_skew_angle_hough(gray)
            first_pass_used_min_area = abs(hough_angle) < 0.3 and abs(angle) >= 0.25
        else:
            angle = _refine_skew_correction_angle(gray, first_pass_used_min_area)
        min_angle = 0.25 if pass_index == 0 else 0.3
        if abs(angle) < min_angle:
            break
        working = rotate_image(working, angle, max_degrees=options.max_skew_degrees)
    return working


def detect_skew_angle(gray: np.ndarray) -> float:
    return _skew_correction_for_pass(gray, pass_index=0)


def _refine_skew_correction_angle(
    gray: np.ndarray,
    first_pass_used_min_area: bool,
) -> float:
    hough_angle = detect_skew_angle_hough(gray)
    min_area_angle = detect_skew_angle_min_area(gray)

    if first_pass_used_min_area and abs(hough_angle) >= 0.3:
        return hough_angle

    if abs(min_area_angle) < 0.4:
        return 0.0
    if abs(hough_angle) < 0.3:
        return 0.0
    if np.sign(hough_angle) != np.sign(min_area_angle):
        return 0.0
    return min_area_angle


def _skew_correction_for_pass(gray: np.ndarray, pass_index: int) -> float:
    hough_angle = detect_skew_angle_hough(gray)
    min_area_angle = detect_skew_angle_min_area(gray)

    if abs(hough_angle) >= 0.3:
        return hough_angle
    if abs(min_area_angle) >= 0.25:
        if abs(hough_angle) < 0.3 and abs(min_area_angle) > 6.0:
            return 0.0
        return min_area_angle
    return 0.0


def detect_skew_angle_hough(gray: np.ndarray) -> float:
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    min_line_length = max(gray.shape[1] // 8, 60)
    best_angle = 0.0
    best_count = 0

    for threshold in (80, 100, 120):
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=threshold,
            minLineLength=min_line_length,
            maxLineGap=25,
        )
        if lines is None:
            continue

        angles: list[float] = []
        for raw_line in lines:
            coords = np.asarray(raw_line).reshape(-1)
            if coords.size < 4:
                continue
            x1, y1, x2, y2 = coords[:4]
            delta_x = x2 - x1
            if delta_x == 0:
                continue
            angle = float(np.degrees(np.arctan2(y2 - y1, delta_x)))
            if angle < -45:
                angle += 90
            elif angle > 45:
                angle -= 90
            if abs(angle) <= 15:
                angles.append(angle)

        if len(angles) <= best_count:
            continue
        best_count = len(angles)
        best_angle = float(np.median(angles))

    return best_angle


def detect_skew_angle_min_area(gray: np.ndarray) -> float:
    binary = _content_binary(gray)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 100:
        return 0.0

    height, width = gray.shape[:2]
    if len(coords) > height * width * 0.35:
        return 0.0

    _, _, angle = cv2.minAreaRect(coords.astype(np.float32))
    if abs(angle) > 60.0:
        return 0.0
    if angle < -45:
        angle = 90 + angle
    return float(-angle)


def rotate_image(
    image: np.ndarray,
    angle_degrees: float,
    *,
    max_degrees: float,
) -> np.ndarray:
    clamped = max(-max_degrees, min(max_degrees, angle_degrees))
    if abs(clamped) < 0.3:
        return image

    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, clamped, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def try_page_perspective_correction(
    image: np.ndarray,
    gray: np.ndarray,
    options: PreprocessOptions,
) -> np.ndarray | None:
    height, width = gray.shape[:2]
    page_area = height * width
    if page_area <= 0:
        return None

    quads = _find_quadrilaterals(gray)
    for contour, approx in quads:
        area = cv2.contourArea(contour)
        area_ratio = area / page_area
        if area_ratio < options.min_page_quad_area_ratio:
            continue
        if area_ratio > options.max_page_quad_area_ratio:
            continue

        points = order_quadrilateral_points(approx.reshape(4, 2).astype(np.float32))
        return warp_quadrilateral(image, points)

    return None


def try_document_boundary_warp(
    image: np.ndarray,
    gray: np.ndarray,
    options: PreprocessOptions,
) -> np.ndarray | None:
    height, width = gray.shape[:2]
    page_area = height * width
    if page_area <= 0:
        return None

    binary = _content_binary(gray)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(contour) / page_area
    if area_ratio < options.min_document_boundary_area_ratio:
        return None
    if area_ratio > options.max_document_boundary_area_ratio:
        return None

    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    for epsilon_ratio in (0.005, 0.01, 0.015, 0.02, 0.03):
        approx = cv2.approxPolyDP(hull, epsilon_ratio * perimeter, True)
        if len(approx) != 4:
            continue
        points = order_quadrilateral_points(approx.reshape(4, 2).astype(np.float32))
        if not _should_apply_document_boundary_warp(points, width, height):
            continue
        return warp_quadrilateral(image, points)
    return None


def _should_apply_document_boundary_warp(
    points: np.ndarray,
    width: int,
    height: int,
) -> bool:
    xs = points[:, 0]
    ys = points[:, 1]
    insets = (
        float(xs.min()),
        float(width - 1 - xs.max()),
        float(ys.min()),
        float(height - 1 - ys.max()),
    )
    min_inset = min(insets)
    max_inset = max(insets)
    min_dim = min(width, height)

    # Only warp perspective captures where the page is anchored to a scanner
    # edge on one side but inset on the opposite side. Flat full-page scans
    # should rely on deskew + crop instead.
    if min_inset >= 0.02 * min_dim:
        return False
    if max_inset < 0.18 * min_dim:
        return False
    if _quad_parallel_side_difference(points) < 0.02:
        return False
    return True


def _quad_parallel_side_difference(points: np.ndarray) -> float:
    ordered = order_quadrilateral_points(points)
    top = float(np.linalg.norm(ordered[1] - ordered[0]))
    bottom = float(np.linalg.norm(ordered[2] - ordered[3]))
    left = float(np.linalg.norm(ordered[3] - ordered[0]))
    right = float(np.linalg.norm(ordered[2] - ordered[1]))
    return max(
        abs(top - bottom) / max(top, bottom, 1.0),
        abs(left - right) / max(left, right, 1.0),
    )


def try_card_extraction(
    image: np.ndarray,
    gray: np.ndarray,
    options: PreprocessOptions,
) -> np.ndarray | None:
    height, width = gray.shape[:2]
    page_area = height * width
    if page_area <= 0:
        return None

    if _page_looks_like_embedded_photo(gray):
        candidates: list[tuple[float, np.ndarray, int]] = []

        luminance_card = _find_card_via_luminance_blob(image, gray, options)
        if luminance_card is not None:
            candidates.append((*luminance_card, 3))

        for contour, approx in _find_quadrilaterals(gray):
            candidate = _score_card_quadrilateral(
                image,
                contour,
                approx,
                page_area,
                options,
            )
            if candidate is not None:
                candidates.append((*candidate, 1))

        upper_region_card = _find_card_via_upper_region(image, gray, options)
        if upper_region_card is not None:
            candidates.append((*upper_region_card, 2))

        edge_projection_card = _find_card_via_edge_projection(image, gray, options)
        if edge_projection_card is not None:
            candidates.append((*edge_projection_card, 2))

        scanner_bed_card = _find_card_via_scanner_bed_content(image, gray, options)
        if scanner_bed_card is not None:
            candidates.append((*scanner_bed_card, 3))

        selected = _select_card_extraction_candidate(candidates)
        if selected is not None:
            return selected

    if _page_looks_like_flat_scan_card(gray):
        flat_card = _find_flat_scan_card(image, gray, options)
        if flat_card is not None:
            card = flat_card[1]
            card_area_ratio = (card.shape[0] * card.shape[1]) / page_area
            if card_area_ratio <= 0.25:
                return card
        if (
            float(np.mean(gray)) >= 240.0
            and _page_has_full_page_content_span(gray)
        ):
            return None
        if flat_card is not None:
            return flat_card[1]

    return None


def _page_has_full_page_content_span(gray: np.ndarray) -> bool:
    binary = _content_binary(gray)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) == 0:
        return False

    height, width = gray.shape[:2]
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    span_y = (y_max - y_min) / max(height - 1, 1)
    span_x = (x_max - x_min) / max(width - 1, 1)
    return span_y > 0.65 and span_x > 0.70


def _is_solid_card_extraction(
    card: np.ndarray,
    page_area: float,
    options: PreprocessOptions,
) -> bool:
    card_gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    if not _is_balanced_card_crop(card_gray):
        return False
    area_ratio = (card.shape[0] * card.shape[1]) / max(page_area, 1)
    if area_ratio < 0.04:
        return False
    aspect_ratio = _card_aspect_ratio(card)
    return options.min_card_aspect_ratio <= aspect_ratio <= options.max_card_aspect_ratio


def _page_looks_like_embedded_photo(gray: np.ndarray) -> bool:
    return float(np.mean(gray < 90)) >= 0.04


def _page_looks_like_flat_scan_card(gray: np.ndarray) -> bool:
    return float(np.mean(gray)) >= 240.0 and float(np.mean(gray < 90)) < 0.04


def _card_aspect_ratio(image: np.ndarray) -> float:
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return 0.0
    return float(max(width, height) / min(width, height))


def _normalize_card_orientation(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    if height > width * 1.05:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def _card_barcode_edge_score(gray: np.ndarray) -> tuple[float, float, float]:
    height, width = gray.shape[:2]
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
    top_edge = float(edges[: max(int(height * 0.42), 1), :].mean())
    bottom_edge = float(edges[int(height * 0.58) :, :].mean())
    return top_edge - bottom_edge, top_edge, bottom_edge


def _photo_side_score(gray: np.ndarray) -> float:
    height, width = gray.shape[:2]
    x1 = int(width * 0.12)
    x2 = int(width * 0.88)
    y1 = int(height * 0.08)
    y2 = int(height * 0.92)
    if x2 <= x1 or y2 <= y1:
        roi = gray
    else:
        roi = gray[y1:y2, x1:x2]
    mid = max(roi.shape[1] // 2, 1)
    left_dark = float(np.mean(roi[:, :mid] < 120))
    right_dark = float(np.mean(roi[:, mid:] < 120))
    return (left_dark - right_dark) * 40.0


def _strong_barcode_back_signal(gray: np.ndarray, photo_score: float) -> bool:
    barcode_score, top_edge, bottom_edge = _card_barcode_edge_score(gray)
    if max(top_edge, bottom_edge) <= 18.0 or abs(barcode_score) <= 12.0:
        return False
    if abs(photo_score) >= 5.0 and photo_score * barcode_score < 0:
        return False
    return True


def _should_trust_photo_orientation(gray: np.ndarray, photo_score: float) -> bool:
    return abs(photo_score) >= 4.0 and not _strong_barcode_back_signal(gray, photo_score)


def _card_orientation_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    barcode_score, top_edge, bottom_edge = _card_barcode_edge_score(gray)
    photo_score = _photo_side_score(gray)
    if _strong_barcode_back_signal(gray, photo_score):
        return barcode_score * 2.0 + photo_score * 0.15
    return photo_score * 1.5 + barcode_score * 0.5


def _trim_uniform_scanner_margins(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    if height < 120 or width < 120:
        return image

    row_mean = gray.mean(axis=1)
    row_std = np.array([gray[max(0, y - 2) : y + 3].std() for y in range(height)])
    content_rows = np.where((row_mean < 250.0) | (row_std > 5.0))[0]
    if content_rows.size == 0:
        return image

    pad = 8
    y1 = max(0, int(content_rows[0]) - pad)
    y2 = min(height, int(content_rows[-1]) + pad)

    col_mean = gray.mean(axis=0)
    col_std = np.array([gray[:, max(0, x - 2) : x + 3].std() for x in range(width)])
    content_cols = np.where((col_mean < 250.0) | (col_std > 5.0))[0]
    if content_cols.size == 0:
        return image[y1:y2, :]

    x1 = max(0, int(content_cols[0]) - pad)
    x2 = min(width, int(content_cols[-1]) + pad)
    trimmed = image[y1:y2, x1:x2]
    if trimmed.shape[0] < 80 or trimmed.shape[1] < 80:
        return image
    trimmed_ratio = (trimmed.shape[0] * trimmed.shape[1]) / max(height * width, 1)
    if trimmed_ratio > 0.98:
        return image
    return trimmed


def _upright_card_if_upside_down(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flipped = cv2.rotate(image, cv2.ROTATE_180)
    flipped_gray = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)

    photo_score = _photo_side_score(gray)
    flipped_photo = _photo_side_score(flipped_gray)
    if _should_trust_photo_orientation(gray, photo_score):
        if flipped_photo > photo_score + 1.0:
            image = flipped
        return _trim_embedded_card_margins(image)

    score = _card_orientation_score(image)
    flipped_score = _card_orientation_score(flipped)
    if flipped_score > score + 1.0:
        image = flipped
    return _trim_embedded_card_margins(image)


def _trim_leading_blank_band(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    if height < 120 or width < 120:
        return image

    row_mean = gray.mean(axis=1)
    row_std = np.array([gray[max(0, y - 2) : y + 3].std() for y in range(height)])
    top = 0
    while top < height and row_mean[top] > 252.0 and row_std[top] < 4.0:
        top += 1
    if top < 8 or top > int(height * 0.35):
        return image

    pad = 6
    trimmed = image[max(0, top - pad) :, :]
    if trimmed.shape[0] < 80:
        return image
    return trimmed


def _trim_embedded_card_margins(image: np.ndarray) -> np.ndarray:
    trimmed = _trim_leading_blank_band(image)
    return _trim_uniform_scanner_margins(trimmed)


def _tighten_wallet_card_crop(
    image: np.ndarray,
    options: PreprocessOptions,
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    if height * width < 1_000_000 or height < 900:
        return image

    barcode_score, top_edge, bottom_edge = _card_barcode_edge_score(gray)
    if max(top_edge, bottom_edge) > 18.0 and abs(barcode_score) > 8.0:
        return image

    band_height = max(int(height * 0.06), 4)
    header_y = 0
    header_dark = 0.0
    for y in range(int(height * 0.04), int(height * 0.72), 2):
        dark = float(np.mean(gray[y : y + band_height, :] < 100))
        if dark > 0.12 and dark > header_dark:
            header_dark = dark
            header_y = y

    if header_y <= 0:
        return image

    pad = max(options.min_padding_pixels, 24)
    top = max(0, header_y - int(height * 0.08))
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    row_edges = edges.sum(axis=1)
    col_edges = edges.sum(axis=0)
    if row_edges.max() <= 0 or col_edges.max() <= 0:
        return image

    rows = np.where(row_edges > row_edges.max() * 0.10)[0]
    cols = np.where(col_edges > col_edges.max() * 0.10)[0]
    if rows.size == 0 or cols.size == 0:
        return image

    bottom = min(height, int(rows[-1]) + pad)
    left = max(0, int(cols[0]) - pad)
    right = min(width, int(cols[-1]) + pad)
    if bottom - top < 300 or right - left < 450:
        return image

    cropped = image[top:bottom, left:right]
    cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    if not _is_balanced_card_crop(cropped_gray):
        return image
    aspect_ratio = _card_aspect_ratio(cropped)
    if not options.min_card_aspect_ratio <= aspect_ratio <= options.max_card_aspect_ratio:
        return image
    if (bottom - top) * (right - left) > height * width * 0.96:
        return image
    return cropped


def _is_balanced_card_crop(gray: np.ndarray) -> bool:
    height, width = gray.shape[:2]
    if height < 80 or width < 80:
        return False

    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
    third = max(height // 3, 1)
    bands = [
        float(edges[i * third : (i + 1) * third, :].mean())
        for i in range(3)
    ]
    if max(bands) <= 0:
        return True
    if bands[2] <= 0:
        return bands[0] > 0 or bands[1] > 0
    if bands[0] < bands[2] * 0.2 and bands[1] < bands[2] * 0.25:
        return False
    # Content crushed into the top band with almost no mid-band content.
    if bands[0] > bands[2] * 4.0 and bands[1] < bands[0] * 0.20:
        return False
    return True


def _expand_quadrilateral_points(
    points: np.ndarray,
    ratio: float,
    width: int,
    height: int,
) -> np.ndarray:
    center = points.mean(axis=0)
    expanded = center + (points - center) * (1.0 + ratio)
    expanded[:, 0] = np.clip(expanded[:, 0], 0, width - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, height - 1)
    return expanded.astype(np.float32)


def _dark_bottom_crop_height(gray: np.ndarray, threshold: float = 120.0, margin: int = 20) -> int:
    height = gray.shape[0]
    row_mean = np.mean(gray, axis=1)
    dark_rows = np.where(row_mean < threshold)[0]
    if dark_rows.size == 0:
        return height
    crop_height = int(dark_rows[0]) - margin
    return max(int(height * 0.5), min(crop_height, int(height * 0.85)))


def _select_card_extraction_candidate(
    candidates: list[tuple[float, np.ndarray, int]],
) -> np.ndarray | None:
    if not candidates:
        return None

    method_boost = {3: 3.0, 2: 1.5, 1: 1.0}

    def adjusted_score(item: tuple[float, np.ndarray, int]) -> float:
        score, warped, priority = item
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        if not _is_balanced_card_crop(warped_gray):
            return float("-inf")
        height, width = warped_gray.shape[:2]
        # Prefer compact card crops over large scanner-bed rectangles.
        area_penalty = 1.0
        if height * width > 1_500_000:
            area_penalty = 0.35
        elif height * width > 1_000_000:
            area_penalty = 0.65
        return score * method_boost.get(priority, 1.0) * area_penalty

    return max(candidates, key=adjusted_score)[1]


def _find_flat_scan_card(
    image: np.ndarray,
    gray: np.ndarray,
    options: PreprocessOptions,
) -> tuple[float, np.ndarray] | None:
    height, width = gray.shape[:2]
    page_area = height * width

    blur = cv2.GaussianBlur(gray.astype(np.float32), (15, 15), 0)
    squared = cv2.GaussianBlur((gray.astype(np.float32)) ** 2, (15, 15), 0)
    local_std = np.sqrt(np.maximum(squared - blur**2, 0))

    best_score = 0.0
    best_warp: np.ndarray | None = None

    for std_threshold in (4.0, 6.0, 8.0, 10.0, 12.0):
        mask = (local_std > std_threshold).astype(np.uint8) * 255
        closed = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
            iterations=2,
        )
        opened = cv2.morphologyEx(
            closed,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
            iterations=1,
        )
        contours, _ = cv2.findContours(
            opened,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
            candidate = _score_flat_scan_card_contour(
                image,
                contour,
                page_area,
                height,
                options,
            )
            if candidate is None:
                continue
            score, warped = candidate
            if score <= best_score:
                continue
            best_score = score
            best_warp = warped

    if best_warp is None:
        return None
    return best_score, best_warp


def _score_flat_scan_card_contour(
    image: np.ndarray,
    contour: np.ndarray,
    page_area: float,
    page_height: int,
    options: PreprocessOptions,
) -> tuple[float, np.ndarray] | None:
    area = cv2.contourArea(contour)
    area_ratio = area / page_area
    if area_ratio < 0.02 or area_ratio > 0.25:
        return None

    rect = cv2.minAreaRect(contour)
    box_width, box_height = rect[1]
    if min(box_width, box_height) < 80:
        return None

    aspect_ratio = max(box_width, box_height) / min(box_width, box_height)
    if not options.min_card_aspect_ratio <= aspect_ratio <= options.max_card_aspect_ratio:
        return None

    _, _, bbox_width, bbox_height = cv2.boundingRect(contour)
    center_y = cv2.boundingRect(contour)[1] + bbox_height / 2
    if center_y > page_height * 0.55:
        return None

    points = cv2.boxPoints(rect).astype(np.float32)
    ordered = order_quadrilateral_points(points)
    image_height, image_width = image.shape[:2]
    ordered = _expand_quadrilateral_points(
        ordered,
        options.card_quad_padding_ratio,
        image_width,
        image_height,
    )
    aspect_score = 1.0 - min(abs(aspect_ratio - 1.58) / 0.6, 1.0)
    area_score = 1.0 - min(abs(area_ratio - 0.08) / 0.12, 1.0)
    position_score = 1.0 - min(center_y / max(page_height * 0.55, 1.0), 1.0)
    score = area_ratio * max(aspect_score, 0.2) * max(area_score, 0.2) * max(position_score, 0.15)

    warped = warp_quadrilateral(image, ordered)
    if warped.shape[0] < 80 or warped.shape[1] < 80:
        return None
    return score, warped


def _score_card_quadrilateral(
    image: np.ndarray,
    contour: np.ndarray,
    approx: np.ndarray,
    page_area: float,
    options: PreprocessOptions,
) -> tuple[float, np.ndarray] | None:
    area = cv2.contourArea(contour)
    area_ratio = area / page_area
    if area_ratio < options.min_card_area_ratio:
        return None
    if area_ratio > options.max_card_area_ratio:
        return None
    if area_ratio > 0.30:
        return None

    points = approx.reshape(4, 2).astype(np.float32)
    aspect_ratio = _quadrilateral_aspect_ratio(points)
    if not options.min_card_aspect_ratio <= aspect_ratio <= options.max_card_aspect_ratio:
        return None

    ordered = order_quadrilateral_points(points)
    if _quad_parallel_side_difference(ordered) > options.max_card_perspective_distortion:
        return None

    rectangularity = area / max(_quad_bounding_area(ordered), 1.0)
    aspect_score = 1.0 - min(abs(aspect_ratio - 1.58) / 0.6, 1.0)
    area_score = 1.0 - min(abs(area_ratio - 0.18) / 0.2, 1.0)
    score = area_ratio * rectangularity * max(aspect_score, 0.2) * max(area_score, 0.2)
    score = min(score, 0.05)

    warped = warp_quadrilateral(image, ordered)
    if warped.shape[0] < 80 or warped.shape[1] < 80:
        return None
    warped_area_ratio = (warped.shape[0] * warped.shape[1]) / page_area
    if warped_area_ratio > 0.35:
        return None
    return score, warped


def _find_card_via_luminance_blob(
    image: np.ndarray,
    gray: np.ndarray,
    options: PreprocessOptions,
) -> tuple[float, np.ndarray] | None:
    height, width = gray.shape[:2]
    page_area = height * width
    page_mean = float(np.mean(gray))
    dark_ratio = float(np.mean(gray < 90))
    # Mid-gray scanner-bed pages are handled by scanner-bed content detection.
    if page_mean >= 155.0 and dark_ratio < 0.20:
        return None

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    luminance = lab[:, :, 0]

    dark_mask = (luminance < 100).astype(np.uint8) * 255
    dark_mask = cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21)),
        iterations=2,
    )
    dark_contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not dark_contours:
        return None

    wallet = max(dark_contours, key=cv2.contourArea)
    wallet_x, wallet_y, wallet_w, wallet_h = cv2.boundingRect(wallet)
    if wallet_w < 80 or wallet_h < 80:
        return None

    if wallet_y > height * 0.45:
        roi_y = 0
        roi_h = max(int(height * 0.68), 1)
        roi_bgr = image[roi_y : roi_y + roi_h, :]
        roi_luminance = luminance[roi_y : roi_y + roi_h, :]
        wallet_x = 0
        wallet_y = 0
        wallet_w = width
        wallet_h = roi_h
        upper_limit = max(int(roi_h * 0.68), 1)
    else:
        roi_bgr = image[wallet_y : wallet_y + wallet_h, wallet_x : wallet_x + wallet_w]
        roi_luminance = luminance[wallet_y : wallet_y + wallet_h, wallet_x : wallet_x + wallet_w]
        upper_limit = max(int(wallet_h * 0.68), 1)

    roi_saturation = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]

    best_score = 0.0
    best_warp: np.ndarray | None = None

    mask_sources: list[np.ndarray] = []
    for threshold in range(190, 226, 5):
        white_card = (
            (roi_luminance[:upper_limit] > threshold)
            & (roi_saturation[:upper_limit] < 50)
        ).astype(np.uint8) * 255
        mask_sources.append(white_card)

    for threshold in range(170, 221, 10):
        bright_only = (roi_luminance[:upper_limit] > threshold).astype(np.uint8) * 255
        mask_sources.append(bright_only)

    for card_mask in mask_sources:
        closed = cv2.morphologyEx(
            card_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
            iterations=2,
        )
        opened = cv2.morphologyEx(
            closed,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
            iterations=1,
        )
        card_contours, _ = cv2.findContours(
            opened,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        for contour in sorted(card_contours, key=cv2.contourArea, reverse=True)[:6]:
            candidate = _score_wallet_card_contour(
                contour,
                page_area,
                wallet_w,
                wallet_h,
                wallet_x,
                wallet_y,
                upper_limit,
                image,
                options,
            )
            if candidate is None:
                continue
            score, warped = candidate
            if score <= best_score:
                continue
            best_score = score
            best_warp = warped

    if best_warp is None:
        return None
    return best_score, best_warp


def _find_card_via_upper_region(
    image: np.ndarray,
    gray: np.ndarray,
    options: PreprocessOptions,
) -> tuple[float, np.ndarray] | None:
    crop_height = _dark_bottom_crop_height(gray)
    roi_img = image[:crop_height, :]
    roi_gray = gray[:crop_height, :]
    page_area = gray.shape[0] * gray.shape[1]

    best_score = 0.0
    best_warp: np.ndarray | None = None

    laplacian = cv2.Laplacian(cv2.GaussianBlur(roi_gray, (3, 3), 0), cv2.CV_64F)
    laplacian = np.abs(laplacian)
    for threshold in (20.0, 25.0, 30.0):
        candidate = _find_card_from_mask(
            roi_img,
            (laplacian > threshold).astype(np.uint8) * 255,
            page_area,
            options,
        )
        if candidate is None:
            continue
        score, warped = candidate
        if score <= best_score:
            continue
        best_score = score
        best_warp = warped

    for threshold in (190, 200, 210):
        candidate = _find_card_from_mask(
            roi_img,
            (roi_gray < threshold).astype(np.uint8) * 255,
            page_area,
            options,
            max_area_ratio=0.30,
        )
        if candidate is None:
            continue
        score, warped = candidate
        if score <= best_score:
            continue
        best_score = score
        best_warp = warped

    if best_warp is None:
        return None
    return best_score, best_warp


def _find_card_via_edge_projection(
    image: np.ndarray,
    gray: np.ndarray,
    options: PreprocessOptions,
) -> tuple[float, np.ndarray] | None:
    crop_height = _dark_bottom_crop_height(gray)
    roi_img = image[:crop_height, :]
    roi_gray = gray[:crop_height, :]
    roi_height, roi_width = roi_gray.shape[:2]
    page_area = gray.shape[0] * gray.shape[1]

    edges = cv2.Canny(cv2.GaussianBlur(roi_gray, (5, 5), 0), 40, 120)
    row_sum = edges.sum(axis=1).astype(np.float32)
    col_sum = edges.sum(axis=0).astype(np.float32)
    if row_sum.max() <= 0 or col_sum.max() <= 0:
        return None

    best_score = 0.0
    best_crop: np.ndarray | None = None
    for fraction in (0.12, 0.15, 0.18):
        rows = np.where(row_sum > row_sum.max() * fraction)[0]
        cols = np.where(col_sum > col_sum.max() * fraction)[0]
        if rows.size == 0 or cols.size == 0:
            continue

        y1 = max(0, int(rows[0]) - 12)
        y2 = min(roi_height, int(rows[-1]) + 12)
        x1 = max(0, int(cols[0]) - 12)
        x2 = min(roi_width, int(cols[-1]) + 12)
        crop = roi_img[y1:y2, x1:x2]
        crop_gray = roi_gray[y1:y2, x1:x2]
        if not _is_balanced_card_crop(crop_gray):
            continue

        area_ratio = ((y2 - y1) * (x2 - x1)) / page_area
        if area_ratio < 0.08 or area_ratio > 0.45:
            continue

        aspect_ratio = _card_aspect_ratio(crop)
        if aspect_ratio < 1.15 or aspect_ratio > options.max_card_aspect_ratio:
            continue

        aspect_score = 1.0 - min(abs(aspect_ratio - 1.58) / 0.6, 1.0)
        area_score = 1.0 - min(abs(area_ratio - 0.18) / 0.35, 1.0)
        score = max(aspect_score, 0.2) * max(area_score, 0.2)
        if score <= best_score:
            continue
        best_score = score
        best_crop = crop

    if best_crop is None:
        return None
    return best_score, best_crop


def _bbox_card_crop(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    margin_ratio: float,
) -> np.ndarray:
    image_height, image_width = image.shape[:2]
    full_width_bbox = width >= image_width * 0.72
    pad_x = int(width * margin_ratio)
    pad_y = int(height * (margin_ratio + (0.14 if full_width_bbox else 0.0)))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image_width, x + width + pad_x)
    y2 = min(image_height, y + height + pad_y)
    return image[y1:y2, x1:x2]


def _card_crop_from_contour(
    image: np.ndarray,
    contour: np.ndarray,
    offset_x: int,
    offset_y: int,
    options: PreprocessOptions,
) -> np.ndarray | None:
    x, y, width, height = cv2.boundingRect(contour)
    x += offset_x
    y += offset_y
    image_height, image_width = image.shape[:2]

    if width >= image_width * 0.72:
        crop = _bbox_card_crop(
            image,
            x,
            y,
            width,
            height,
            options.card_quad_padding_ratio,
        )
    else:
        rect = cv2.minAreaRect(contour)
        points = cv2.boxPoints(rect).astype(np.float32)
        points[:, 0] += offset_x
        points[:, 1] += offset_y
        ordered = order_quadrilateral_points(points)
        if _quad_parallel_side_difference(ordered) > options.max_card_perspective_distortion:
            crop = _bbox_card_crop(
                image,
                x,
                y,
                width,
                height,
                options.card_quad_padding_ratio,
            )
        else:
            ordered = _expand_quadrilateral_points(
                ordered,
                options.card_quad_padding_ratio,
                image_width,
                image_height,
            )
            crop = warp_quadrilateral(image, ordered)

    if crop.shape[0] < 80 or crop.shape[1] < 80:
        return None
    return crop


def _trim_scanner_bed_margin(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    if height < 200 or width < 200:
        return image

    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
    row_edge = edges.mean(axis=1)
    col_edge = edges.mean(axis=0)
    edge_peak = float(row_edge.max())
    if edge_peak < 20.0:
        return image

    # Only trim large scanner-bed margins; skip already-tight card crops.
    top_band = float(row_edge[: max(height // 5, 1)].mean())
    bottom_band = float(row_edge[4 * height // 5 :].mean())
    if top_band > edge_peak * 0.25 and bottom_band > edge_peak * 0.15:
        return image

    strong_rows = np.where(row_edge > edge_peak * 0.35)[0]
    strong_cols = np.where(col_edge > float(col_edge.max()) * 0.25)[0]
    if strong_rows.size == 0 or strong_cols.size == 0:
        return image

    top = max(0, int(strong_rows[0]) - 12)
    bottom = min(height, int(strong_rows[-1]) + 12)
    left = max(0, int(strong_cols[0]) - 12)
    right = min(width, int(strong_cols[-1]) + 12)
    if bottom - top < int(height * 0.35) or right - left < int(width * 0.45):
        return image
    trimmed_ratio = ((bottom - top) * (right - left)) / max(height * width, 1)
    if trimmed_ratio > 0.92:
        return image
    return image[top:bottom, left:right]


def _find_card_via_scanner_bed_content(
    image: np.ndarray,
    gray: np.ndarray,
    options: PreprocessOptions,
) -> tuple[float, np.ndarray] | None:
    page_mean = float(np.mean(gray))
    if page_mean < 125.0 or page_mean > 235.0:
        return None

    crop_height = _dark_bottom_crop_height(gray)
    roi_img = image[:crop_height, :]
    roi_gray = gray[:crop_height, :]
    roi_height, roi_width = roi_gray.shape[:2]
    page_area = gray.shape[0] * gray.shape[1]
    y_offset = int(roi_height * 0.35)
    sub_gray = roi_gray[y_offset:, :]
    sub_img = roi_img[y_offset:, :]

    blur = cv2.GaussianBlur(sub_gray.astype(np.float32), (15, 15), 0)
    squared = cv2.GaussianBlur(sub_gray.astype(np.float32) ** 2, (15, 15), 0)
    local_std = np.sqrt(np.maximum(squared - blur**2, 0))

    best_score = 0.0
    best_crop: np.ndarray | None = None

    for std_threshold in (6.0, 8.0, 10.0, 12.0):
        for lum_min in (145, 155, 165, 175):
            mask = (
                (local_std > std_threshold)
                & (sub_gray > lum_min)
                & (sub_gray < 245)
            ).astype(np.uint8) * 255
            closed = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)),
                iterations=2,
            )
            contours, _ = cv2.findContours(
                closed,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:6]:
                area = cv2.contourArea(contour)
                area_ratio = area / page_area
                if area_ratio < 0.05 or area_ratio > 0.24:
                    continue

                x, y, width, height = cv2.boundingRect(contour)
                center_y = y_offset + y + height / 2
                if center_y < crop_height * 0.52:
                    continue
                if height < 380 or width < 480:
                    continue

                crop = _bbox_card_crop(
                    roi_img,
                    x,
                    y_offset + y,
                    width,
                    height,
                    options.card_quad_padding_ratio,
                )
                crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                if float(np.mean(crop_gray)) < 100.0:
                    continue
                if not _is_balanced_card_crop(crop_gray):
                    continue

                crop = _trim_scanner_bed_margin(crop)
                crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                if crop_gray.shape[0] < 380 or crop_gray.shape[1] < 480:
                    continue
                if not _is_balanced_card_crop(crop_gray):
                    continue

                aspect_ratio = _card_aspect_ratio(crop)
                if aspect_ratio < 1.30 or aspect_ratio > 2.05:
                    continue

                aspect_score = 1.0 - min(abs(aspect_ratio - 1.58) / 0.6, 1.0)
                area_score = 1.0 - min(abs(area_ratio - 0.14) / 0.18, 1.0)
                position_score = min((center_y / crop_height) - 0.4, 1.0)
                score = area_ratio * max(aspect_score, 0.2) * max(area_score, 0.2) * max(position_score, 0.2)
                if score <= best_score:
                    continue
                best_score = score
                best_crop = crop

    if best_crop is None:
        return None
    return best_score, best_crop


def _find_card_from_mask(
    image: np.ndarray,
    mask: np.ndarray,
    page_area: float,
    options: PreprocessOptions,
    *,
    max_area_ratio: float = 0.40,
) -> tuple[float, np.ndarray] | None:
    closed = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
        iterations=2,
    )
    opened = cv2.morphologyEx(
        closed,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
        iterations=1,
    )
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_score = 0.0
    best_warp: np.ndarray | None = None
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        area = cv2.contourArea(contour)
        area_ratio = area / page_area
        if area_ratio < 0.04 or area_ratio > max_area_ratio:
            continue

        rect = cv2.minAreaRect(contour)
        box_width, box_height = rect[1]
        if min(box_width, box_height) < 80:
            continue

        aspect_ratio = max(box_width, box_height) / min(box_width, box_height)
        if not options.min_card_aspect_ratio <= aspect_ratio <= options.max_card_aspect_ratio:
            continue

        points = cv2.boxPoints(rect).astype(np.float32)
        ordered = order_quadrilateral_points(points)
        if _quad_parallel_side_difference(ordered) > options.max_card_perspective_distortion:
            continue

        image_height, image_width = image.shape[:2]
        ordered = _expand_quadrilateral_points(
            ordered,
            options.card_quad_padding_ratio,
            image_width,
            image_height,
        )

        aspect_score = 1.0 - min(abs(aspect_ratio - 1.58) / 0.6, 1.0)
        area_score = 1.0 - min(abs(area_ratio - 0.12) / 0.18, 1.0)
        score = area_ratio * max(aspect_score, 0.2) * max(area_score, 0.2)
        if score <= best_score:
            continue

        warped = warp_quadrilateral(image, ordered)
        if warped.shape[0] < 80 or warped.shape[1] < 80:
            continue
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        if float(np.mean(warped_gray)) < 100.0:
            continue
        if not _is_balanced_card_crop(warped_gray):
            continue

        best_score = score
        best_warp = warped

    if best_warp is None:
        return None
    return best_score, best_warp


def _score_wallet_card_contour(
    contour: np.ndarray,
    page_area: float,
    wallet_w: int,
    wallet_h: int,
    wallet_x: int,
    wallet_y: int,
    upper_limit: int,
    image: np.ndarray,
    options: PreprocessOptions,
) -> tuple[float, np.ndarray] | None:
    area = cv2.contourArea(contour)
    area_ratio = area / page_area
    if area_ratio < options.min_card_area_ratio:
        return None
    if area_ratio > min(options.max_card_area_ratio, 0.30):
        return None

    _, _, bbox_w, bbox_h = cv2.boundingRect(contour)
    full_page_wallet = wallet_x == 0 and wallet_w >= image.shape[1] * 0.95
    max_width_ratio = 1.0 if full_page_wallet else 0.90
    max_height_ratio = 0.78 if full_page_wallet else 0.55
    if bbox_h > wallet_h * max_height_ratio or bbox_w > wallet_w * max_width_ratio:
        return None

    center_y = cv2.boundingRect(contour)[1] + bbox_h / 2
    max_center_ratio = 0.72 if full_page_wallet else 0.52
    if center_y > wallet_h * max_center_ratio:
        return None

    rect = cv2.minAreaRect(contour)
    points = cv2.boxPoints(rect).astype(np.float32)
    points[:, 0] += wallet_x
    points[:, 1] += wallet_y
    aspect_ratio = _quadrilateral_aspect_ratio(points)
    if not options.min_card_aspect_ratio <= aspect_ratio <= options.max_card_aspect_ratio:
        return None

    ordered = order_quadrilateral_points(points)
    rectangularity = area / max(_quad_bounding_area(ordered), 1.0)
    aspect_score = 1.0 - min(abs(aspect_ratio - 1.58) / 0.6, 1.0)
    area_score = 1.0 - min(abs(area_ratio - 0.18) / 0.2, 1.0)
    position_score = 1.0 - min(center_y / max(upper_limit, 1), 1.0)
    score = (
        area_ratio
        * rectangularity
        * max(aspect_score, 0.2)
        * max(area_score, 0.2)
        * max(position_score, 0.15)
    )
    if full_page_wallet:
        score *= 1.35

    crop = _card_crop_from_contour(image, contour, wallet_x, wallet_y, options)
    if crop is None:
        return None
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if not _is_balanced_card_crop(crop_gray):
        return None
    return score, crop


def _find_quadrilaterals(gray: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edge_sources: list[np.ndarray] = []

    for low, high in ((30, 100), (50, 150)):
        edges = cv2.Canny(blurred, low, high)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
        closed = cv2.dilate(closed, kernel, iterations=2)
        edge_sources.append(closed)

    legacy_edges = cv2.Canny(blurred, 50, 150)
    edge_sources.append(legacy_edges)

    quads: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[tuple[int, int, int, int]] = set()

    for edges in edge_sources:
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:24]:
            hull = cv2.convexHull(contour)
            perimeter = cv2.arcLength(hull, True)
            for epsilon_ratio in (0.02, 0.03, 0.04, 0.05):
                approx = cv2.approxPolyDP(hull, epsilon_ratio * perimeter, True)
                if len(approx) != 4 or cv2.contourArea(contour) <= 0:
                    continue
                key = tuple(np.asarray(approx).reshape(-1).astype(int).tolist())
                if key in seen:
                    break
                seen.add(key)
                quads.append((contour, approx))
                break
    return quads


def order_quadrilateral_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)

    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]
    rect[1] = points[np.argmin(diffs)]
    rect[3] = points[np.argmax(diffs)]
    return rect


def warp_quadrilateral(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    width_a = np.linalg.norm(points[2] - points[3])
    width_b = np.linalg.norm(points[1] - points[0])
    height_a = np.linalg.norm(points[1] - points[2])
    height_b = np.linalg.norm(points[0] - points[3])

    max_width = int(max(width_a, width_b))
    max_height = int(max(height_a, height_b))
    if max_width < 50 or max_height < 50:
        return image

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(points, destination)
    return cv2.warpPerspective(
        image,
        matrix,
        (max_width, max_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def crop_to_page(image: np.ndarray, gray: np.ndarray, options: PreprocessOptions) -> np.ndarray:
    cropped = crop_to_content(image, gray, options)
    if cropped.shape[:2] != image.shape[:2]:
        return cropped
    return _crop_to_paper_bounds(image, gray, options)


def crop_to_content(image: np.ndarray, gray: np.ndarray, options: PreprocessOptions) -> np.ndarray:
    binary = _content_binary(gray)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) == 0:
        return image

    height, width = gray.shape[:2]
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    pad_y = max(options.min_padding_pixels, int(height * options.padding_ratio))
    pad_x = max(options.min_padding_pixels, int(width * options.padding_ratio))

    y_min = max(0, int(y_min) - pad_y)
    x_min = max(0, int(x_min) - pad_x)
    y_max = min(height - 1, int(y_max) + pad_y)
    x_max = min(width - 1, int(x_max) + pad_x)

    margin_y = min(y_min, height - 1 - y_max)
    margin_x = min(x_min, width - 1 - x_max)
    if margin_y < height * options.min_crop_margin_ratio and (
        margin_x < width * options.min_crop_margin_ratio
    ):
        return image

    if y_max <= y_min or x_max <= x_min:
        return image

    cropped = image[y_min : y_max + 1, x_min : x_max + 1]
    retained_ratio = (cropped.shape[0] * cropped.shape[1]) / (height * width)
    if retained_ratio < options.min_retained_area_ratio:
        return image

    return cropped


def _crop_to_paper_bounds(
    image: np.ndarray,
    gray: np.ndarray,
    options: PreprocessOptions,
) -> np.ndarray:
    bounds = _scanner_paper_bounds(gray)
    if bounds is None:
        return image
    if not _paper_bounds_covers_content(gray, bounds):
        return image

    y_min, y_max, x_min, x_max = bounds
    height, width = gray.shape[:2]
    margin_y = min(y_min, height - 1 - y_max)
    margin_x = min(x_min, width - 1 - x_max)
    if max(margin_y, margin_x) < min(height, width) * options.min_crop_margin_ratio:
        return image

    pad_y = max(options.min_padding_pixels, int(height * options.padding_ratio))
    pad_x = max(options.min_padding_pixels, int(width * options.padding_ratio))
    y_min = max(0, y_min - pad_y)
    x_min = max(0, x_min - pad_x)
    y_max = min(height - 1, y_max + pad_y)
    x_max = min(width - 1, x_max + pad_x)

    if y_max <= y_min or x_max <= x_min:
        return image

    cropped = image[y_min : y_max + 1, x_min : x_max + 1]
    retained_ratio = (cropped.shape[0] * cropped.shape[1]) / (height * width)
    if retained_ratio < options.min_document_boundary_area_ratio:
        return image
    return cropped


def _scanner_paper_bounds(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    height, width = gray.shape[:2]
    page_area = height * width
    if page_area <= 0:
        return None

    mask = (gray < 250).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(contour) / page_area
    if area_ratio < 0.20:
        return None
    if area_ratio > 0.98:
        return None

    x, y, box_width, box_height = cv2.boundingRect(contour)
    return y, y + box_height - 1, x, x + box_width - 1


def _paper_bounds_covers_content(
    gray: np.ndarray,
    bounds: tuple[int, int, int, int],
    *,
    min_coverage: float = 0.92,
) -> bool:
    binary = _content_binary(gray)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) == 0:
        return True

    y_min, y_max, x_min, x_max = bounds
    inside = np.sum(
        (coords[:, 0] >= y_min)
        & (coords[:, 0] <= y_max)
        & (coords[:, 1] >= x_min)
        & (coords[:, 1] <= x_max)
    )
    return inside / len(coords) >= min_coverage


def _quadrilateral_aspect_ratio(points: np.ndarray) -> float:
    ordered = order_quadrilateral_points(points)
    width = max(
        np.linalg.norm(ordered[1] - ordered[0]),
        np.linalg.norm(ordered[2] - ordered[3]),
    )
    height = max(
        np.linalg.norm(ordered[3] - ordered[0]),
        np.linalg.norm(ordered[2] - ordered[1]),
    )
    if width <= 0 or height <= 0:
        return 0.0
    long_side = max(width, height)
    short_side = min(width, height)
    return float(long_side / short_side)


def _quad_bounding_area(points: np.ndarray) -> float:
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    return float((x_coords.max() - x_coords.min()) * (y_coords.max() - y_coords.min()))


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


def pixmap_to_bgr(sample_bytes: bytes, width: int, height: int, channels: int) -> np.ndarray:
    array = np.frombuffer(sample_bytes, dtype=np.uint8).reshape(height, width, channels)
    if channels == 4:
        return cv2.cvtColor(array, cv2.COLOR_RGBA2BGR)
    if channels == 3:
        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)


def bgr_to_png_bytes(image: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Failed to encode processed page as PNG")
    return encoded.tobytes()
