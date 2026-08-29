"""Normative V2 primitives. No function in this module opens sealed Test."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .masks import _srgb_to_lab


@dataclass(frozen=True)
class BBox:
    x0: int
    y0: int
    x1: int
    y1: int


def _odd_scaled(dimension: int, fraction: float, minimum: int = 3) -> int:
    return max(minimum, int(round(dimension * fraction))) | 1


def _line_kernel(length: int, thickness: int, angle: float) -> np.ndarray:
    kernel = np.zeros((length, length), dtype=np.uint8)
    center = (length - 1) / 2
    radians = np.deg2rad(angle)
    dx, dy = np.cos(radians) * center, np.sin(radians) * center
    cv2.line(kernel, (round(center - dx), round(center - dy)),
             (round(center + dx), round(center + dy)), 1, max(1, thickness))
    return kernel


def _component_points(labels: np.ndarray, stats: np.ndarray, label: int) -> np.ndarray | None:
    x, y = int(stats[label, cv2.CC_STAT_LEFT]), int(stats[label, cv2.CC_STAT_TOP])
    width, height = int(stats[label, cv2.CC_STAT_WIDTH]), int(stats[label, cv2.CC_STAT_HEIGHT])
    if int(stats[label, cv2.CC_STAT_AREA]) < 3:
        return None
    local_y, local_x = np.where(labels[y:y + height, x:x + width] == label)
    return np.column_stack((local_x + x, local_y + y)).astype(np.float32)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def bbox_from_binary(mask: np.ndarray) -> BBox:
    ys, xs = np.where(np.asarray(mask) > 0)
    if xs.size == 0:
        raise ValueError("ground-truth lesion mask is empty")
    return BBox(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def required_symmetric_margin(predicted: BBox, ground_truth: BBox) -> float:
    width, height = predicted.x1 - predicted.x0, predicted.y1 - predicted.y0
    if width <= 0 or height <= 0:
        raise ValueError("predicted bbox must have positive area")
    return max(
        0.0,
        (predicted.x0 - ground_truth.x0) / width,
        (ground_truth.x1 - predicted.x1) / width,
        (predicted.y0 - ground_truth.y0) / height,
        (ground_truth.y1 - predicted.y1) / height,
    )


def q95(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("Q95 requires at least one finite observation")
    return float(np.quantile(array, 0.95, method="linear"))


def expand_bbox(bbox: BBox, margin_fraction: float, width: int, height: int) -> BBox:
    dx = (bbox.x1 - bbox.x0) * margin_fraction
    dy = (bbox.y1 - bbox.y0) * margin_fraction
    return BBox(
        max(0, int(np.floor(bbox.x0 - dx))),
        max(0, int(np.floor(bbox.y0 - dy))),
        min(width, int(np.ceil(bbox.x1 + dx))),
        min(height, int(np.ceil(bbox.y1 + dy))),
    )


def detect_hair_mask(rgb: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """D12–D13 detector on the full ROI; returns a mask and never inpaints."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("hair detector expects HxWx3 uint8 RGB")
    height, width = rgb.shape[:2]
    minimum_dimension = min(height, width)
    length = _odd_scaled(minimum_dimension, float(config["line_length_fraction"]), minimum=5)
    thickness = max(1, round(minimum_dimension * float(config["line_thickness_fraction"])))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    responses = [
        cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, _line_kernel(length, thickness, float(angle)))
        for angle in config["orientations_degrees"]
    ]
    response = np.maximum.reduce(responses)
    threshold = max(5.0, float(np.percentile(response, float(config["response_percentile"]))))
    hair = np.zeros((height, width), dtype=np.uint8)
    maximum_thickness = max(2.0, minimum_dimension * float(config["maximum_component_thickness_fraction"]))
    minimum_length = max(3.0, length * 0.35)
    for oriented in responses:
        count, labels, stats, _ = cv2.connectedComponentsWithStats((oriented >= threshold).astype(np.uint8), 8)
        for label in range(1, count):
            points = _component_points(labels, stats, label)
            if points is None:
                continue
            (_, _), (side_a, side_b), _ = cv2.minAreaRect(points)
            long_side, short_side = max(side_a, side_b), max(1.0, min(side_a, side_b))
            if (long_side >= minimum_length and
                    long_side / short_side >= float(config["minimum_elongation"]) and
                    short_side <= maximum_thickness):
                hair[labels == label] = 1
    dilation = _odd_scaled(minimum_dimension, float(config["dilation_fraction"]))
    if hair.any():
        hair = cv2.dilate(hair, np.ones((dilation, dilation), np.uint8))
    coverage = float(hair.mean())
    return hair.astype(np.uint8), {
        "hair_coverage_fraction": coverage,
        "hair_mask_high_coverage": coverage > float(config["maximum_coverage_fraction"]),
        "effective_line_length": length,
        "effective_line_thickness": thickness,
        "effective_threshold": threshold,
        "effective_dilation_kernel": dilation,
    }


def highlight_mask(rgb: np.ndarray, channel_minimum: int = 248, channel_spread_maximum: int = 22) -> np.ndarray:
    maximum, minimum = rgb.max(axis=2), rgb.min(axis=2)
    return ((maximum >= channel_minimum) & ((maximum.astype(np.int16) - minimum) <= channel_spread_maximum)).astype(np.uint8)


def safety_margin_fraction(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    pred, gt = prediction.astype(bool), ground_truth.astype(bool)
    if not pred.any() or pred.all():
        raise ValueError("degenerate lesion masks do not calibrate D35")
    false_negative = gt & ~pred
    if not false_negative.any():
        return 0.0
    distances = cv2.distanceTransform((~pred).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[false_negative]
    return float(np.quantile(distances, 0.95, method="linear") / min(pred.shape))


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    distance = cv2.distanceTransform((~mask.astype(bool)).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return distance <= radius


def clean_skin_mask(
    rgb: np.ndarray,
    lesion_mask: np.ndarray,
    hair: np.ndarray,
    margin_fraction: float,
    *,
    minimum_pixels: int = 256,
    minimum_fraction: float = 0.005,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    lesion = lesion_mask.astype(bool)
    if not lesion.any() or lesion.all():
        return None, {"color_status": "unavailable_degenerate_lesion_mask"}
    radius = int(np.ceil(margin_fraction * min(lesion.shape)))
    highlights = highlight_mask(rgb).astype(bool)
    clean = ~_dilate(lesion, radius) & ~hair.astype(bool) & ~highlights
    required = max(minimum_pixels, int(np.ceil(minimum_fraction * lesion.size)))
    metadata = {
        "lesion_safety_margin_pixels": radius,
        "candidate_skin_pixel_count": int(clean.sum()),
        "minimum_clean_skin_pixels": required,
        "hair_pixel_count": int(hair.sum()),
        "highlight_pixel_count": int(highlights.sum()),
    }
    if clean.sum() < required:
        return None, {**metadata, "color_status": "unavailable_insufficient_clean_skin"}
    return clean.astype(np.uint8), {**metadata, "color_status": "available"}


def colour_metrics(rgb: np.ndarray, clean: np.ndarray) -> dict[str, Any]:
    pixels = rgb[clean.astype(bool)]
    if pixels.size == 0:
        raise ValueError("clean skin mask is empty")
    lower, upper = np.quantile(pixels, [0.1, 0.9], axis=0, method="linear")
    trimmed = pixels[np.all((pixels >= lower) & (pixels <= upper), axis=1)]
    if trimmed.size == 0:
        return {"color_status": "unavailable_empty_trimmed_set", "color_metrics": None}
    lab = np.median(_srgb_to_lab(pixels), axis=0)
    return {
        "color_status": "available",
        "pixel_count": int(len(pixels)),
        "coverage_fraction": float(len(pixels) / clean.size),
        "rgb_median": np.median(pixels, axis=0).tolist(),
        "rgb_trimmed_mean_10_90": trimmed.mean(axis=0).tolist(),
        "lab_median": lab.tolist(),
        "ita_degrees": float(np.degrees(np.arctan2(lab[0] - 50.0, lab[2]))),
    }


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb)
    if values.shape[-1] != 3: raise ValueError("sRGB input must end in three channels")
    return _srgb_to_lab(values.reshape(-1, 3)).reshape(values.shape)


def lab_to_srgb_unclipped(lab: np.ndarray) -> np.ndarray:
    """Inverse D41 conversion, returning float64 sRGB before gamut clipping."""
    values = np.asarray(lab, dtype=np.float64)
    fy = (values[..., 0] + 16.0) / 116.0
    fx, fz = fy + values[..., 1] / 500.0, fy - values[..., 2] / 200.0
    delta = 6.0 / 29.0
    inverse = lambda component: np.where(component > delta, component**3, 3*delta**2*(component-4.0/29.0))
    xyz = np.stack((.95047*inverse(fx), inverse(fy), 1.08883*inverse(fz)), axis=-1)
    linear = xyz @ np.array([[3.2404542, -1.5371385, -.4985314],
                             [-.9692660, 1.8760108, .0415560],
                             [.0556434, -.2040259, 1.0572252]], dtype=np.float64).T
    return np.where(linear <= .0031308, 12.92*linear, 1.055*np.power(np.maximum(linear, 0), 1/2.4)-.055)
