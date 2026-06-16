from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .contracts import DetectionResult


def normalize_angle_deg(angle_deg: float) -> float:
    return ((float(angle_deg) + 90.0) % 180.0) - 90.0


def rotated_box_to_polygon(
    cx: float, cy: float, w: float, h: float, angle_deg: float,
) -> tuple[tuple[float, float], ...]:
    r = math.radians(angle_deg)
    cos_a, sin_a = math.cos(r), math.sin(r)
    hw, hh = w / 2.0, h / 2.0
    corners = ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))
    return tuple(
        (float(x * cos_a - y * sin_a + cx), float(x * sin_a + y * cos_a + cy))
        for x, y in corners
    )


def polygons_batch(
    cxs: np.ndarray, cys: np.ndarray,
    ws: np.ndarray, hs: np.ndarray,
    angles_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r = np.radians(angles_deg)
    cos_a = np.cos(r)
    sin_a = np.sin(r)
    hw = ws * 0.5
    hh = hs * 0.5
    dx = np.array([-1.0, 1.0, 1.0, -1.0], dtype=np.float32)
    dy = np.array([-1.0, -1.0, 1.0, 1.0], dtype=np.float32)
    px = np.outer(hw, dx)
    py = np.outer(hh, dy)
    xs = px * cos_a[:, None] - py * sin_a[:, None] + cxs[:, None]
    ys = px * sin_a[:, None] + py * cos_a[:, None] + cys[:, None]
    return xs, ys


def enclosing_bbox(
    polygon: Sequence[tuple[float, float]],
) -> tuple[float, float, float, float]:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))


def polygon_area(polygon: Sequence[tuple[float, float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    area = 0.0
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def _signed_area(polygon: Sequence[tuple[float, float]]) -> float:
    area = 0.0
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]
        area += x1 * y2 - x2 * y1
    return area * 0.5


def _inside(point, a, b, ccw):
    x, y = point
    cross = (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])
    return cross >= -1e-9 if ccw else cross <= 1e-9


def _intersect(p1, p2, q1, q2):
    x1, y1 = p1; x2, y2 = p2; x3, y3 = q1; x4, y4 = q2
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-12:
        return float(x2), float(y2)
    d1 = x1 * y2 - y1 * x2
    d2 = x3 * y4 - y3 * x4
    return float((d1 * (x3 - x4) - (x1 - x2) * d2) / d), float((d1 * (y3 - y4) - (y1 - y2) * d2) / d)


def polygon_clip(subject, clip):
    if len(subject) < 3 or len(clip) < 3:
        return tuple()
    out = list(subject)
    ccw = _signed_area(clip) >= 0.0
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        inp = out; out = []
        if not inp:
            break
        s = inp[-1]
        for e in inp:
            ei = _inside(e, a, b, ccw)
            si = _inside(s, a, b, ccw)
            if ei:
                if not si:
                    out.append(_intersect(s, e, a, b))
                out.append((float(e[0]), float(e[1])))
            elif si:
                out.append(_intersect(s, e, a, b))
            s = e
    return tuple(out)


def rotated_iou(a, b) -> float:
    aa, ab = polygon_area(a), polygon_area(b)
    if aa <= 0.0 or ab <= 0.0:
        return 0.0
    inter = polygon_area(polygon_clip(a, b))
    if inter <= 0.0:
        return 0.0
    union = aa + ab - inter
    return float(inter / union) if union > 0 else 0.0


def _bbox_iou_matrix(
    x1s: np.ndarray, y1s: np.ndarray,
    x2s: np.ndarray, y2s: np.ndarray,
) -> np.ndarray:
    areas = (x2s - x1s) * (y2s - y1s)
    ix1 = np.maximum(x1s[:, None], x1s[None, :])
    iy1 = np.maximum(y1s[:, None], y1s[None, :])
    ix2 = np.minimum(x2s[:, None], x2s[None, :])
    iy2 = np.minimum(y2s[:, None], y2s[None, :])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    union = areas[:, None] + areas[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def rotated_nms(
    detections: Sequence[DetectionResult],
    iou_threshold: float,
) -> list[DetectionResult]:
    if not detections:
        return []

    kept: list[DetectionResult] = []
    grouped: dict[int, list[DetectionResult]] = {}
    for det in detections:
        grouped.setdefault(det.class_id, []).append(det)

    for group in grouped.values():
        ordered = sorted(group, key=lambda d: d.score, reverse=True)
        if len(ordered) == 1:
            kept.append(ordered[0])
            continue

        bboxes = np.array([d.bbox for d in ordered], dtype=np.float32)
        bbox_iou = _bbox_iou_matrix(bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3])
        pre_threshold = iou_threshold * 0.25

        while ordered:
            best = ordered.pop(0)
            kept.append(best)
            bi = len(ordered)
            survivors = []
            for j, cand in enumerate(ordered):
                idx = bi - len(ordered) + j
                if bbox_iou[0, idx + 1] < pre_threshold:
                    survivors.append(cand)
                elif rotated_iou(best.polygon, cand.polygon) < iou_threshold:
                    survivors.append(cand)
            ordered = survivors

    kept.sort(key=lambda d: d.score, reverse=True)
    return kept


def filter_detections(
    detections: Sequence[DetectionResult],
    min_area: float = 0.0,
    max_area: float | None = None,
    max_aspect_ratio: float | None = None,
    min_score: float = 0.0,
) -> list[DetectionResult]:
    if not detections:
        return []
    out = []
    for det in detections:
        if det.score < min_score:
            continue
        w, h = det.size
        if min_area > 0.0 or max_area is not None:
            area = w * h
            if area < min_area:
                continue
            if max_area is not None and area > max_area:
                continue
        if max_aspect_ratio is not None and min(w, h) > 0:
            if max(w, h) / min(w, h) > max_aspect_ratio:
                continue
        out.append(det)
    return out
