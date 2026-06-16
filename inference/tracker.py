from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .contracts import DetectionResult
from .postprocess import rotated_iou


@dataclass
class _Track:
    track_id: int
    class_id: int
    label: str
    bbox: tuple[float, float, float, float]
    polygon: tuple[tuple[float, float], ...]
    centroid: tuple[float, float]
    size: tuple[float, float]
    angle_deg: float
    score: float
    hits: int = 1
    age: int = 1
    disappeared: int = 0


class PolygonTracker:
    def __init__(
        self,
        max_disappeared: int = 30,
        max_distance: float = 75.0,
        polygon: tuple[tuple[int, int], ...] = tuple(),
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        new_track_thresh: float = 0.5,
        match_iou_threshold: float = 0.15,
        min_hits: int = 1,
    ) -> None:
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.polygon = polygon
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_iou_threshold = match_iou_threshold
        self.min_hits = min_hits
        self._next_id = 1
        self._tracks: OrderedDict[int, _Track] = OrderedDict()
        self.frame_size: tuple[int, int] = (1, 1)

    def reset(self) -> None:
        self._next_id = 1
        self._tracks.clear()
        self.frame_size = (1, 1)

    def update(self, detections: Sequence[DetectionResult]) -> list[DetectionResult]:
        dets = list(detections)
        if not dets:
            self._age_tracks(list(self._tracks.keys()))
            return []

        high = [i for i, d in enumerate(dets) if d.score >= self.high_thresh]
        low  = [i for i, d in enumerate(dets) if self.low_thresh <= d.score < self.high_thresh]
        results: list[DetectionResult | None] = [None] * len(dets)

        if not self._tracks:
            for i in high:
                if dets[i].score >= self.new_track_thresh:
                    results[i] = self._register(dets[i])
            return [r for r in results if r is not None]

        tids = list(self._tracks.keys())
        matched_h, unmatched_t, unmatched_h = self._match(tids, dets, high)
        for tid, di in matched_h:
            results[di] = self._update(tid, dets[di])

        matched_l, still_unmatched, _ = self._match(unmatched_t, dets, low)
        for tid, di in matched_l:
            results[di] = self._update(tid, dets[di])

        self._age_tracks(still_unmatched)

        for i in unmatched_h:
            if dets[i].score >= self.new_track_thresh:
                results[i] = self._register(dets[i])

        if self.min_hits > 1:
            out = []
            for r in results:
                if r is None:
                    continue
                t = self._tracks.get(r.track_id)
                if t is not None and t.hits >= self.min_hits:
                    out.append(r)
            return out

        return [r for r in results if r is not None]

    def _match(
        self,
        tids: Sequence[int],
        dets: Sequence[DetectionResult],
        det_indices: Sequence[int],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not tids or not det_indices:
            return [], list(tids), list(det_indices)

        candidates: list[tuple[float, float, int, int]] = []
        for tid in tids:
            t = self._tracks[tid]
            for di in det_indices:
                d = dets[di]
                if t.class_id != d.class_id:
                    continue
                dist = float(np.linalg.norm(np.array(t.centroid) - np.array(d.center)))
                if dist > self.max_distance:
                    continue
                iou = rotated_iou(t.polygon, d.polygon)
                if iou < self.match_iou_threshold:
                    continue
                candidates.append((-iou, dist, tid, di))

        candidates.sort()
        matched: list[tuple[int, int]] = []
        used_t: set[int] = set()
        used_d: set[int] = set()
        for _, _, tid, di in candidates:
            if tid in used_t or di in used_d:
                continue
            matched.append((tid, di))
            used_t.add(tid)
            used_d.add(di)

        return (
            matched,
            [tid for tid in tids if tid not in used_t],
            [di for di in det_indices if di not in used_d],
        )

    def _register(self, det: DetectionResult) -> DetectionResult:
        tid = self._next_id
        self._next_id += 1
        self._tracks[tid] = _Track(
            track_id=tid, class_id=det.class_id, label=det.label,
            bbox=det.bbox, polygon=det.polygon, centroid=det.center,
            size=det.size, angle_deg=det.angle_deg, score=det.score,
        )
        return self._snap(tid, det)

    def _update(self, tid: int, det: DetectionResult) -> DetectionResult:
        prev = self._tracks[tid]
        self._tracks[tid] = _Track(
            track_id=tid, class_id=det.class_id, label=det.label,
            bbox=det.bbox, polygon=det.polygon, centroid=det.center,
            size=det.size, angle_deg=det.angle_deg, score=det.score,
            hits=prev.hits + 1, age=prev.age + 1, disappeared=0,
        )
        return self._snap(tid, det)

    def _snap(self, tid: int, det: DetectionResult) -> DetectionResult:
        return replace(det, track_id=tid, inside_polygon=self._inside(det.center))

    def _age_tracks(self, tids: Sequence[int]) -> None:
        for tid in list(tids):
            t = self._tracks.get(tid)
            if t is None:
                continue
            t.age += 1
            t.disappeared += 1
            if t.disappeared > self.max_disappeared:
                del self._tracks[tid]

    def _inside(self, point: tuple[float, float]) -> bool:
        if len(self.polygon) < 3:
            return True
        fw, fh = self.frame_size
        x, y = point
        inside = False
        for i in range(len(self.polygon)):
            nx1, ny1 = self.polygon[i]
            nx2, ny2 = self.polygon[(i + 1) % len(self.polygon)]
            x1, y1 = nx1 * fw, ny1 * fh
            x2, y2 = nx2 * fw, ny2 * fh
            if ((y1 > y) != (y2 > y)) and x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-9) + x1:
                inside = not inside
        return inside
