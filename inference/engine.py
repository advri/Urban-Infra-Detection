from __future__ import annotations

import asyncio
import math
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import onnxruntime as ort

from .contracts import DetectionResult, FrameInferenceResult, VideoInferenceResult
from .model_manager import AppSettings, get_settings, get_shared_session
from .postprocess import (
    enclosing_bbox, filter_detections, normalize_angle_deg,
    polygons_batch, rotated_box_to_polygon, rotated_nms,
)
from .tracker import PolygonTracker

_EXECUTOR = ThreadPoolExecutor(max_workers=None, thread_name_prefix="onnx")

_PALETTE: tuple[tuple[int, int, int], ...] = (
    (220, 20, 60), (30, 144, 255), (50, 205, 50), (0, 215, 255),
    (238, 130, 238), (32, 165, 218), (64, 224, 208), (144, 0, 238),
    (0, 255, 127), (180, 105, 255), (0, 140, 255), (211, 0, 148),
    (71, 139, 139), (0, 100, 255), (205, 133, 63), (112, 255, 112),
    (250, 230, 230), (0, 255, 255), (255, 0, 255), (127, 255, 212),
)
_ROI_COLOR = (0, 255, 255)


def _color(class_id: int) -> tuple[int, int, int]:
    return _PALETTE[class_id % len(_PALETTE)]


@lru_cache(maxsize=8)
def _pil_font(size: int):
    from PIL import ImageFont
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _ffmpeg_to_h264(src: str, dst: str) -> bool:
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", src,
             "-c:v", "libx264", "-preset", "fast", "-crf", "23",
             "-movflags", "+faststart", "-f", "mp4", dst],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300,
        )
        return r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0
    except Exception:
        return False


def _ffmpeg_faststart_copy(src: str, dst: str) -> bool:
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-c", "copy", "-movflags", "+faststart", dst],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120,
        )
        return r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0
    except Exception:
        return False


def _make_writer(path: str, fps: float, w: int, h: int) -> tuple[cv2.VideoWriter, str]:
    for cc in ("mp4v", "avc1"):
        wri = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*cc), fps, (w, h))
        if wri.isOpened():
            return wri, path
        wri.release()
    avi = path.replace(".mp4", ".avi")
    return cv2.VideoWriter(avi, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h)), avi


class ONNXInferenceEngine:
    def __init__(self, settings: AppSettings | None = None, session=None, labels=None):
        self.settings = settings or get_settings()
        self.session = session or get_shared_session(self.settings)
        self.labels = tuple(labels) if labels is not None else self.settings.labels
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        self._use_io_binding = hasattr(self.session, "io_binding")

        self.tracker = PolygonTracker(
            max_disappeared=self.settings.tracker_max_disappeared,
            max_distance=self.settings.tracker_max_distance,
            polygon=self.settings.tracker_polygon,
            high_thresh=self.settings.tracker_high_thresh,
            low_thresh=self.settings.tracker_low_thresh,
            new_track_thresh=self.settings.tracker_new_track_thresh,
            match_iou_threshold=self.settings.tracker_match_iou_threshold,
            min_hits=self.settings.tracker_min_hits,
        )

    def infer_frame(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        use_tracking: bool = False,
    ) -> FrameInferenceResult:
        if frame is None or frame.size == 0:
            raise ValueError("Empty frame")
        h, w = frame.shape[:2]
        t0 = time.perf_counter()
        tensor, scale, px, py = self._preprocess(frame)
        t1 = time.perf_counter()
        outputs = self._run_inference(tensor)
        t2 = time.perf_counter()
        dets = self._decode(outputs, w, h, scale, px, py)
        dets = rotated_nms(dets, self.settings.iou_threshold)
        dets = filter_detections(
            dets,
            min_area=self.settings.min_det_area,
            max_aspect_ratio=self.settings.max_det_aspect_ratio,
        )
        if use_tracking:
            self.tracker.frame_size = (w, h)
            dets = self.tracker.update(dets)
        t3 = time.perf_counter()
        return FrameInferenceResult(
            frame_index=frame_index, width=w, height=h, detections=dets,
            timings_ms={
                "pre": (t1 - t0) * 1e3,
                "inf": (t2 - t1) * 1e3,
                "post": (t3 - t2) * 1e3,
                "total": (t3 - t0) * 1e3,
            },
        )

    def _run_inference(self, tensor: np.ndarray) -> list[np.ndarray]:
        if self._use_io_binding:
            try:
                binding = self.session.io_binding()
                binding.bind_cpu_input(self.input_name, tensor)
                for name in self.output_names:
                    binding.bind_output(name)
                self.session.run_with_iobinding(binding)
                return [o.numpy() for o in binding.get_outputs()]
            except Exception:
                self._use_io_binding = False
        return self.session.run(self.output_names, {self.input_name: tensor})

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: Sequence[DetectionResult],
    ) -> np.ndarray:
        if not detections and len(self.settings.tracker_polygon) < 3:
            return frame.copy()

        out = frame.copy()

        if len(self.settings.tracker_polygon) >= 3:
            fh, fw = out.shape[:2]
            roi_pts = [
                (int(x * fw), int(y * fh))
                for x, y in self.settings.tracker_polygon
            ]
            cv2.polylines(out, [np.array(roi_pts, dtype=np.int32).reshape((-1, 1, 2))], True, _ROI_COLOR, 2)

        if not detections:
            return out

        for det in detections:
            c = _color(det.class_id)
            poly = np.array(det.polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(out, [poly], True, c, 2)

        try:
            from PIL import Image, ImageDraw
            font = _pil_font(15)
            rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            draw = ImageDraw.Draw(pil)
            for det in detections:
                c = _color(det.class_id)
                x1, y1 = int(det.bbox[0]), int(det.bbox[1])
                track = f" [{det.track_id}]" if det.track_id is not None else ""
                text = f"{det.label}{track}"
                tx, ty = max(0, x1), max(4, y1 - 20)
                for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
                    draw.text((tx + dx, ty + dy), text, font=font, fill=(0, 0, 0))
                draw.text((tx, ty), text, font=font, fill=(c[0], c[1], c[2]))
            out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception:
            for det in detections:
                c = _color(det.class_id)
                x1, y1 = int(det.bbox[0]), int(det.bbox[1])
                track = f" [{det.track_id}]" if det.track_id is not None else ""
                cv2.putText(
                    out,
                    f"{det.label}{track}".encode("ascii", "replace").decode(),
                    (max(0, x1), max(4, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA,
                )
        return out

    def infer_video(
        self,
        video_path,
        output_path=None,
        sample_every: int = 1,
        max_frames: int | None = None,
        use_tracking: bool = False,
    ) -> VideoInferenceResult:
        vp = Path(video_path)
        if not vp.exists():
            raise FileNotFoundError(f"Video not found: {vp}")

        cap = cv2.VideoCapture(vp.as_posix())
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {vp}")

        fps = cap.get(cv2.CAP_PROP_FPS) or self.settings.default_fps
        total_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total = total_raw if total_raw > 0 else None
        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        final_out: Path | None = None
        raw_out: str | None = None
        actual_raw: str | None = None
        writer: cv2.VideoWriter | None = None

        if output_path is not None:
            final_out = Path(output_path)
            final_out.parent.mkdir(parents=True, exist_ok=True)
            fd, raw_out = tempfile.mkstemp(suffix=".mp4", dir=str(final_out.parent))
            os.close(fd)
            writer, actual_raw = _make_writer(raw_out, fps, vw, vh)
            if actual_raw != raw_out:
                try:
                    os.unlink(raw_out)
                except OSError:
                    pass

        if use_tracking:
            self.tracker.reset()

        results: list[FrameInferenceResult] = []
        fi = 0
        processed = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if fi % sample_every != 0:
                    # Пишем исходный кадр без инференса
                    if writer is not None:
                        writer.write(frame)
                    fi += 1
                    continue

                if max_frames is not None and processed >= max_frames:
                    break

                r = self.infer_frame(frame, frame_index=fi, use_tracking=use_tracking)
                results.append(r)
                processed += 1

                if writer is not None:
                    writer.write(self.annotate_frame(frame, r.detections))

                fi += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        if actual_raw and final_out:
            if _ffmpeg_to_h264(actual_raw, str(final_out)):
                try:
                    os.unlink(actual_raw)
                except OSError:
                    pass
            elif _ffmpeg_faststart_copy(actual_raw, str(final_out)):
                try:
                    os.unlink(actual_raw)
                except OSError:
                    pass
            else:
                os.replace(actual_raw, str(final_out))

        return VideoInferenceResult(
            video_path=vp,
            output_path=final_out,
            fps=float(fps),
            total_frames=total,
            processed_frames=processed,
            sampled_every=sample_every,
            frames=results,
        )

    def close(self):
        pass

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        sz = self.settings.input_size
        h, w = frame.shape[:2]
        scale = min(sz / w, sz / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))

        # INTER_AREA точнее и быстрее при downscale
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(frame, (nw, nh), interpolation=interp)

        canvas = np.full((sz, sz, 3), 114, dtype=np.uint8)
        px, py = (sz - nw) // 2, (sz - nh) // 2
        canvas[py:py + nh, px:px + nw] = resized

        img = canvas[:, :, ::-1].astype(np.float32)
        if self.settings.normalize_01:
            img *= (1.0 / 255.0)

        tensor = np.ascontiguousarray(img.transpose(2, 0, 1)[np.newaxis])
        return tensor, scale, px, py

    def _decode(
        self,
        outputs: list[np.ndarray],
        ow: int, oh: int,
        scale: float, px: int, py: int,
    ) -> list[DetectionResult]:
        idx = self.settings.model_output_index
        if idx >= len(outputs):
            raise IndexError(f"output index {idx} out of range ({len(outputs)})")
        preds = np.squeeze(np.asarray(outputs[idx]))
        if preds.ndim != 2:
            raise ValueError(f"Expected 2D predictions, got {preds.shape}")
        return (
            self._decode_predecoded(preds, ow, oh, scale, px, py)
            if self.settings.model_predecoded
            else self._decode_standard(preds, ow, oh, scale, px, py)
        )

    def _decode_predecoded(
        self,
        preds: np.ndarray,
        ow: int, oh: int,
        scale: float, px: int, py: int,
    ) -> list[DetectionResult]:
        if preds.shape[1] < 7:
            if preds.shape[0] >= 7:
                preds = preds.T
            else:
                raise ValueError(f"Pre-decoded format needs >=7 cols, got {preds.shape}")

        preds = preds.astype(np.float32)
        cxs = preds[:, 0]; cys = preds[:, 1]
        ws  = preds[:, 2]; hs  = preds[:, 3]
        angles = preds[:, 4]
        class_ids_f = preds[:, 5]
        scores = preds[:, 6]

        mask = (ws > 0) & (hs > 0) & (scores >= self.settings.confidence_threshold) & (class_ids_f >= 0)
        if not mask.any():
            return []

        cxs = cxs[mask]; cys = cys[mask]
        ws  = ws[mask];  hs  = hs[mask]
        angles = angles[mask]
        class_ids = np.round(class_ids_f[mask]).astype(np.int32)
        scores = scores[mask]

        if not self.settings.model_angle_in_degrees:
            angles_deg = np.degrees(angles)
        else:
            angles_deg = angles.copy()

        angles_deg = (angles_deg + 90.0) % 180.0 - 90.0
        swap = hs > ws
        if swap.any():
            ws[swap], hs[swap] = hs[swap].copy(), ws[swap].copy()
            angles_deg[swap] = (angles_deg[swap] + 90.0) % 180.0 - 90.0

        cxs = (cxs - px) / scale
        cys = (cys - py) / scale
        ws  = ws / scale
        hs  = hs / scale

        xs_poly, ys_poly = polygons_batch(cxs, cys, ws, hs, angles_deg)

        mx = float(max(ow - 1, 0))
        my = float(max(oh - 1, 0))
        xs_poly = np.clip(xs_poly, 0.0, mx)
        ys_poly = np.clip(ys_poly, 0.0, my)

        x1s = xs_poly.min(1).clip(0, ow - 1)
        y1s = ys_poly.min(1).clip(0, oh - 1)
        x2s = xs_poly.max(1).clip(0, ow - 1)
        y2s = ys_poly.max(1).clip(0, oh - 1)

        cxs_c = np.clip(cxs, 0, ow - 1)
        cys_c = np.clip(cys, 0, oh - 1)

        valid = (x2s > x1s) & (y2s > y1s)

        out: list[DetectionResult] = []
        n_labels = len(self.labels)
        for i in np.where(valid)[0]:
            cid = int(class_ids[i])
            label = self.labels[cid] if self.labels and 0 <= cid < n_labels else f"class_{cid}"
            polygon = tuple(
                (float(xs_poly[i, j]), float(ys_poly[i, j]))
                for j in range(4)
            )
            out.append(DetectionResult(
                class_id=cid,
                label=label,
                score=float(scores[i]),
                bbox=(float(x1s[i]), float(y1s[i]), float(x2s[i]), float(y2s[i])),
                center=(float(cxs_c[i]), float(cys_c[i])),
                size=(float(ws[i]), float(hs[i])),
                angle_deg=float(angles_deg[i]),
                polygon=polygon,
                metadata={},
            ))
        return out

    def _decode_standard(
        self,
        preds: np.ndarray,
        ow: int, oh: int,
        scale: float, px: int, py: int,
    ) -> list[DetectionResult]:
        preds = self._fix_layout(preds)
        out: list[DetectionResult] = []
        for row in preds:
            row = row.astype(np.float32)
            cx, cy, w, h, angle = map(float, row[:5])
            if w <= 0 or h <= 0:
                continue
            tail = row[5:]
            if self.settings.model_has_objectness:
                if tail.size < 2:
                    continue
                obj = float(tail[0])
                cs  = tail[1:].astype(np.float32)
                cid = int(cs.argmax())
                score = obj * float(cs[cid])
            else:
                cs = tail.astype(np.float32)
                if cs.size == 0:
                    continue
                cid   = int(cs.argmax())
                score = float(cs[cid])
            if score < self.settings.confidence_threshold:
                continue
            ad = angle if self.settings.model_angle_in_degrees else math.degrees(angle)
            cx, cy, w, h = (cx - px) / scale, (cy - py) / scale, w / scale, h / scale
            cx, cy, w, h, ad = self._canon(cx, cy, w, h, ad)
            if w <= 0 or h <= 0:
                continue
            det = self._make_det(cx, cy, w, h, ad, score, cid, ow, oh)
            if det:
                out.append(det)
        return out

    def _make_det(self, cx, cy, w, h, angle_deg, score, class_id, ow, oh):
        polygon = rotated_box_to_polygon(cx, cy, w, h, angle_deg)
        polygon = self._clip(polygon, ow, oh)
        bbox    = enclosing_bbox(polygon)
        x1 = max(0.0, min(float(ow - 1), bbox[0]))
        y1 = max(0.0, min(float(oh - 1), bbox[1]))
        x2 = max(0.0, min(float(ow - 1), bbox[2]))
        y2 = max(0.0, min(float(oh - 1), bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return None
        n_labels = len(self.labels)
        label = self.labels[class_id] if self.labels and 0 <= class_id < n_labels else f"class_{class_id}"
        return DetectionResult(
            class_id=class_id, label=label, score=score,
            bbox=(x1, y1, x2, y2),
            center=(max(0.0, min(float(ow - 1), cx)), max(0.0, min(float(oh - 1), cy))),
            size=(w, h), angle_deg=angle_deg, polygon=polygon, metadata={},
        )

    def _canon(self, cx, cy, w, h, ad):
        ad = normalize_angle_deg(ad)
        if h > w:
            w, h = h, w
            ad = normalize_angle_deg(ad + 90.0)
        return float(cx), float(cy), float(w), float(h), float(ad)

    @staticmethod
    def _clip(polygon, width, height):
        mx, my = float(max(width - 1, 0)), float(max(height - 1, 0))
        return tuple(
            (float(max(0.0, min(mx, x))), float(max(0.0, min(my, y))))
            for x, y in polygon
        )

    def _fix_layout(self, preds):
        emin = 7 if self.settings.model_has_objectness else 6
        if preds.shape[1] < emin:
            if preds.shape[0] >= emin:
                preds = preds.T
            if preds.shape[1] < emin:
                raise ValueError(f"Cannot fix prediction layout: {preds.shape}")
        return preds


class _ONNXPipeline:
    def __init__(self, engine: ONNXInferenceEngine, settings: AppSettings):
        self._e = engine
        self._s = settings

    def infer_frame(self, frame, frame_index=0, use_tracking=False):
        return self._e.infer_frame(frame, frame_index=frame_index, use_tracking=use_tracking)

    def annotate_frame(self, frame, detections):
        return self._e.annotate_frame(frame, detections)

    async def async_infer_frame(self, frame, frame_index=0, use_tracking=False):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _EXECUTOR,
            lambda: self._e.infer_frame(frame, frame_index=frame_index, use_tracking=use_tracking),
        )

    async def async_annotate_frame(self, frame, detections):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _EXECUTOR,
            lambda: self._e.annotate_frame(frame, detections),
        )

    async def analyze_video(self, video_path, sample_every=1, max_frames=None, use_tracking=False):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _EXECUTOR,
            lambda: self._e.infer_video(
                video_path=video_path,
                sample_every=sample_every,
                max_frames=max_frames,
                use_tracking=use_tracking,
            ),
        )

    async def annotate_video(self, video_path, output_path, sample_every=1, max_frames=None, use_tracking=False):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            _EXECUTOR,
            lambda: self._e.infer_video(
                video_path=video_path,
                output_path=output_path,
                sample_every=sample_every,
                max_frames=max_frames,
                use_tracking=use_tracking,
            ),
        )

    def close(self):
        self._e.close()


@lru_cache(maxsize=4)
def _shared_engine(settings: AppSettings) -> ONNXInferenceEngine:
    return ONNXInferenceEngine(session=get_shared_session(settings), settings=settings)


def create_pipeline(settings: AppSettings) -> _ONNXPipeline:
    return _ONNXPipeline(engine=_shared_engine(settings), settings=settings)
