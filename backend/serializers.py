from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile

from backend.schemas import DetectionSchema, InferenceImageResponseSchema
from backend.video_schemas import VideoAnalyzeResponseSchema
from inference.contracts import DetectionResult, VideoInferenceResult


def _seq(v: Any) -> bool:
    return isinstance(v, Sequence) and not isinstance(v, (str, bytes, bytearray))


def _floats(v: Any, n: int | None = None) -> list[float] | None:
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if not _seq(v):
        return None
    r = [float(x) for x in v]
    return r if n is None or len(r) == n else None


def _polygon(v: Any) -> list[list[float]] | None:
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if not _seq(v):
        return None
    items = list(v)
    if not items:
        return None
    if all(isinstance(x, (int, float, np.integer, np.floating)) for x in items):
        if len(items) != 8:
            return None
        return [[float(items[i]), float(items[i+1])] for i in range(0, 8, 2)]
    pts = []
    for p in items:
        if isinstance(p, np.ndarray):
            p = p.tolist()
        if _seq(p) and len(list(p)) == 2:
            pts.append([float(list(p)[0]), float(list(p)[1])])
    return pts or None


def _label(raw: str) -> str:
    s = str(raw).strip()
    return s if s else "unknown"


def _normalize(det: DetectionResult | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(det, DetectionResult):
        bbox = _floats(det.bbox, 4) or [0.0, 0.0, 0.0, 0.0]
        cx = _floats(getattr(det, "center", None), 2)
        sz = _floats(getattr(det, "size", None), 2)
        if cx is None:
            x1, y1, x2, y2 = bbox
            cx = [x1 + (x2-x1)/2, y1 + (y2-y1)/2]
        if sz is None:
            x1, y1, x2, y2 = bbox
            sz = [max(0.0, x2-x1), max(0.0, y2-y1)]
        ang = getattr(det, "angle_deg", getattr(det, "angle", None))
        tid = getattr(det, "track_id", None)
        ins = getattr(det, "inside_polygon", None)
        return {
            "class_id": int(det.class_id),
            "label": _label(getattr(det, "label", getattr(det, "class_name", ""))),
            "score": float(getattr(det, "score", getattr(det, "confidence", 0.0))),
            "bbox": bbox, "center": cx, "size": sz,
            "angle_deg": float(ang) if ang is not None else None,
            "polygon": _polygon(getattr(det, "polygon", None)),
            "track_id": int(tid) if tid is not None else None,
            "inside_polygon": bool(ins) if ins is not None else None,
        }
    if isinstance(det, Mapping):
        def mg(*keys, default=None):
            for k in keys:
                if k in det:
                    return det[k]
            return default
        bbox = _floats(mg("bbox"), 4) or [0.0, 0.0, 0.0, 0.0]
        cx = _floats(mg("center"), 2)
        sz = _floats(mg("size"), 2)
        if cx is None:
            x1, y1, x2, y2 = bbox
            cx = [x1 + (x2-x1)/2, y1 + (y2-y1)/2]
        if sz is None:
            x1, y1, x2, y2 = bbox
            sz = [max(0.0, x2-x1), max(0.0, y2-y1)]
        ang = mg("angle_deg", "angle")
        tid = mg("track_id")
        ins = mg("inside_polygon")
        return {
            "class_id": int(mg("class_id", default=0)),
            "label": _label(str(mg("label", "class_name", default=""))),
            "score": float(mg("score", "confidence", default=0.0)),
            "bbox": bbox, "center": cx, "size": sz,
            "angle_deg": float(ang) if ang is not None else None,
            "polygon": _polygon(mg("polygon", "poly8")),
            "track_id": int(tid) if tid is not None else None,
            "inside_polygon": bool(ins) if ins is not None else None,
        }
    raise TypeError(f"Unsupported detection type: {type(det)!r}")


def build_image_response(image: np.ndarray, detections: Sequence) -> InferenceImageResponseSchema:
    h, w = image.shape[:2]
    dets = [DetectionSchema.model_validate(_normalize(d)) for d in detections]
    return InferenceImageResponseSchema(width=int(w), height=int(h), detections=dets, count=len(dets))


def build_video_response(payload) -> VideoAnalyzeResponseSchema:
    if isinstance(payload, VideoAnalyzeResponseSchema):
        return payload

    if isinstance(payload, VideoInferenceResult):
        fps = payload.fps if payload.fps and payload.fps > 0 else 1.0
        total = payload.total_frames or 0
        frames = []
        for f in payload.frames:
            frames.append({
                "frame_index": f.frame_index,
                "timestamp_sec": round(f.frame_index / fps, 4),
                "detections": [_normalize(d) for d in f.detections],
            })
        total_dets = sum(len(f["detections"]) for f in frames)
        unique_ids = len({d["track_id"] for f in frames for d in f["detections"] if d["track_id"] is not None})
        ff = payload.frames[0] if payload.frames else None
        return VideoAnalyzeResponseSchema.model_validate({
            "metadata": {
                "fps": fps, "width": ff.width if ff else 0, "height": ff.height if ff else 0,
                "total_frames": total, "processed_frames": payload.processed_frames,
                "duration_sec": round(total / fps, 4) if total else 0.0,
            },
            "frames": frames, "total_detections": total_dets, "unique_track_ids": unique_ids,
            "output_video_path": str(payload.output_path) if payload.output_path else None,
        })

    p = payload.model_dump(mode="python") if hasattr(payload, "model_dump") else dict(payload)
    frames = []
    for f in p.get("frames", []):
        f = f.model_dump(mode="python") if hasattr(f, "model_dump") else dict(f)
        frames.append({
            "frame_index": int(f["frame_index"]),
            "timestamp_sec": float(f["timestamp_sec"]),
            "detections": [_normalize(d) for d in f.get("detections", [])],
        })
    total_dets = int(p.get("total_detections", sum(len(f["detections"]) for f in frames)))
    unique_ids = int(p.get("unique_track_ids", len({
        d["track_id"] for f in frames for d in f["detections"] if d.get("track_id") is not None
    })))
    return VideoAnalyzeResponseSchema.model_validate({
        "metadata": p["metadata"], "frames": frames,
        "total_detections": total_dets, "unique_track_ids": unique_ids,
        "output_video_path": str(p["output_video_path"]) if p.get("output_video_path") else None,
    })


def safe_remove(path: str | None) -> None:
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except Exception:
        pass


def _check_presence(file: UploadFile, label: str) -> None:
    if not file or not file.filename:
        raise HTTPException(400, f"Файл «{label}» не передан")


def _check_content_type(file: UploadFile, prefix: str, label: str) -> None:
    ct = (file.content_type or "").lower().strip()
    neutral = {"application/octet-stream", "binary/octet-stream", ""}
    if ct not in neutral and not ct.startswith(prefix):
        raise HTTPException(415, f"Неподдерживаемый тип для «{label}»: {file.content_type}")


async def decode_image_upload(file: UploadFile, max_size_bytes: int) -> np.ndarray:
    _check_presence(file, "изображение")
    _check_content_type(file, "image/", "изображение")
    data = await file.read(max_size_bytes + 1)
    if not data:
        raise HTTPException(400, "Изображение пустое")
    if len(data) > max_size_bytes:
        raise HTTPException(413, f"Изображение превышает {max_size_bytes} байт")
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Не удалось декодировать изображение")
    return img


async def save_upload_to_temp(file: UploadFile, default_suffix: str, max_size_bytes: int,
                               chunk_size: int, prefix: str, label: str) -> str:
    _check_presence(file, label)
    _check_content_type(file, prefix, label)
    suffix = os.path.splitext(file.filename or "")[1] or default_suffix
    total = 0
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            tmp = f.name
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_size_bytes:
                    raise HTTPException(413, f"«{label}» превышает {max_size_bytes} байт")
                f.write(chunk)
        if not tmp or os.path.getsize(tmp) == 0:
            safe_remove(tmp)
            raise HTTPException(400, f"Файл «{label}» пустой")
        return tmp
    except Exception:
        safe_remove(tmp)
        raise
