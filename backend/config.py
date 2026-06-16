from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass


def _csv(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if not value:
        return default
    items = tuple(p.strip() for p in value.split(",") if p.strip())
    return items or default


def _bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _polygon(value: str | None) -> tuple[tuple[float, float], ...]:
    if not value or not value.strip():
        return tuple()
    try:
        pts = []
        for pair in value.strip().split(";"):
            pair = pair.strip()
            if not pair:
                continue
            x, y = pair.split(",", 1)
            pts.append((float(x.strip()), float(y.strip())))
        return tuple(pts)
    except (ValueError, TypeError):
        return tuple()


@dataclass(frozen=True, slots=True)
class Settings:
    APP_NAME: str
    APP_VERSION: str
    API_PREFIX: str
    MODEL_PATH: str
    MODEL_INPUT_SIZE: int
    CONFIDENCE_THRESHOLD: float
    IOU_THRESHOLD: float
    ONNX_PROVIDERS: tuple[str, ...]
    MODEL_LABELS: tuple[str, ...]
    MODEL_HAS_OBJECTNESS: bool
    MODEL_ANGLE_IN_DEGREES: bool
    MODEL_OUTPUT_INDEX: int
    NORMALIZE_01: bool
    MODEL_PREDECODED: bool
    TRACKER_MAX_DISAPPEARED: int
    TRACKER_MAX_DISTANCE: float
    TRACKER_POLYGON: tuple[tuple[int, int], ...]
    TRACKER_HIGH_THRESH: float
    TRACKER_LOW_THRESH: float
    TRACKER_NEW_TRACK_THRESH: float
    TRACKER_MATCH_IOU_THRESHOLD: float
    TRACKER_MIN_HITS: int
    MIN_DET_AREA: float
    MAX_DET_ASPECT_RATIO: float | None
    DEFAULT_FPS: float
    PIPELINE_FACTORY: str
    CORS_ALLOW_ORIGINS: tuple[str, ...]
    MAX_IMAGE_SIZE_BYTES: int
    MAX_VIDEO_SIZE_BYTES: int
    UPLOAD_CHUNK_SIZE_BYTES: int
    TEMP_DIR: str
    VIDEO_OUTPUT_DIR: str
    ENABLE_LEGACY_HEALTH_ENDPOINT: bool

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            APP_NAME=os.getenv("APP_NAME", "Urban Infra Detection API"),
            APP_VERSION=os.getenv("APP_VERSION", "1.0.0"),
            API_PREFIX=os.getenv("API_PREFIX", "/api"),
            MODEL_PATH=os.getenv("MODEL_PATH", "models/model.onnx"),
            MODEL_INPUT_SIZE=int(os.getenv("MODEL_INPUT_SIZE", "1024")),
            CONFIDENCE_THRESHOLD=float(os.getenv("CONFIDENCE_THRESHOLD", "0.25")),
            IOU_THRESHOLD=float(os.getenv("IOU_THRESHOLD", "0.45")),
            ONNX_PROVIDERS=_csv(os.getenv("ONNX_PROVIDERS"), ("CPUExecutionProvider",)),
            MODEL_LABELS=_csv(os.getenv("MODEL_LABELS"), tuple()),
            MODEL_HAS_OBJECTNESS=_bool(os.getenv("MODEL_HAS_OBJECTNESS"), False),
            MODEL_ANGLE_IN_DEGREES=_bool(os.getenv("MODEL_ANGLE_IN_DEGREES"), False),
            MODEL_OUTPUT_INDEX=int(os.getenv("MODEL_OUTPUT_INDEX", "0")),
            NORMALIZE_01=_bool(os.getenv("NORMALIZE_01"), True),
            MODEL_PREDECODED=_bool(os.getenv("MODEL_PREDECODED"), True),
            TRACKER_MAX_DISAPPEARED=int(os.getenv("TRACKER_MAX_DISAPPEARED", "30")),
            TRACKER_MAX_DISTANCE=float(os.getenv("TRACKER_MAX_DISTANCE", "100.0")),
            TRACKER_POLYGON=_polygon(os.getenv("TRACKER_POLYGON")),
            TRACKER_HIGH_THRESH=float(os.getenv("TRACKER_HIGH_THRESH", "0.5")),
            TRACKER_LOW_THRESH=float(os.getenv("TRACKER_LOW_THRESH", "0.1")),
            TRACKER_NEW_TRACK_THRESH=float(os.getenv("TRACKER_NEW_TRACK_THRESH", "0.5")),
            TRACKER_MATCH_IOU_THRESHOLD=float(os.getenv("TRACKER_MATCH_IOU_THRESHOLD", "0.15")),
            TRACKER_MIN_HITS=int(os.getenv("TRACKER_MIN_HITS", "1")),
            MIN_DET_AREA=float(os.getenv("MIN_DET_AREA", "0.0")),
            MAX_DET_ASPECT_RATIO=float(os.getenv("MAX_DET_ASPECT_RATIO")) if os.getenv("MAX_DET_ASPECT_RATIO") else None,
            DEFAULT_FPS=float(os.getenv("DEFAULT_FPS", "30.0")),
            PIPELINE_FACTORY=os.getenv("PIPELINE_FACTORY", "inference.engine:create_pipeline"),
            CORS_ALLOW_ORIGINS=_csv(os.getenv("CORS_ALLOW_ORIGINS"), ("http://localhost:3000", "http://127.0.0.1:3000")),
            MAX_IMAGE_SIZE_BYTES=int(os.getenv("MAX_IMAGE_SIZE_BYTES", str(10 * 1024 * 1024))),
            MAX_VIDEO_SIZE_BYTES=int(os.getenv("MAX_VIDEO_SIZE_BYTES", str(200 * 1024 * 1024))),
            UPLOAD_CHUNK_SIZE_BYTES=int(os.getenv("UPLOAD_CHUNK_SIZE_BYTES", str(1024 * 1024))),
            TEMP_DIR=os.getenv("TEMP_DIR", "artifacts/tmp"),
            VIDEO_OUTPUT_DIR=os.getenv("VIDEO_OUTPUT_DIR", "artifacts/video"),
            ENABLE_LEGACY_HEALTH_ENDPOINT=_bool(os.getenv("ENABLE_LEGACY_HEALTH_ENDPOINT"), True),
        )

    def validate(self) -> None:
        if not self.API_PREFIX.startswith("/"):
            raise ValueError("API_PREFIX must start with '/'")
        if not self.PIPELINE_FACTORY or ":" not in self.PIPELINE_FACTORY:
            raise ValueError("PIPELINE_FACTORY must be 'module:function'")
        if self.MODEL_INPUT_SIZE <= 0:
            raise ValueError("MODEL_INPUT_SIZE must be positive")

    def ensure_dirs(self) -> None:
        Path(self.TEMP_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.VIDEO_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
