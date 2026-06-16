from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from typing import Callable, cast

from fastapi import HTTPException, Request

from backend.config import Settings
from inference.contracts import PipelineProtocol
from inference.model_manager import AppSettings, get_settings as _make_app_settings


@dataclass(slots=True)
class RuntimeComponents:
    settings: Settings
    pipeline: PipelineProtocol


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings.from_env()
    s.validate()
    s.ensure_dirs()
    return s


def _resolve_factory(path: str) -> Callable:
    module_path, fn_name = path.split(":", 1)
    mod = import_module(module_path)
    fn = getattr(mod, fn_name, None)
    if fn is None or not callable(fn):
        raise RuntimeError(f"Pipeline factory not found: {path}")
    return fn


def _make_pipeline_settings(s: Settings) -> AppSettings:
    return _make_app_settings(
        model_path=s.MODEL_PATH,
        labels=list(s.MODEL_LABELS),
        input_size=s.MODEL_INPUT_SIZE,
        confidence_threshold=s.CONFIDENCE_THRESHOLD,
        iou_threshold=s.IOU_THRESHOLD,
        providers=s.ONNX_PROVIDERS,
        default_fps=s.DEFAULT_FPS,
        model_has_objectness=s.MODEL_HAS_OBJECTNESS,
        model_angle_in_degrees=s.MODEL_ANGLE_IN_DEGREES,
        model_output_index=s.MODEL_OUTPUT_INDEX,
        normalize_01=s.NORMALIZE_01,
        model_predecoded=s.MODEL_PREDECODED,
        tracker_max_disappeared=s.TRACKER_MAX_DISAPPEARED,
        tracker_max_distance=s.TRACKER_MAX_DISTANCE,
        tracker_polygon=s.TRACKER_POLYGON,
        tracker_high_thresh=s.TRACKER_HIGH_THRESH,
        tracker_low_thresh=s.TRACKER_LOW_THRESH,
        tracker_new_track_thresh=s.TRACKER_NEW_TRACK_THRESH,
        tracker_match_iou_threshold=s.TRACKER_MATCH_IOU_THRESHOLD,
        tracker_min_hits=s.TRACKER_MIN_HITS,
        min_det_area=s.MIN_DET_AREA,
        max_det_aspect_ratio=s.MAX_DET_ASPECT_RATIO,
    )


def create_runtime(settings: Settings | None = None) -> RuntimeComponents:
    s = settings or get_settings()
    app_settings = _make_pipeline_settings(s)
    factory = _resolve_factory(s.PIPELINE_FACTORY)
    pipeline = factory(app_settings)
    if pipeline is None:
        raise RuntimeError("Pipeline factory returned None")
    return RuntimeComponents(settings=s, pipeline=cast(PipelineProtocol, pipeline))


def close_runtime(runtime: RuntimeComponents | None) -> None:
    if runtime is None:
        return
    fn = getattr(runtime.pipeline, "close", None)
    if callable(fn):
        fn()


def get_runtime(request: Request) -> RuntimeComponents:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Runtime не инициализирован")
    return cast(RuntimeComponents, runtime)
