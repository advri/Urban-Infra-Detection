from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import onnxruntime as ort


@dataclass(frozen=True)
class AppSettings:
    model_path: Path = Path("models/model.onnx")
    labels: tuple[str, ...] = tuple()
    input_size: int = 1024
    normalize_01: bool = True
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    model_output_index: int = 0
    model_has_objectness: bool = False
    model_angle_in_degrees: bool = False
    model_predecoded: bool = True
    providers: tuple[str, ...] = ("CPUExecutionProvider",)
    # ORT threading: 0 = auto (ORT решает сам)
    intra_op_num_threads: int = 0
    inter_op_num_threads: int = 0
    graph_optimization_level: str = "ORT_ENABLE_ALL"
    default_fps: float = 30.0
    tracker_max_disappeared: int = 30
    tracker_max_distance: float = 100.0
    tracker_polygon: tuple[tuple[float, float], ...] = tuple()
    tracker_high_thresh: float = 0.5
    tracker_low_thresh: float = 0.1
    tracker_new_track_thresh: float = 0.5
    tracker_match_iou_threshold: float = 0.15
    tracker_min_hits: int = 1
    min_det_area: float = 0.0
    max_det_aspect_ratio: float | None = None


def get_settings(
    model_path: str | Path | None = None,
    labels: Sequence[str] | None = None,
    **overrides,
) -> AppSettings:
    data: dict = {}
    if model_path is not None:
        data["model_path"] = Path(model_path)
    if labels is not None:
        data["labels"] = tuple(str(x) for x in labels)
    data.update(overrides)
    return AppSettings(**data)


@lru_cache(maxsize=8)
def get_shared_session(settings: AppSettings) -> ort.InferenceSession:
    if not settings.model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {settings.model_path}")

    opts = ort.SessionOptions()

    level_map = {
        "ORT_DISABLE_ALL":     ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "ORT_ENABLE_BASIC":    ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "ORT_ENABLE_EXTENDED": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "ORT_ENABLE_ALL":      ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    opts.graph_optimization_level = level_map.get(
        settings.graph_optimization_level,
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    )

    import os
    cpu_count = os.cpu_count() or 1
    n_threads = settings.intra_op_num_threads if settings.intra_op_num_threads > 0 else cpu_count
    opts.intra_op_num_threads = n_threads
    opts.inter_op_num_threads = settings.inter_op_num_threads if settings.inter_op_num_threads > 0 else 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    opts.enable_profiling = False

    return ort.InferenceSession(
        settings.model_path.as_posix(),
        sess_options=opts,
        providers=list(settings.providers),
    )
