from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class DetectionResult:
    class_id: int
    label: str
    score: float
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    size: tuple[float, float]
    angle_deg: float
    polygon: tuple[tuple[float, float], ...]
    track_id: int | None = None
    inside_polygon: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FrameInferenceResult:
    frame_index: int
    width: int
    height: int
    detections: list[DetectionResult]
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class VideoInferenceResult:
    video_path: Path
    output_path: Path | None
    fps: float
    total_frames: int | None
    processed_frames: int
    sampled_every: int
    frames: list[FrameInferenceResult]


@runtime_checkable
class PipelineProtocol(Protocol):
    def infer_frame(
        self,
        frame: Any,
        frame_index: int = 0,
        use_tracking: bool = False,
    ) -> FrameInferenceResult: ...

    def annotate_frame(
        self,
        frame: Any,
        detections: Sequence[DetectionResult],
    ) -> Any: ...

    async def analyze_video(
        self,
        video_path: str | Path,
        sample_every: int = 1,
        max_frames: int | None = None,
        use_tracking: bool = False,
    ) -> VideoInferenceResult: ...

    async def annotate_video(
        self,
        video_path: str | Path,
        output_path: str | Path,
        sample_every: int = 1,
        max_frames: int | None = None,
        use_tracking: bool = False,
    ) -> None: ...

    def close(self) -> None: ...
