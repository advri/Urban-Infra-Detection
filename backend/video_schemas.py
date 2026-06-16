from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.schemas import DetectionSchema


class FrameResultSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_index: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0.0)
    detections: list[DetectionSchema]


class VideoMetadataSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fps: float = Field(ge=0.0)
    width: int = Field(ge=0)
    height: int = Field(ge=0)
    total_frames: int = Field(ge=0)
    processed_frames: int = Field(ge=0)
    duration_sec: float = Field(ge=0.0)


class VideoAnalyzeResponseSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: VideoMetadataSchema
    frames: list[FrameResultSchema]
    total_detections: int = Field(ge=0)
    unique_track_ids: int = Field(ge=0)
    output_video_path: str | None = None

    @model_validator(mode="after")
    def _check(self):
        if self.metadata.processed_frames != len(self.frames):
            raise ValueError(
                f"processed_frames={self.metadata.processed_frames} != frames={len(self.frames)}"
            )
        return self
