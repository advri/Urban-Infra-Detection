from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DetectionSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_id: int = Field(ge=0)
    label: str
    score: float = Field(ge=0.0)
    bbox: list[float] = Field(min_length=4, max_length=4)
    center: list[float] | None = Field(default=None, min_length=2, max_length=2)
    size: list[float] | None = Field(default=None, min_length=2, max_length=2)
    angle_deg: float | None = None
    polygon: list[list[float]] | None = None
    track_id: int | None = None
    inside_polygon: bool | None = None

    @field_validator("bbox")
    @classmethod
    def _bbox(cls, v):
        if len(v) != 4:
            raise ValueError("bbox must have 4 values")
        return [float(x) for x in v]

    @field_validator("center")
    @classmethod
    def _center(cls, v):
        return [float(x) for x in v] if v is not None else None

    @field_validator("size")
    @classmethod
    def _size(cls, v):
        return [float(x) for x in v] if v is not None else None

    @field_validator("polygon")
    @classmethod
    def _polygon(cls, v):
        if v is None:
            return None
        return [[float(p[0]), float(p[1])] for p in v if len(p) == 2]


class InferenceImageResponseSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width: int = Field(ge=1)
    height: int = Field(ge=1)
    detections: list[DetectionSchema]
    count: int = Field(ge=0)

    @model_validator(mode="after")
    def _count(self):
        if self.count != len(self.detections):
            raise ValueError("count mismatch")
        return self


class HealthResponseSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "degraded", "not_initialized"]
    app_name: str
    version: str
    model_path: str
