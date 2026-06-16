from __future__ import annotations

import sys
import subprocess
import os

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from backend.schemas import HealthResponseSchema

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponseSchema)
async def health(request: Request) -> HealthResponseSchema:
    settings = getattr(request.app.state, "settings", None)
    runtime = getattr(request.app.state, "runtime", None)
    if settings is None:
        return HealthResponseSchema(status="not_initialized", app_name="unknown",
                                    version="unknown", model_path="unknown")
    return HealthResponseSchema(
        status="ok" if runtime is not None else "degraded",
        app_name=str(settings.APP_NAME),
        version=str(settings.APP_VERSION),
        model_path=str(settings.MODEL_PATH),
    )


@router.get("/debug", include_in_schema=False)
async def debug(request: Request):
    info: dict = {"python": sys.version}
    info["runtime"] = getattr(request.app.state, "runtime", None) is not None
    settings = getattr(request.app.state, "settings", None)
    if settings:
        info["labels_count"] = len(settings.MODEL_LABELS)
        info["labels"] = list(settings.MODEL_LABELS)
        info["model_predecoded"] = settings.MODEL_PREDECODED
        info["angle_in_degrees"] = settings.MODEL_ANGLE_IN_DEGREES
        info["confidence_threshold"] = settings.CONFIDENCE_THRESHOLD
    for pkg in ("fastapi", "pydantic", "cv2", "numpy", "onnxruntime", "PIL"):
        try:
            m = __import__(pkg)
            info[pkg] = getattr(m, "__version__", "ok")
        except ImportError:
            info[pkg] = "NOT INSTALLED"
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    info["pil_font"] = next((p for p in font_paths if os.path.exists(p)), "NOT FOUND")
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        info["ffmpeg"] = "ok" if r.returncode == 0 else "error"
    except Exception:
        info["ffmpeg"] = "NOT FOUND"
    return JSONResponse(content=info)
