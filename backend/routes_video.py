from __future__ import annotations

import os
import traceback
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from backend.dependencies import RuntimeComponents, get_runtime
from backend.serializers import build_video_response, safe_remove, save_upload_to_temp
from backend.video_schemas import VideoAnalyzeResponseSchema

router = APIRouter(prefix="/video", tags=["video"])


@router.post("/analyze", response_model=VideoAnalyzeResponseSchema)
async def analyze_video(
    file: UploadFile = File(...),
    sample_every: int = Query(1, ge=1),
    max_frames: int | None = Query(None, ge=1),
    use_tracking: bool = Query(True),
    runtime: RuntimeComponents = Depends(get_runtime),
) -> VideoAnalyzeResponseSchema:
    tmp = None
    try:
        tmp = await save_upload_to_temp(file, ".mp4", runtime.settings.MAX_VIDEO_SIZE_BYTES,
                                        runtime.settings.UPLOAD_CHUNK_SIZE_BYTES, "video/", "Видео")
        result = await runtime.pipeline.analyze_video(
            video_path=tmp, sample_every=sample_every, max_frames=max_frames, use_tracking=use_tracking,
        )
        return build_video_response(result)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Ошибка анализа видео: {type(e).__name__}: {e}")
    finally:
        await file.close()
        safe_remove(tmp)


@router.post("/annotated")
async def annotated_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    sample_every: int = Query(1, ge=1),
    max_frames: int | None = Query(None, ge=1),
    use_tracking: bool = Query(True),
    runtime: RuntimeComponents = Depends(get_runtime),
):
    tmp = out = None
    try:
        tmp = await save_upload_to_temp(file, ".mp4", runtime.settings.MAX_VIDEO_SIZE_BYTES,
                                        runtime.settings.UPLOAD_CHUNK_SIZE_BYTES, "video/", "Видео")
        out_dir = Path(runtime.settings.VIDEO_OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"annotated_{uuid.uuid4().hex}.mp4"
        await runtime.pipeline.annotate_video(
            video_path=tmp, output_path=str(out),
            sample_every=sample_every, max_frames=max_frames, use_tracking=use_tracking,
        )
        if not out.exists() or os.path.getsize(out) == 0:
            raise HTTPException(500, "Аннотированное видео не создано")
        background_tasks.add_task(safe_remove, str(out))
        return FileResponse(path=str(out), filename=out.name, media_type="video/mp4",
                            background=background_tasks)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Ошибка аннотирования видео: {type(e).__name__}: {e}")
    finally:
        await file.close()
        safe_remove(tmp)
