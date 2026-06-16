from __future__ import annotations

import asyncio
import traceback

import cv2
from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile

from backend.dependencies import RuntimeComponents, get_runtime
from backend.schemas import InferenceImageResponseSchema
from backend.serializers import build_image_response, decode_image_upload

router = APIRouter(prefix="/inference", tags=["inference"])


@router.post("/image", response_model=InferenceImageResponseSchema)
async def infer_image(
    file: UploadFile = File(...),
    runtime: RuntimeComponents = Depends(get_runtime),
) -> InferenceImageResponseSchema:
    try:
        image = await decode_image_upload(file, runtime.settings.MAX_IMAGE_SIZE_BYTES)
        result = await runtime.pipeline.async_infer_frame(image, use_tracking=False)
        return build_image_response(image, result.detections)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Ошибка инференса: {type(e).__name__}: {e}")
    finally:
        await file.close()


@router.post("/annotated")
async def annotated_image(
    file: UploadFile = File(...),
    runtime: RuntimeComponents = Depends(get_runtime),
) -> Response:
    try:
        image = await decode_image_upload(file, runtime.settings.MAX_IMAGE_SIZE_BYTES)
        result = await runtime.pipeline.async_infer_frame(image, use_tracking=False)
        annotated = await runtime.pipeline.async_annotate_frame(image, result.detections)
        loop = asyncio.get_event_loop()
        ok, encoded = await loop.run_in_executor(None, lambda: cv2.imencode(".jpg", annotated))
        if not ok:
            raise HTTPException(500, "Ошибка кодирования изображения")
        return Response(content=encoded.tobytes(), media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Ошибка аннотирования: {type(e).__name__}: {e}")
    finally:
        await file.close()
