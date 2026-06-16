from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from backend.dependencies import close_runtime, create_runtime, get_settings
from backend.routes_health import router as health_router
from backend.routes_inference import router as inference_router
from backend.routes_video import router as video_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).resolve().parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.runtime = None

    log.info("MODEL_PATH      : %s", settings.MODEL_PATH)
    log.info("MODEL_PREDECODED: %s", settings.MODEL_PREDECODED)
    log.info("ANGLE_IN_DEGREES: %s", settings.MODEL_ANGLE_IN_DEGREES)
    log.info("CONFIDENCE      : %s", settings.CONFIDENCE_THRESHOLD)
    log.info("LABELS (%d)     : %s", len(settings.MODEL_LABELS), settings.MODEL_LABELS)

    loop = asyncio.get_event_loop()
    runtime = await loop.run_in_executor(None, create_runtime, settings)
    app.state.runtime = runtime
    log.info("Runtime ready — labels=%d", len(runtime.pipeline._e.labels))
    yield
    close_runtime(runtime)
    app.state.runtime = None


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION, lifespan=lifespan)
    app.state.settings = settings
    app.state.runtime = None

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.CORS_ALLOW_ORIGINS),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router, prefix=settings.API_PREFIX)
    app.include_router(inference_router, prefix=settings.API_PREFIX)
    app.include_router(video_router, prefix=settings.API_PREFIX)
    if settings.ENABLE_LEGACY_HEALTH_ENDPOINT:
        app.include_router(health_router)

    gui_dir = _BASE_DIR / "gui"
    frontend_dir = _BASE_DIR / "frontend"
    if gui_dir.is_dir():
        app.mount("/gui", StaticFiles(directory=str(gui_dir)), name="gui")
    if frontend_dir.is_dir():
        app.mount("/frontend", StaticFiles(directory=str(frontend_dir)), name="frontend")

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(url="/gui/index.html")

    @app.get("/gui/index.html", include_in_schema=False)
    async def gui_index():
        p = gui_dir / "index.html"
        if not p.exists():
            from fastapi import HTTPException
            raise HTTPException(404, "GUI не найден")
        return FileResponse(str(p), media_type="text/html")

    return app


app = create_app()
