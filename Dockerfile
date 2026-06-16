FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH="/install/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libx264-dev \
        libglib2.0-0 \
        ffmpeg \
        fonts-dejavu-core \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1001 appgroup \
 && useradd --uid 1001 --gid 1001 --no-create-home appuser

WORKDIR /app

COPY --from=builder /install /install

COPY backend/   ./backend/
COPY frontend/  ./frontend/
COPY gui/       ./gui/
COPY inference/ ./inference/

RUN mkdir -p /app/models /app/artifacts/tmp /app/artifacts/video \
 && chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--http", "httptools"]
