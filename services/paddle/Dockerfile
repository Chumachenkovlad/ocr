# syntax=docker/dockerfile:1

# ============================================================
# Stage: base — shared dependencies
# Build context must be repo root (for shared/ access)
# ============================================================
FROM python:3.10-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libmagic1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/paddle/pyproject.toml ./

# ============================================================
# Stage: cpu — CPU inference
# ============================================================
FROM base AS cpu

COPY shared/ /app/shared/
RUN pip install /app/shared

COPY services/paddle/src/ ./src/
COPY services/paddle/public/ ./public/

RUN pip install --no-cache-dir \
    paddlepaddle==3.2.0 \
    "paddleocr[doc-parser]==3.4.0" \
    && pip install --no-cache-dir .

ENV FLAGS_use_mkldnn=0 \
    FLAGS_allocator_strategy=auto_growth \
    FLAGS_call_stack_level=2 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    GLOG_v=0 \
    PYTHONFAULTHANDLER=1

RUN python -c "from paddleocr import PaddleOCR; PaddleOCR()" || true

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --retries=5 --start-period=120s \
    CMD curl -f http://localhost:8080/healthz || exit 1

ENTRYPOINT ["uvicorn", "src.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]

# ============================================================
# Stage: cpu-local — local dev on ARM64 (Apple Silicon) via RapidOCR
# ============================================================
FROM python:3.12-slim AS cpu-local

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/paddle/pyproject.toml ./

COPY shared/ /app/shared/
RUN pip install /app/shared

COPY services/paddle/src/ ./src/
COPY services/paddle/public/ ./public/

RUN pip install --no-cache-dir \
    "rapidocr-onnxruntime>=1.2.0" \
    && pip install --no-cache-dir . || true

ENV OCR_BACKEND=rapid \
    PYTHONFAULTHANDLER=1

EXPOSE 8080
ENTRYPOINT ["uvicorn", "src.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]

# ============================================================
# Stage: gpu — GPU inference with CUDA
# ============================================================
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 AS gpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libmagic1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/paddle/pyproject.toml ./
COPY shared/ /app/shared/
RUN pip install --no-cache-dir /app/shared

COPY services/paddle/src/ ./src/
COPY services/paddle/public/ ./public/

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ \
    && pip install --no-cache-dir "paddleocr[doc-parser]==3.4.0" \
    && pip install --no-cache-dir .

ENV FLAGS_use_mkldnn=0 \
    FLAGS_allocator_strategy=auto_growth \
    FLAGS_call_stack_level=2 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    GLOG_v=0

RUN python3 -c "from paddleocr import PaddleOCR; PaddleOCR()" || true

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --retries=5 --start-period=120s \
    CMD curl -f http://localhost:8080/healthz || exit 1

ENTRYPOINT ["uvicorn", "src.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
