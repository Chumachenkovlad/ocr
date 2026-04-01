"""Surya OCR FastAPI service."""

from __future__ import annotations

import io
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Form, HTTPException, UploadFile
from PIL import Image

from src.adapter import adapt_surya_result
from src.config import Settings
from src.engine import SuryaEngine
from src.pdf_renderer import pdf_to_images

settings = Settings()

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

engine: SuryaEngine | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    global engine  # noqa: PLW0603
    logger.info("Starting Surya OCR engine (device=%s)", settings.device)
    engine = SuryaEngine(device=settings.device)
    yield


app = FastAPI(title="Surya OCR Service", lifespan=lifespan)


@app.post("/api/v1/ocr")
async def ocr_endpoint(
    file: UploadFile,
    dpi: int = Form(default=300),
    language: str = Form(default="en"),
):
    """Accept an image or PDF and return structured OCR results."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready")

    content = await file.read()

    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {settings.max_file_size_mb} MB limit",
        )

    mime = file.content_type or ""

    t0 = time.monotonic()

    if "pdf" in mime:
        images = pdf_to_images(content, dpi=dpi)
        if len(images) > settings.max_pages:
            raise HTTPException(
                status_code=413,
                detail=f"PDF exceeds {settings.max_pages} page limit",
            )
    else:
        images = [Image.open(io.BytesIO(content))]

    languages = [[language]] * len(images)
    results = engine.ocr(images, languages)

    latency_ms = (time.monotonic() - t0) * 1000
    return adapt_surya_result(results, images, latency_ms)


@app.get("/healthz")
async def health():
    """Liveness/readiness probe."""
    return {"status": "ok", "provider": "surya"}
