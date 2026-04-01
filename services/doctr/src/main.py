"""FastAPI application for the docTR OCR service."""

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from doctr.io import DocumentFile
from fastapi import FastAPI, UploadFile, Form

from src.adapter import adapt_doctr_result
from src.config import Settings
from src.engine import DoctrEngine

settings = Settings()
engine: DoctrEngine | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    global engine  # noqa: PLW0603
    engine = DoctrEngine(device=settings.device)
    yield


app = FastAPI(title="docTR OCR Service", lifespan=lifespan)


@app.post("/api/v1/ocr")
async def ocr_endpoint(
    file: UploadFile,
    dpi: int = Form(default=300),
    language: str = Form(default="en"),
):
    content = await file.read()
    mime = file.content_type or ""

    t0 = time.monotonic()

    if "pdf" in mime:
        doc = DocumentFile.from_pdf(content)
    else:
        doc = DocumentFile.from_images(content)

    exported = engine.ocr(doc)
    latency_ms = (time.monotonic() - t0) * 1000

    return adapt_doctr_result(exported, latency_ms)


@app.get("/healthz")
async def health():
    return {"status": "ok", "provider": "doctr"}
