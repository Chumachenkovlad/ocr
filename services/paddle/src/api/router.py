"""API router with OCR and health endpoints."""

from __future__ import annotations

import base64
import io
import time
from typing import Annotated

import magic
import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from PIL import Image

from src.api.dependencies import get_cached_settings, get_ocr_engine
from src.api.schemas import HealthResponse, OCRResponse, PageSchema
from src.config import OCRMode, Settings
from src.core.adapter import adapt_ocr_result
from src.core.engine import OCREngine, OCREngineError
from src.core.pdf_renderer import PDFRenderError, load_image_from_bytes, render_pdf_pages

from ocr_schema.models import (
    BoundingBox as SharedBBox,
    OcrBlock as SharedBlock,
    OcrLine as SharedLine,
    OcrPage as SharedPage,
    OcrResult,
    OcrWord as SharedWord,
)

logger = structlog.get_logger()

router = APIRouter()

ALLOWED_MIME_TYPES = frozenset({
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/tiff",
})

PDF_MIME = "application/pdf"


@router.get("/healthz", response_model=HealthResponse)
async def health_check(
    engine: Annotated[OCREngine, Depends(get_ocr_engine)],
) -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        models_loaded=engine.models_loaded,
        version="0.2.0",
        backend=engine.backend.value,
    )


def _parse_page_range(pages_str: str, max_pages: int) -> tuple[int | None, int | None]:
    """Parse page range string into (first_page, last_page) 1-based.

    Supports formats: "all", "1-5", "3", "1,3,5" (first and last only).
    """
    if pages_str.strip().lower() == "all":
        return None, None

    pages_str = pages_str.strip()

    if "-" in pages_str:
        parts = pages_str.split("-", 1)
        first = int(parts[0])
        last = int(parts[1])
        if first < 1 or last < first:
            msg = f"Invalid page range: {pages_str}"
            raise ValueError(msg)
        if last - first + 1 > max_pages:
            msg = f"Page range exceeds max_pages ({max_pages})"
            raise ValueError(msg)
        return first, last

    # Single page number
    page = int(pages_str)
    if page < 1:
        msg = f"Page number must be >= 1, got {page}"
        raise ValueError(msg)
    return page, page


@router.post("/api/v1/ocr", response_model=OcrResult)
async def ocr_endpoint(
    file: Annotated[UploadFile, File(description="PDF or image file")],
    engine: Annotated[OCREngine, Depends(get_ocr_engine)],
    settings: Annotated[Settings, Depends(get_cached_settings)],
    dpi: Annotated[int, Form()] = 300,
    mode: Annotated[str, Form()] = "structure",
    language: Annotated[str, Form()] = "auto",
    pages: Annotated[str, Form()] = "all",
) -> OcrResult:
    """Process a PDF or image file and return structured OCR results."""
    start_time = time.monotonic()

    # Read file bytes
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    # Validate file size
    if len(file_bytes) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds {settings.max_file_size_mb} MB limit",
        )

    # Validate file type via magic bytes
    detected_mime = magic.from_buffer(file_bytes[:2048], mime=True)
    if detected_mime not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {detected_mime}. "
            f"Allowed: {', '.join(sorted(ALLOWED_MIME_TYPES))}",
        )

    # Validate DPI
    if not 72 <= dpi <= 600:
        raise HTTPException(status_code=400, detail="DPI must be between 72 and 600")

    # Validate mode
    try:
        ocr_mode = OCRMode(mode)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode: {mode}. Must be 'ocr' or 'structure'",
        ) from None

    # Parse page range
    try:
        first_page, last_page = _parse_page_range(pages, settings.max_pages)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Convert to images
    try:
        if detected_mime == PDF_MIME:
            images = render_pdf_pages(
                file_bytes,
                dpi=dpi,
                first_page=first_page,
                last_page=last_page,
            )
        else:
            images = [load_image_from_bytes(file_bytes)]
    except PDFRenderError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if len(images) > settings.max_pages:
        raise HTTPException(
            status_code=400,
            detail=f"Document has too many pages (max: {settings.max_pages})",
        )

    # Process each page
    page_results = []
    base_page_index = (first_page - 1) if first_page else 0

    for i, image in enumerate(images):
        try:
            raw_result = await engine.process_image(image, mode=ocr_mode, language=language)
        except OCREngineError as exc:
            logger.error("page_ocr_failed", page=i, error=str(exc))
            raise HTTPException(status_code=500, detail=f"OCR failed on page {i}: {exc}") from exc

        # Encode the image used for OCR as base64 PNG so the frontend overlay
        # is guaranteed to be in the same coordinate space as the bounding boxes.
        page_image_b64 = _encode_image_base64(image)

        page = adapt_ocr_result(
            raw_result=raw_result,
            page_index=base_page_index + i,
            image_width=image.width,
            image_height=image.height,
            dpi=dpi,
            language=language if language != "auto" else "en",
            page_image_base64=page_image_b64,
        )
        page_results.append(page)

    elapsed_ms = int((time.monotonic() - start_time) * 1000)

    logger.info(
        "ocr_completed",
        page_count=len(page_results),
        elapsed_ms=elapsed_ms,
    )

    return to_shared_result(
        pages=page_results,
        provider="paddle",
        latency_ms=float(elapsed_ms),
    )


def to_shared_result(
    pages: list[PageSchema],
    provider: str,
    latency_ms: float,
    raw: dict | None = None,
) -> OcrResult:
    """Convert internal PageSchema list to the shared OcrResult format."""
    shared_pages = []
    for page in pages:
        shared_blocks = []
        for block in page.blocks:
            shared_lines = []
            for line in block.lines:
                shared_words = [
                    SharedWord(
                        text=w.text,
                        confidence=w.confidence,
                        bbox=SharedBBox(
                            x1=w.bbox.x,
                            y1=w.bbox.y,
                            x2=w.bbox.x + w.bbox.w,
                            y2=w.bbox.y + w.bbox.h,
                        ),
                    )
                    for w in line.words
                ]
                line_bbox = SharedBBox(
                    x1=line.bbox.x,
                    y1=line.bbox.y,
                    x2=line.bbox.x + line.bbox.w,
                    y2=line.bbox.y + line.bbox.h,
                )
                shared_lines.append(
                    SharedLine(
                        text=" ".join(w.text for w in shared_words),
                        words=shared_words,
                        bbox=line_bbox,
                    )
                )
            block_bbox = SharedBBox(
                x1=block.bbox.x,
                y1=block.bbox.y,
                x2=block.bbox.x + block.bbox.w,
                y2=block.bbox.y + block.bbox.h,
            )
            shared_blocks.append(
                SharedBlock(
                    block_type=block.block_type,
                    lines=shared_lines,
                    bbox=block_bbox,
                )
            )
        shared_pages.append(
            SharedPage(
                page_number=page.page_index,
                width=page.width_px,
                height=page.height_px,
                blocks=shared_blocks,
            )
        )
    return OcrResult(
        provider=provider,
        pages=shared_pages,
        raw=raw,
        latency_ms=latency_ms,
    )


def _encode_image_base64(image: Image.Image) -> str:
    """Encode a PIL Image as a base64 PNG string."""
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")
