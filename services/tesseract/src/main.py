"""FastAPI application factory and entrypoint for the Tesseract OCR service."""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Annotated

import magic
import structlog
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.adapter import adapt_page, build_ocr_result
from src.config import Settings, get_settings
from src.engine import TesseractError, TesseractNotFoundError, recognize_image
from src.hocr_parser import parse_hocr
from src.pdf_renderer import PDFRenderError, load_image_from_bytes, render_pdf_pages

logger = structlog.get_logger()

_LOG_LEVEL_MAP: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}

ALLOWED_MIME_TYPES = frozenset({
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/tiff",
})

PDF_MIME = "application/pdf"


# ---------------------------------------------------------------------------
# Settings singleton
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_cached_settings() -> Settings:
    """Cached settings singleton."""
    return get_settings()


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

def _configure_logging(settings: Settings) -> None:
    """Configure structlog for JSON output."""
    level = _LOG_LEVEL_MAP.get(settings.log_level, logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer()
            if settings.log_level == "DEBUG"
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


# ---------------------------------------------------------------------------
# Page range parsing
# ---------------------------------------------------------------------------

def _parse_page_range(pages_str: str, max_pages: int) -> tuple[int | None, int | None]:
    """Parse page range string into (first_page, last_page) 1-based.

    Supports: ``"all"``, ``"1-5"``, ``"3"``.
    """
    stripped = pages_str.strip().lower()
    if stripped == "all":
        return None, None

    if "-" in stripped:
        parts = stripped.split("-", 1)
        first = int(parts[0])
        last = int(parts[1])
        if first < 1 or last < first:
            msg = f"Invalid page range: {pages_str}"
            raise ValueError(msg)
        if last - first + 1 > max_pages:
            msg = f"Page range exceeds max_pages ({max_pages})"
            raise ValueError(msg)
        return first, last

    page = int(stripped)
    if page < 1:
        msg = f"Page number must be >= 1, got {page}"
        raise ValueError(msg)
    return page, page


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Optional settings override (useful for testing).
    """
    if settings is None:
        settings = get_cached_settings()

    _configure_logging(settings)

    app = FastAPI(
        title="Tesseract OCR Service",
        description="OCR microservice using Tesseract for PDF/image text extraction",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ------------------------------------------------------------------
    # POST /api/v1/ocr
    # ------------------------------------------------------------------

    @app.post("/api/v1/ocr")
    async def ocr_endpoint(
        file: Annotated[UploadFile, File(description="PDF or image file")],
        dpi: Annotated[int, Form()] = 300,
        language: Annotated[str, Form()] = "eng",
        pages: Annotated[str, Form()] = "all",
    ) -> JSONResponse:
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
        if not 72 <= dpi <= 1200:
            raise HTTPException(status_code=400, detail="DPI must be between 72 and 1200")

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

        # Normalize ISO 639-1 codes to Tesseract's ISO 639-3
        _LANG_MAP = {"en": "eng", "de": "deu", "fr": "fra", "es": "spa", "uk": "ukr", "ru": "rus"}
        tess_lang = _LANG_MAP.get(language, language)

        # Process each page through Tesseract
        ocr_pages = []
        raw_hocr_list: list[str] = []
        base_page_index = (first_page - 1) if first_page else 0

        for i, image in enumerate(images):
            try:
                hocr_html = await recognize_image(
                    image, settings, lang=tess_lang, dpi=dpi,
                )
            except TesseractNotFoundError as exc:
                raise HTTPException(
                    status_code=503,
                    detail=str(exc),
                ) from exc
            except TesseractError as exc:
                logger.error("page_ocr_failed", page=i, error=str(exc))
                raise HTTPException(
                    status_code=500,
                    detail=f"OCR failed on page {i}: {exc}",
                ) from exc

            raw_hocr_list.append(hocr_html)
            hocr_page = parse_hocr(hocr_html, dpi=dpi)
            ocr_page = adapt_page(hocr_page, page_number=base_page_index + i)
            ocr_pages.append(ocr_page)

        elapsed_ms = (time.monotonic() - start_time) * 1000.0

        result = build_ocr_result(
            ocr_pages,
            raw_hocr=raw_hocr_list,
            latency_ms=round(elapsed_ms, 2),
        )

        logger.info(
            "ocr_completed",
            page_count=len(ocr_pages),
            elapsed_ms=round(elapsed_ms, 2),
        )

        return JSONResponse(content=result.model_dump(mode="json"))

    # ------------------------------------------------------------------
    # GET /healthz
    # ------------------------------------------------------------------

    @app.get("/healthz")
    async def health_check() -> dict:
        """Health check — verifies Tesseract binary is reachable."""
        from src.engine import _check_tesseract_available

        available = _check_tesseract_available(settings.tesseract_binary)
        status = "ok" if available else "degraded"
        return {
            "status": status,
            "tesseract_available": available,
            "version": "0.1.0",
        }

    return app


# Default app instance for uvicorn
app = create_app()
