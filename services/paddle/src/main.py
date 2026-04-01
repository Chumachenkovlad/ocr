"""FastAPI application factory and entrypoint."""

from __future__ import annotations

import logging
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.api.dependencies import get_cached_settings, init_engine
from src.api.router import router
from src.config import Settings

logger = structlog.get_logger()

_LOG_LEVEL_MAP: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


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


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory.

    Args:
        settings: Optional settings override (useful for testing).
    """
    if settings is None:
        settings = get_cached_settings()

    _configure_logging(settings)

    app = FastAPI(
        title="OCR Paddle Service",
        description="OCR microservice using PaddleOCR for PDF/image text extraction",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.include_router(router)

    # Serve the OCR viewer frontend
    public_dir = Path(__file__).resolve().parent.parent / "public"
    if public_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(public_dir), html=True), name="static")

    @app.on_event("startup")
    async def startup_event() -> None:
        """Load OCR models at startup."""
        engine = init_engine(settings)
        try:
            engine.load_models()
            logger.info("app_started", models_loaded=True)
        except Exception:
            logger.warning("app_started_without_models", models_loaded=False)

    return app


# Default app instance for uvicorn
app = create_app()
