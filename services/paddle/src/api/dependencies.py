"""FastAPI dependency injection for OCR engine and settings."""

from __future__ import annotations

from functools import lru_cache

from src.config import Settings, get_settings
from src.core.engine import OCREngine


@lru_cache(maxsize=1)
def get_cached_settings() -> Settings:
    """Cached settings singleton."""
    return get_settings()


_engine_instance: OCREngine | None = None


def get_ocr_engine() -> OCREngine:
    """Get the singleton OCR engine instance.

    The engine must be initialized via init_engine() during app startup.
    """
    if _engine_instance is None:
        msg = "OCR engine not initialized. Call init_engine() during startup."
        raise RuntimeError(msg)
    return _engine_instance


def init_engine(settings: Settings) -> OCREngine:
    """Initialize the global OCR engine singleton."""
    global _engine_instance
    _engine_instance = OCREngine(settings)
    return _engine_instance
