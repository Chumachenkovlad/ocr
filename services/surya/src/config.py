"""Service configuration loaded from environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Surya OCR service settings.

    All values can be overridden via env vars prefixed with ``OCR_``,
    e.g. ``OCR_DEVICE=cuda``.
    """

    host: str = "0.0.0.0"
    port: int = 8080
    device: str = "cpu"
    max_file_size_mb: int = 50
    max_pages: int = 100
    log_level: str = "INFO"

    model_config = {"env_prefix": "OCR_"}
