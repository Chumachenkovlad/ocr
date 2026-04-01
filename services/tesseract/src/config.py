"""Application configuration via environment variables."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Tesseract OCR service configuration loaded from environment variables."""

    model_config = {"env_prefix": "OCR_"}

    host: str = Field(default="0.0.0.0", description="Server bind host")  # noqa: S104
    port: int = Field(default=8080, ge=1, le=65535, description="Server bind port")
    tesseract_binary: str = Field(default="tesseract", description="Path to tesseract binary")
    tesseract_lang: str = Field(default="eng", description="Default Tesseract language")
    tesseract_dpi: int = Field(default=300, ge=72, le=1200, description="Default rendering DPI")
    tesseract_timeout: int = Field(
        default=30, ge=5, le=300, description="Per-page Tesseract timeout in seconds"
    )
    max_file_size_mb: int = Field(
        default=50, ge=1, le=500, description="Max upload file size in MB"
    )
    max_pages: int = Field(default=100, ge=1, le=1000, description="Max pages to process per PDF")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Logging level"
    )

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


def get_settings() -> Settings:
    """Create settings instance from environment."""
    return Settings()
