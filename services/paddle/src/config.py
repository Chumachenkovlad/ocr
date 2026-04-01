"""Application configuration via environment variables."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class DeviceType(str, Enum):
    CPU = "cpu"
    GPU = "gpu"


class OCRMode(str, Enum):
    OCR = "ocr"
    STRUCTURE = "structure"


class OCRBackend(str, Enum):
    PADDLE = "paddle"
    RAPID = "rapid"
    AUTO = "auto"


class Settings(BaseSettings):
    """OCR service configuration loaded from environment variables."""

    model_config = {"env_prefix": "OCR_"}

    dpi: int = Field(default=300, ge=72, le=600, description="Default rendering DPI for PDFs")
    max_file_size_mb: int = Field(
        default=50, ge=1, le=500, description="Max upload file size in MB",
    )
    max_pages: int = Field(
        default=100, ge=1, le=1000, description="Max pages to process per request",
    )
    device: DeviceType = Field(default=DeviceType.CPU, description="Inference device")
    mode: OCRMode = Field(default=OCRMode.OCR, description="Default processing mode")
    backend: OCRBackend = Field(
        default=OCRBackend.AUTO,
        description="OCR backend: paddle (x86 prod), rapid (ARM64 local), auto (detect)",
    )
    request_timeout_s: int = Field(default=120, ge=10, le=600, description="Per-request timeout")
    max_concurrent: int = Field(default=4, ge=1, le=32, description="Max concurrent OCR operations")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Logging level"
    )
    model_dir: str = Field(default="/models", description="Custom model directory")
    lang: str = Field(default="auto", description="Default language for recognition")
    host: str = Field(default="0.0.0.0", description="Server bind host")  # noqa: S104
    port: int = Field(default=8080, ge=1, le=65535, description="Server bind port")

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


def get_settings() -> Settings:
    """Create settings instance from environment."""
    return Settings()
