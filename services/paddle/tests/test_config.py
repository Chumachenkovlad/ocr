"""Tests for configuration module."""

from __future__ import annotations

import pytest
from src.config import DeviceType, OCRMode, Settings


class TestSettings:
    """Test suite for Settings configuration."""

    def test_default_values(self) -> None:
        """Settings should have sensible defaults."""
        settings = Settings()
        assert settings.dpi == 300
        assert settings.max_file_size_mb == 50
        assert settings.max_pages == 100
        assert settings.device == DeviceType.CPU
        assert settings.mode == OCRMode.OCR
        assert settings.request_timeout_s == 120
        assert settings.max_concurrent == 4
        assert settings.log_level == "INFO"
        assert settings.lang == "auto"
        assert settings.port == 8080

    def test_max_file_size_bytes(self) -> None:
        """max_file_size_bytes should convert MB to bytes."""
        settings = Settings(max_file_size_mb=10)
        assert settings.max_file_size_bytes == 10 * 1024 * 1024

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Settings should be overridable via environment variables."""
        monkeypatch.setenv("OCR_DPI", "150")
        monkeypatch.setenv("OCR_MAX_PAGES", "50")
        monkeypatch.setenv("OCR_DEVICE", "gpu")
        monkeypatch.setenv("OCR_LOG_LEVEL", "DEBUG")

        settings = Settings()
        assert settings.dpi == 150
        assert settings.max_pages == 50
        assert settings.device == DeviceType.GPU
        assert settings.log_level == "DEBUG"

    def test_any_lang_accepted(self) -> None:
        """Any language code should pass (PaddleOCR handles 111 languages natively)."""
        for lang in ["auto", "en", "ch", "fr", "zz_custom"]:
            settings = Settings(lang=lang)
            assert settings.lang == lang

    def test_dpi_range_validation(self) -> None:
        """DPI must be between 72 and 600."""
        with pytest.raises(ValueError):
            Settings(dpi=10)
        with pytest.raises(ValueError):
            Settings(dpi=1000)

        settings = Settings(dpi=72)
        assert settings.dpi == 72
        settings = Settings(dpi=600)
        assert settings.dpi == 600
