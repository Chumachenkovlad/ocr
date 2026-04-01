"""Tests for the config module."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from src.config import Settings, get_settings


class TestSettings:
    """Tests for Settings."""

    def test_defaults(self) -> None:
        settings = Settings()
        assert settings.host == "0.0.0.0"
        assert settings.port == 8080
        assert settings.tesseract_binary == "tesseract"
        assert settings.tesseract_lang == "eng"
        assert settings.tesseract_dpi == 300
        assert settings.tesseract_timeout == 30
        assert settings.max_file_size_mb == 50
        assert settings.max_pages == 100
        assert settings.log_level == "INFO"

    def test_max_file_size_bytes(self) -> None:
        settings = Settings(max_file_size_mb=10)
        assert settings.max_file_size_bytes == 10 * 1024 * 1024

    def test_env_prefix(self) -> None:
        with patch.dict(os.environ, {"OCR_TESSERACT_LANG": "deu", "OCR_PORT": "9090"}):
            settings = Settings()
            assert settings.tesseract_lang == "deu"
            assert settings.port == 9090

    def test_get_settings(self) -> None:
        settings = get_settings()
        assert isinstance(settings, Settings)
