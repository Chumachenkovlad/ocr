"""Tests for main application factory."""

from __future__ import annotations

from src.config import Settings
from src.main import _configure_logging, create_app


class TestConfigureLogging:
    def test_debug_level(self, test_settings: Settings) -> None:
        _configure_logging(test_settings)

    def test_info_level(self, test_settings: Settings) -> None:
        settings = test_settings.model_copy(update={"log_level": "INFO"})
        _configure_logging(settings)

    def test_unknown_level_defaults_to_info(self, test_settings: Settings) -> None:
        settings = test_settings.model_copy(update={"log_level": "UNKNOWN"})
        _configure_logging(settings)


class TestCreateApp:
    def test_creates_fastapi_app(self, test_settings: Settings) -> None:
        app = create_app(settings=test_settings)
        assert app.title == "OCR Paddle Service"

    def test_includes_routes(self, test_settings: Settings) -> None:
        app = create_app(settings=test_settings)
        paths = [r.path for r in app.routes]
        assert "/healthz" in paths
        assert "/api/v1/ocr" in paths
