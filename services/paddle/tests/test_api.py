"""Tests for API endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from src.config import Settings
from src.main import create_app


@pytest.fixture
def app_with_mock_engine(test_settings: Settings, mock_ocr_engine: MagicMock) -> TestClient:
    """Create test app with mocked OCR engine."""
    app = create_app(settings=test_settings)

    # Override dependencies
    from src.api import dependencies

    dependencies._engine_instance = mock_ocr_engine
    # Clear lru_cache so test_settings is used
    dependencies.get_cached_settings.cache_clear()

    app.dependency_overrides[dependencies.get_cached_settings] = lambda: test_settings

    return TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /healthz."""

    def test_health_ok(self, app_with_mock_engine: TestClient) -> None:
        response = app_with_mock_engine.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["models_loaded"] is True
        assert data["version"] == "0.2.0"

    def test_health_models_not_loaded(self, test_settings: Settings) -> None:
        app = create_app(settings=test_settings)
        engine = MagicMock()
        engine.models_loaded = False
        engine.backend.value = "rapid_ocr"

        from src.api import dependencies

        dependencies._engine_instance = engine

        client = TestClient(app)
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["models_loaded"] is False


class TestOCREndpoint:
    """Tests for POST /api/v1/ocr."""

    def test_empty_file_rejected(self, app_with_mock_engine: TestClient) -> None:
        response = app_with_mock_engine.post(
            "/api/v1/ocr",
            files={"file": ("test.pdf", b"", "application/pdf")},
        )
        assert response.status_code == 400

    def test_oversized_file_rejected(self, app_with_mock_engine: TestClient) -> None:
        # test_settings has max_file_size_mb=10
        huge_file = b"x" * (11 * 1024 * 1024)  # 11 MB

        with patch("src.api.router.magic") as mock_magic:
            mock_magic.from_buffer.return_value = "application/pdf"
            response = app_with_mock_engine.post(
                "/api/v1/ocr",
                files={"file": ("test.pdf", huge_file, "application/pdf")},
            )
        assert response.status_code == 413

    def test_unsupported_mime_rejected(self, app_with_mock_engine: TestClient) -> None:
        with patch("src.api.router.magic") as mock_magic:
            mock_magic.from_buffer.return_value = "text/plain"
            response = app_with_mock_engine.post(
                "/api/v1/ocr",
                files={"file": ("test.txt", b"Hello", "text/plain")},
            )
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_invalid_dpi_rejected(self, app_with_mock_engine: TestClient) -> None:
        with patch("src.api.router.magic") as mock_magic:
            mock_magic.from_buffer.return_value = "image/png"
            with patch("src.api.router.load_image_from_bytes"):
                response = app_with_mock_engine.post(
                    "/api/v1/ocr",
                    files={"file": ("test.png", b"\x89PNG\r\n\x1a\n", "image/png")},
                    data={"dpi": "9999"},
                )
        assert response.status_code == 400

    def test_invalid_mode_rejected(self, app_with_mock_engine: TestClient) -> None:
        with patch("src.api.router.magic") as mock_magic:
            mock_magic.from_buffer.return_value = "image/png"
            response = app_with_mock_engine.post(
                "/api/v1/ocr",
                files={"file": ("test.png", b"\x89PNG\r\n\x1a\n", "image/png")},
                data={"mode": "invalid"},
            )
        assert response.status_code == 400

    def test_successful_image_ocr(self, app_with_mock_engine: TestClient) -> None:
        """Full happy-path test with a mock image."""
        import io

        from PIL import Image

        # Create a small test image
        img = Image.new("RGB", (100, 100), color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        with patch("src.api.router.magic") as mock_magic:
            mock_magic.from_buffer.return_value = "image/png"
            with patch("src.api.router.load_image_from_bytes") as mock_load:
                mock_load.return_value = img
                response = app_with_mock_engine.post(
                    "/api/v1/ocr",
                    files={"file": ("test.png", img_bytes, "image/png")},
                )

        assert response.status_code == 200
        data = response.json()
        assert data["provider"] == "paddle"
        assert data["latency_ms"] >= 0
        assert len(data["pages"]) == 1
        page = data["pages"][0]
        assert page["page_number"] == 0
        assert page["width"] == 100
        assert page["height"] == 100

    def test_page_range_parsing(self, app_with_mock_engine: TestClient) -> None:
        """Invalid page range should return 400."""
        with patch("src.api.router.magic") as mock_magic:
            mock_magic.from_buffer.return_value = "image/png"
            with patch("src.api.router.load_image_from_bytes"):
                response = app_with_mock_engine.post(
                    "/api/v1/ocr",
                    files={"file": ("test.png", b"\x89PNG", "image/png")},
                    data={"pages": "5-2"},
                )
        assert response.status_code == 400


class TestParsePageRange:
    """Test page range parsing utility."""

    def test_all_pages(self) -> None:
        from src.api.router import _parse_page_range

        first, last = _parse_page_range("all", 100)
        assert first is None
        assert last is None

    def test_single_page(self) -> None:
        from src.api.router import _parse_page_range

        first, last = _parse_page_range("3", 100)
        assert first == 3
        assert last == 3

    def test_page_range(self) -> None:
        from src.api.router import _parse_page_range

        first, last = _parse_page_range("2-5", 100)
        assert first == 2
        assert last == 5

    def test_invalid_range(self) -> None:
        from src.api.router import _parse_page_range

        with pytest.raises(ValueError):
            _parse_page_range("5-2", 100)

    def test_zero_page_rejected(self) -> None:
        from src.api.router import _parse_page_range

        with pytest.raises(ValueError):
            _parse_page_range("0", 100)
