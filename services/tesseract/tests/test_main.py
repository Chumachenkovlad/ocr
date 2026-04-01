"""Tests for the main FastAPI application."""

from __future__ import annotations

import subprocess
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.config import Settings
from src.main import create_app


@pytest.fixture()
def test_settings() -> Settings:
    return Settings(
        tesseract_binary="tesseract",
        tesseract_lang="eng",
        tesseract_dpi=300,
        tesseract_timeout=30,
        max_file_size_mb=10,
        max_pages=5,
        log_level="DEBUG",
    )


@pytest.fixture()
def app(test_settings: Settings):  # noqa: ANN201
    return create_app(settings=test_settings)


@pytest.fixture()
async def client(app):  # noqa: ANN001, ANN201
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


class TestHealthEndpoint:
    """Tests for GET /healthz."""

    async def test_health_ok(self, client: AsyncClient) -> None:
        with patch("src.engine._check_tesseract_available", return_value=True):
            response = await client.get("/healthz")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["tesseract_available"] is True
        assert data["version"] == "0.1.0"

    async def test_health_degraded(self, client: AsyncClient) -> None:
        with patch("src.engine._check_tesseract_available", return_value=False):
            response = await client.get("/healthz")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["tesseract_available"] is False


class TestOcrEndpoint:
    """Tests for POST /api/v1/ocr."""

    async def test_empty_file(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/ocr",
            files={"file": ("test.png", b"", "image/png")},
        )
        assert response.status_code == 400
        assert "Empty" in response.json()["detail"]

    async def test_file_too_large(self, client: AsyncClient) -> None:
        """Files exceeding max_file_size_mb are rejected."""
        big_data = b"\x89PNG\r\n\x1a\n" + b"x" * (11 * 1024 * 1024)  # >10MB
        with patch("src.main.magic.from_buffer", return_value="image/png"):
            response = await client.post(
                "/api/v1/ocr",
                files={"file": ("big.png", big_data, "image/png")},
            )
        assert response.status_code == 413

    async def test_unsupported_file_type(self, client: AsyncClient) -> None:
        with patch("src.main.magic.from_buffer", return_value="text/plain"):
            response = await client.post(
                "/api/v1/ocr",
                files={"file": ("test.txt", b"hello world", "text/plain")},
            )
        assert response.status_code == 400
        assert "Unsupported" in response.json()["detail"]

    async def test_invalid_dpi(self, client: AsyncClient) -> None:
        with patch("src.main.magic.from_buffer", return_value="image/png"):
            response = await client.post(
                "/api/v1/ocr",
                files={"file": ("test.png", b"\x89PNG\r\n\x1a\nfakedata", "image/png")},
                data={"dpi": "50"},  # Below minimum
            )
        assert response.status_code == 400
        assert "DPI" in response.json()["detail"]

    async def test_invalid_page_range(self, client: AsyncClient) -> None:
        with patch("src.main.magic.from_buffer", return_value="image/png"):
            response = await client.post(
                "/api/v1/ocr",
                files={"file": ("test.png", b"\x89PNG\r\n\x1a\nfakedata", "image/png")},
                data={"pages": "5-2"},
            )
        assert response.status_code == 400

    async def test_successful_image_ocr(self, client: AsyncClient) -> None:
        """Full success path for a single image."""
        from tests.conftest import SAMPLE_HOCR

        fake_image_bytes = b"\x89PNG\r\n\x1a\nfakedata"

        with (
            patch("src.main.magic.from_buffer", return_value="image/png"),
            patch("src.main.load_image_from_bytes") as mock_load,
            patch("src.main.recognize_image", new_callable=AsyncMock) as mock_recognize,
        ):
            mock_img = type("FakeImage", (), {"width": 2550, "height": 3300})()
            mock_load.return_value = mock_img
            mock_recognize.return_value = SAMPLE_HOCR

            response = await client.post(
                "/api/v1/ocr",
                files={"file": ("test.png", fake_image_bytes, "image/png")},
                data={"dpi": "300", "language": "eng"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["provider"] == "tesseract"
        assert len(data["pages"]) == 1
        assert data["pages"][0]["page_number"] == 0
        assert data["pages"][0]["width"] == 2550.0
        assert data["pages"][0]["height"] == 3300.0
        assert len(data["pages"][0]["blocks"]) == 2
        assert data["latency_ms"] >= 0
        assert data["raw"] is not None


class TestParsePageRange:
    """Tests for _parse_page_range()."""

    def test_all(self) -> None:
        from src.main import _parse_page_range

        assert _parse_page_range("all", 100) == (None, None)

    def test_single_page(self) -> None:
        from src.main import _parse_page_range

        assert _parse_page_range("3", 100) == (3, 3)

    def test_range(self) -> None:
        from src.main import _parse_page_range

        assert _parse_page_range("2-5", 100) == (2, 5)

    def test_invalid_range(self) -> None:
        from src.main import _parse_page_range

        with pytest.raises(ValueError, match="Invalid"):
            _parse_page_range("5-2", 100)

    def test_exceeds_max(self) -> None:
        from src.main import _parse_page_range

        with pytest.raises(ValueError, match="exceeds"):
            _parse_page_range("1-200", 100)

    def test_zero_page(self) -> None:
        from src.main import _parse_page_range

        with pytest.raises(ValueError, match=">= 1"):
            _parse_page_range("0", 100)
