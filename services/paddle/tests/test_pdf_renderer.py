"""Tests for PDF renderer module."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from PIL import Image
from src.core.pdf_renderer import PDFRenderError, load_image_from_bytes, render_pdf_pages


class TestRenderPdfPages:
    def test_basic_render(self) -> None:
        mock_img = Image.new("RGB", (100, 100))
        with patch(
            "src.core.pdf_renderer.convert_from_bytes", return_value=[mock_img],
        ) as mock_convert:
            result = render_pdf_pages(b"fake-pdf", dpi=150)
        assert len(result) == 1
        mock_convert.assert_called_once()
        call_kwargs = mock_convert.call_args
        assert call_kwargs[1]["dpi"] == 150

    def test_page_range_passed(self) -> None:
        with patch("src.core.pdf_renderer.convert_from_bytes", return_value=[]) as mock_convert:
            render_pdf_pages(b"fake", dpi=300, first_page=2, last_page=5)
        kwargs = mock_convert.call_args[1]
        assert kwargs["first_page"] == 2
        assert kwargs["last_page"] == 5

    def test_no_page_range(self) -> None:
        with patch("src.core.pdf_renderer.convert_from_bytes", return_value=[]) as mock_convert:
            render_pdf_pages(b"fake", dpi=300)
        kwargs = mock_convert.call_args[1]
        assert "first_page" not in kwargs
        assert "last_page" not in kwargs

    def test_render_failure_raises_pdf_render_error(self) -> None:
        with (
            patch(
                "src.core.pdf_renderer.convert_from_bytes",
                side_effect=RuntimeError("poppler error"),
            ),
            pytest.raises(PDFRenderError, match="Failed to render PDF"),
        ):
            render_pdf_pages(b"bad-pdf")


class TestLoadImageFromBytes:
    def test_load_rgb_image(self) -> None:
        img = Image.new("RGB", (50, 50), color="red")
        import io

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result = load_image_from_bytes(buf.getvalue())
        assert result.mode == "RGB"
        assert result.size == (50, 50)

    def test_load_grayscale_converts_to_rgb(self) -> None:
        img = Image.new("L", (50, 50))
        import io

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result = load_image_from_bytes(buf.getvalue())
        assert result.mode == "RGB"

    def test_invalid_bytes_raises_error(self) -> None:
        with pytest.raises(PDFRenderError, match="Failed to load image"):
            load_image_from_bytes(b"not-an-image")
