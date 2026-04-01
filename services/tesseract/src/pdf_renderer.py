"""PDF to image conversion using pdf2image (poppler) and image loading."""

from __future__ import annotations

import io
from typing import Any

import structlog
from pdf2image import convert_from_bytes
from PIL import Image, ImageOps

logger = structlog.get_logger()


class PDFRenderError(Exception):
    """Raised when PDF rendering fails."""


def render_pdf_pages(
    pdf_bytes: bytes,
    dpi: int = 300,
    first_page: int | None = None,
    last_page: int | None = None,
) -> list[Image.Image]:
    """Render PDF pages to PIL Images at the specified DPI.

    Args:
        pdf_bytes: Raw PDF file bytes.
        dpi: Rendering resolution (72-1200).
        first_page: First page to render (1-based, inclusive). None = first.
        last_page: Last page to render (1-based, inclusive). None = last.

    Returns:
        List of PIL Images in RGB mode, one per rendered page.

    Raises:
        PDFRenderError: If pdf2image/poppler fails.
    """
    kwargs: dict[str, Any] = {"fmt": "RGB", "dpi": dpi}
    if first_page is not None:
        kwargs["first_page"] = first_page
    if last_page is not None:
        kwargs["last_page"] = last_page

    try:
        images: list[Image.Image] = convert_from_bytes(pdf_bytes, **kwargs)
    except Exception as exc:
        logger.error("pdf_render_failed", error=str(exc))
        msg = f"Failed to render PDF: {exc}"
        raise PDFRenderError(msg) from exc

    logger.info("pdf_rendered", page_count=len(images), dpi=dpi)
    return images


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Load a single image from raw bytes (PNG, JPEG, TIFF).

    Applies EXIF orientation and converts to RGB.

    Args:
        image_bytes: Raw image file bytes.

    Returns:
        PIL Image in RGB mode.

    Raises:
        PDFRenderError: If the image cannot be loaded.
    """
    try:
        img: Image.Image = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as exc:
        msg = f"Failed to load image: {exc}"
        raise PDFRenderError(msg) from exc
