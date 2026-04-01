"""Convert PDF bytes to a list of PIL images for OCR processing."""

from __future__ import annotations

from PIL import Image
from pdf2image import convert_from_bytes


def pdf_to_images(pdf_bytes: bytes, dpi: int = 300) -> list[Image.Image]:
    """Render every page of *pdf_bytes* as an RGB PIL image.

    Args:
        pdf_bytes: Raw PDF file content.
        dpi: Resolution for rasterisation (higher = better quality but slower).

    Returns:
        One ``PIL.Image.Image`` per page.
    """
    return convert_from_bytes(pdf_bytes, dpi=dpi)
