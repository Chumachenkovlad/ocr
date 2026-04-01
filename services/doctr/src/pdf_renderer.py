"""PDF to image conversion using pdf2image (poppler)."""

from PIL import Image
from pdf2image import convert_from_bytes


def pdf_to_images(pdf_bytes: bytes, dpi: int = 300) -> list[Image.Image]:
    """Convert a PDF byte stream into a list of PIL images."""
    return convert_from_bytes(pdf_bytes, dpi=dpi)
