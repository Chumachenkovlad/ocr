"""Tesseract binary invocation via subprocess."""

from __future__ import annotations

import asyncio
import subprocess
import tempfile
from pathlib import Path

import structlog
from PIL import Image

from src.config import Settings

logger = structlog.get_logger()


class TesseractError(Exception):
    """Raised when Tesseract execution fails."""


class TesseractNotFoundError(TesseractError):
    """Raised when the Tesseract binary is not available."""


def _check_tesseract_available(binary: str) -> bool:
    """Check whether the Tesseract binary is reachable."""
    try:
        result = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def run_tesseract(
    image_path: Path,
    *,
    binary: str = "tesseract",
    lang: str = "eng",
    dpi: int = 300,
    timeout: int = 30,
) -> str:
    """Run Tesseract on a single image file and return raw hOCR output.

    Args:
        image_path: Path to the input image file.
        binary: Tesseract binary name or path.
        lang: Tesseract language code (e.g. ``eng``, ``deu``).
        dpi: DPI hint passed to Tesseract.
        timeout: Maximum execution time in seconds.

    Returns:
        Raw hOCR HTML string.

    Raises:
        TesseractNotFoundError: If the binary is missing.
        TesseractError: If Tesseract returns a non-zero exit code.
    """
    args = [
        binary,
        str(image_path),
        "stdout",
        "-l", lang,
        "--oem", "3",
        "--psm", "3",
        "--dpi", str(dpi),
        "hocr",
    ]

    logger.info(
        "tesseract_exec",
        image=str(image_path),
        lang=lang,
        dpi=dpi,
        timeout=timeout,
    )

    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        msg = f"Tesseract binary not found: {binary}. Ensure Tesseract is installed."
        raise TesseractNotFoundError(msg) from exc
    except subprocess.TimeoutExpired as exc:
        msg = f"Tesseract timed out after {timeout}s"
        raise TesseractError(msg) from exc

    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "not found" in stderr or "ENOENT" in stderr:
            msg = f"Tesseract binary not found: {binary}. Ensure Tesseract is installed."
            raise TesseractNotFoundError(msg)
        msg = f"Tesseract OCR failed (rc={result.returncode}): {stderr}"
        raise TesseractError(msg)

    return result.stdout


async def recognize_image(
    image: Image.Image,
    settings: Settings,
    *,
    lang: str | None = None,
    dpi: int | None = None,
) -> str:
    """Run Tesseract on a PIL Image and return raw hOCR output.

    Writes the image to a temporary file, invokes Tesseract in a thread pool,
    and cleans up afterwards.

    Args:
        image: PIL Image (will be saved as PNG).
        settings: Application settings.
        lang: Override language (defaults to ``settings.tesseract_lang``).
        dpi: Override DPI (defaults to ``settings.tesseract_dpi``).

    Returns:
        Raw hOCR HTML string.
    """
    resolved_lang = lang or settings.tesseract_lang
    resolved_dpi = dpi or settings.tesseract_dpi

    # Write image to a temp file — Tesseract reads from disk
    suffix = ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # Save in a thread to avoid blocking the event loop
        await asyncio.to_thread(image.save, str(tmp_path), "PNG")

        hocr = await asyncio.to_thread(
            run_tesseract,
            tmp_path,
            binary=settings.tesseract_binary,
            lang=resolved_lang,
            dpi=resolved_dpi,
            timeout=settings.tesseract_timeout,
        )
        return hocr
    finally:
        tmp_path.unlink(missing_ok=True)
