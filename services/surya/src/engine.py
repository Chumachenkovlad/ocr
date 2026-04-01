"""Surya model loading and inference.

Wraps Surya's detection and recognition pipelines behind a simple
interface so the rest of the service stays decoupled from Surya
internals.  Import paths are version-sensitive -- the try/except
blocks make it easier to adapt when the upstream API changes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)


class SuryaEngine:
    """Thin wrapper around Surya's detection + recognition models.

    Supports surya-ocr 0.6.x–0.13.x which uses ``run_ocr()`` with
    separately loaded detection and recognition models/processors.
    """

    def __init__(self, device: str = "cpu") -> None:
        from surya.detection import DetectionPredictor
        from surya.recognition import RecognitionPredictor

        logger.info("Loading Surya models on device=%s", device)
        self._det = DetectionPredictor(device=device)
        self._rec = RecognitionPredictor(device=device)
        logger.info("Surya models loaded successfully")

    def ocr(
        self,
        images: list[Image.Image],
        languages: list[list[str]] | None = None,
    ) -> list:
        """Run detection + recognition on a list of PIL images.

        Returns:
            List of Surya OCRResult objects (one per image), each containing
            ``text_lines`` with ``.text``, ``.confidence``, and ``.bbox``.
        """
        if languages is None:
            languages = [["en"]] * len(images)

        # RecognitionPredictor.__call__ takes det_predictor (callable),
        # not detection results — it runs detection internally.
        return self._rec(images, languages, det_predictor=self._det)
