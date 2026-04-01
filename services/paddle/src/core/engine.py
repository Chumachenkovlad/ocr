"""OCR engine wrapper with dual backend: PaddleOCR (prod) + RapidOCR (local dev)."""

from __future__ import annotations

import asyncio
import os
import platform
from typing import Any

# PaddlePaddle reads FLAGS at import time — set before any paddle import
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_call_stack_level", "2")
os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")

import structlog
from PIL import Image

from src.config import DeviceType, OCRBackend, OCRMode, Settings

logger = structlog.get_logger()


class OCREngineError(Exception):
    """Raised when OCR engine processing fails."""


def _resolve_backend(settings: Settings) -> OCRBackend:
    """Resolve AUTO backend to a concrete backend based on platform."""
    if settings.backend != OCRBackend.AUTO:
        return settings.backend

    # ARM64 = Apple Silicon / Graviton — PaddlePaddle won't work
    if platform.machine() in ("arm64", "aarch64"):
        return OCRBackend.RAPID

    # Try importing paddle; fall back to rapid if unavailable
    try:
        import paddle  # noqa: F401

        return OCRBackend.PADDLE
    except (ImportError, Exception):
        return OCRBackend.RAPID


class OCREngine:
    """Singleton wrapper around PaddleOCR / RapidOCR.

    Manages model loading, warm-up, and inference. Designed to be
    created once at application startup and injected via FastAPI deps.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._backend = _resolve_backend(settings)
        self._ocr_engine: Any = None
        self._structure_engine: Any = None
        self._models_loaded = False
        self._semaphore = asyncio.Semaphore(settings.max_concurrent)

    @property
    def models_loaded(self) -> bool:
        return self._models_loaded

    @property
    def backend(self) -> OCRBackend:
        return self._backend

    def load_models(self) -> None:
        """Load OCR models. Call during app startup."""
        if self._backend == OCRBackend.RAPID:
            self._load_rapid()
        else:
            self._load_paddle()

    def _load_rapid(self) -> None:
        """Load RapidOCR (ONNX Runtime) — works on ARM64 + x86."""
        try:
            from rapidocr_onnxruntime import RapidOCR

            self._ocr_engine = RapidOCR()
            self._models_loaded = True
            logger.info("rapid_ocr_loaded", backend="rapid")
        except ImportError as exc:
            logger.error("rapidocr_import_failed", error=str(exc))
            msg = "rapidocr-onnxruntime not installed. pip install rapidocr-onnxruntime"
            raise OCREngineError(msg) from exc
        except Exception as exc:
            logger.error("rapid_model_load_failed", error=str(exc))
            msg = f"Failed to load RapidOCR: {exc}"
            raise OCREngineError(msg) from exc

    def _load_paddle(self) -> None:
        """Load PaddleOCR — requires x86_64 with AVX."""
        try:
            import paddle
            paddle.set_flags({
                "FLAGS_enable_pir_in_executor": False,
                "FLAGS_enable_pir_api": False,
                "FLAGS_use_mkldnn": False,
            })

            from paddleocr import PaddleOCR  # type: ignore[import-untyped,unused-ignore]

            device = "gpu:0" if self._settings.device == DeviceType.GPU else "cpu"

            self._ocr_engine = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device=device,
            )

            logger.info("ocr_models_loaded", device=device, backend="paddle")

            if self._settings.mode == OCRMode.STRUCTURE:
                try:
                    from paddleocr import (
                        PPStructureV3,  # type: ignore[import-untyped,unused-ignore]
                    )

                    self._structure_engine = PPStructureV3(device=device)
                    logger.info("structure_models_loaded", device=device)
                except ImportError:
                    logger.warning("ppstructurev3_not_available", fallback="ocr-only")
            else:
                logger.info("structure_models_skipped", reason="mode is ocr")

            self._models_loaded = True

        except ImportError as exc:
            logger.error("paddleocr_import_failed", error=str(exc))
            msg = "PaddleOCR is not installed. Install with: pip install paddleocr[doc-parser]"
            raise OCREngineError(msg) from exc
        except Exception as exc:
            logger.error("model_load_failed", error=str(exc))
            msg = f"Failed to load OCR models: {exc}"
            raise OCREngineError(msg) from exc

    async def process_image(
        self,
        image: Image.Image,
        mode: OCRMode = OCRMode.OCR,
        language: str = "auto",
    ) -> dict[str, Any]:
        """Run OCR on a single page image.

        Args:
            image: PIL Image to process (RGB mode).
            mode: Processing mode (ocr-only or full structure analysis).
            language: Language hint for recognition.

        Returns:
            Raw OCR result dictionary with unified format.
        """
        if not self._models_loaded:
            msg = "Models not loaded. Call load_models() first."
            raise OCREngineError(msg)

        async with self._semaphore:
            return await asyncio.to_thread(self._run_inference, image, mode)

    def _run_inference(self, image: Image.Image, mode: OCRMode) -> dict[str, Any]:
        """Synchronous inference — runs in a thread pool."""
        import numpy as np

        img_array = np.array(image)

        try:
            if self._backend == OCRBackend.RAPID:
                return self._infer_rapid(img_array)
            elif mode == OCRMode.STRUCTURE and self._structure_engine is not None:
                result = self._structure_engine.predict(img_array)
                return self._wrap_structure_result(result)
            elif self._ocr_engine is not None:
                result = self._ocr_engine.predict(img_array)
                return self._wrap_ocr_result(result)
            else:
                msg = "No OCR engine available"
                raise OCREngineError(msg)
        except OCREngineError:
            raise
        except Exception as exc:
            logger.error("ocr_inference_failed", error=str(exc))
            msg = f"OCR inference failed: {exc}"
            raise OCREngineError(msg) from exc

    def _extract_rotation_angle(self, result: Any) -> float:
        """Extract detected rotation angle from PaddleOCR result."""
        results = result if isinstance(result, list) else [result]
        for r in results:
            # PaddleOCR 3.x stores orientation in doc_preprocessor_res
            preproc = None
            if isinstance(r, dict):
                preproc = r.get("doc_preprocessor_res")
            elif hasattr(r, "doc_preprocessor_res"):
                preproc = getattr(r, "doc_preprocessor_res", None)

            if preproc is not None:
                angle = (
                    preproc.get("angle", -1)
                    if isinstance(preproc, dict)
                    else getattr(preproc, "angle", -1)
                )
                if angle >= 0:
                    return float(angle)
        return 0.0

    def _wrap_ocr_result(self, result: Any) -> dict[str, Any]:
        """Wrap PaddleOCR predict() output with extracted metadata."""
        self._debug_log_result("ocr", result)
        angle = self._extract_rotation_angle(result)
        return {"mode": "ocr", "raw": result, "rotation_angle": angle}

    def _wrap_structure_result(self, result: Any) -> dict[str, Any]:
        """Wrap PPStructureV3 predict() output with extracted metadata and markdown."""
        self._debug_log_result("structure", result)
        angle = self._extract_rotation_angle(result)
        markdown = self._extract_markdown(result)
        return {
            "mode": "structure",
            "raw": result,
            "rotation_angle": angle,
            "markdown": markdown,
        }

    def _extract_markdown(self, result: Any) -> str | None:
        """Extract markdown representation from PPStructureV3 result."""
        results = result if isinstance(result, list) else [result]
        for r in results:
            md = None
            if isinstance(r, dict):
                md = r.get("markdown")
            elif hasattr(r, "markdown"):
                md_val = getattr(r, "markdown", None)
                # PPStructureV3 returns markdown as a dict with 'markdown_text' key
                if isinstance(md_val, dict):
                    md = md_val.get("markdown_text")
                elif isinstance(md_val, str):
                    md = md_val
            if md:
                return md
        return None

    def _debug_log_result(self, label: str, result: Any) -> None:
        """Log the structure of PaddleOCR result for debugging."""
        items = result if isinstance(result, list) else [result]
        for idx, item in enumerate(items):
            item_type = type(item).__name__
            if isinstance(item, dict):
                keys = list(item.keys())
                logger.info(
                    "raw_result_debug",
                    label=label,
                    idx=idx,
                    type=item_type,
                    keys=keys,
                )
                # Deep-inspect critical sub-objects
                self._log_subobj(label, "overall_ocr_res", item.get("overall_ocr_res"))
                self._log_subobj(label, "layout_det_res", item.get("layout_det_res"))
                # Log first item of *_res_list
                for list_key in ("table_res_list", "formula_res_list",
                                 "seal_res_list", "parsing_res_list"):
                    lst = item.get(list_key, [])
                    if isinstance(lst, list) and lst:
                        self._log_subobj(label, f"{list_key}[0]", lst[0])
            else:
                attrs = [a for a in dir(item) if not a.startswith("_")]
                logger.info(
                    "raw_result_debug",
                    label=label,
                    idx=idx,
                    type=item_type,
                    attrs=attrs[:30],
                )

    def _log_subobj(self, label: str, name: str, obj: Any) -> None:
        """Log structure of a PaddleX sub-result object."""
        if obj is None:
            return
        obj_type = type(obj).__name__
        if isinstance(obj, dict):
            logger.info("subobj_debug", label=label, name=name, type=obj_type,
                        keys=list(obj.keys())[:20])
        elif isinstance(obj, (list, tuple)):
            logger.info("subobj_debug", label=label, name=name, type=obj_type,
                        length=len(obj))
        else:
            attrs = [a for a in dir(obj) if not a.startswith("_")][:25]
            logger.info("subobj_debug", label=label, name=name, type=obj_type,
                        attrs=attrs)
            # Log key attribute types/shapes
            for attr_name in ("dt_polys", "rec_texts", "rec_scores", "boxes",
                              "labels", "label_names", "scores", "bbox", "box",
                              "html", "table_html", "latex", "formula",
                              "rec_formula", "label", "type",
                              "input_path", "layout_bbox", "sub_regions"):
                val = getattr(obj, attr_name, None)
                if val is None:
                    continue
                val_type = type(val).__name__
                if isinstance(val, str):
                    preview = val[:150]
                elif isinstance(val, (int, float, bool)):
                    preview = str(val)
                elif isinstance(val, list):
                    preview = f"list[{len(val)}]"
                    if val:
                        preview += f" item0_type={type(val[0]).__name__}"
                elif hasattr(val, "shape"):
                    preview = f"ndarray shape={val.shape} dtype={val.dtype}"
                elif isinstance(val, dict):
                    preview = f"dict keys={list(val.keys())[:8]}"
                else:
                    preview = str(val)[:100]
                logger.info("subobj_attr", label=label, name=name,
                            attr=attr_name, val_type=val_type, preview=preview)

    def _infer_rapid(self, img_array: Any) -> dict[str, Any]:
        """Run RapidOCR and convert results to PaddleOCR-compatible format."""
        result, _ = self._ocr_engine(img_array)

        if not result:
            return {"mode": "ocr", "raw": []}

        # RapidOCR returns: [[box, text, score], ...]
        # Convert to PaddleOCR format: {dt_polys, rec_texts, rec_scores}
        polys = []
        texts = []
        scores = []

        for item in result:
            box, text, score = item
            polys.append(box)
            texts.append(text)
            scores.append(float(score))

        return {
            "mode": "ocr",
            "raw": [{"dt_polys": polys, "rec_texts": texts, "rec_scores": scores}],
        }
