"""Pydantic v2 request/response models for the OCR API.

Internal schemas (BBoxSchema, WordSchema, etc.) are used by the PaddleOCR adapter.
The API boundary converts these to the shared ``ocr_schema`` models via
``src.api.router.to_shared_result``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# Re-export shared models so callers can import from one place when needed.
from ocr_schema.models import (  # noqa: F401
    BoundingBox as SharedBBox,
    OcrBlock as SharedBlock,
    OcrLine as SharedLine,
    OcrPage as SharedPage,
    OcrResult,
    OcrWord as SharedWord,
)


class BBoxSchema(BaseModel):
    """Bounding box in pixel coordinates."""

    x: int = Field(description="Left edge x-coordinate")
    y: int = Field(description="Top edge y-coordinate")
    w: int = Field(description="Width in pixels")
    h: int = Field(description="Height in pixels")


class WordSchema(BaseModel):
    """A single recognized word with position and confidence."""

    text: str = Field(description="Recognized text content")
    confidence: float = Field(ge=0.0, le=1.0, description="Recognition confidence")
    bbox: BBoxSchema = Field(description="Word bounding box")


class LineSchema(BaseModel):
    """A text line containing one or more words."""

    line_index: int = Field(ge=0, description="Line index within its block")
    bbox: BBoxSchema = Field(description="Line bounding box")
    words: list[WordSchema] = Field(default_factory=list, description="Words in reading order")


class BlockSchema(BaseModel):
    """A layout block (text paragraph, title, table, figure, etc.)."""

    block_index: int = Field(ge=0, description="Block index within the page")
    block_type: str = Field(
        default="text",
        description="Layout type: text, title, table, figure, caption,"
        " header, footer, seal, formula, chart, etc.",
    )
    bbox: BBoxSchema = Field(description="Block bounding box")
    lines: list[LineSchema] = Field(default_factory=list, description="Lines within the block")
    font_size_estimate_pt: float = Field(
        default=0.0, ge=0.0, description="Estimated font size in points"
    )
    table_html: str | None = Field(
        default=None, description="HTML table markup (for table blocks)"
    )
    formula_latex: str | None = Field(
        default=None, description="LaTeX formula string (for formula blocks)"
    )
    seal_texts: list[str] | None = Field(
        default=None, description="Recognized seal/stamp texts (for seal blocks)"
    )


class PageSchema(BaseModel):
    """OCR results for a single page."""

    page_index: int = Field(ge=0, description="Zero-based page index")
    width_px: int = Field(gt=0, description="Image width used for OCR")
    height_px: int = Field(gt=0, description="Image height used for OCR")
    dpi: int = Field(gt=0, description="Rendering DPI")
    language: str = Field(default="en", description="Detected or requested language")
    rotation_angle: float = Field(default=0.0, description="Detected rotation correction applied")
    blocks: list[BlockSchema] = Field(default_factory=list, description="Layout blocks")
    reading_order: list[int] = Field(
        default_factory=list, description="Ordered block indices for reading order"
    )
    markdown: str | None = Field(
        default=None, description="Full page content as Markdown (structure mode)"
    )
    page_image_base64: str | None = Field(
        default=None,
        description="Base64-encoded PNG of the page image used for OCR (for overlay alignment)",
    )


class OCRResponse(BaseModel):
    """Top-level OCR API response."""

    version: str = Field(default="1.0", description="Response schema version")
    processing_time_ms: int = Field(ge=0, description="Total processing time in milliseconds")
    pages: list[PageSchema] = Field(default_factory=list, description="Per-page OCR results")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="ok")
    models_loaded: bool = Field(default=False)
    version: str = Field(default="0.2.0")
    backend: str = Field(default="unknown", description="OCR backend in use")


class ErrorDetail(BaseModel):
    """Structured error response."""

    error: str = Field(description="Error type identifier")
    message: str = Field(description="Human-readable error description")
    detail: str | None = Field(default=None, description="Additional detail")
