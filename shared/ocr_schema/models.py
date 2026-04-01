"""Canonical OCR output models.

Derived from DR-001 (docs/DR-001-ocr-adapter.md).
All services normalize their provider-specific output into these types.
Coordinates are always in pixels (providers using normalized 0-1 coords
must denormalize inside their adapter).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in pixel coordinates, top-left origin."""

    x1: float = Field(description="Left edge")
    y1: float = Field(description="Top edge")
    x2: float = Field(description="Right edge")
    y2: float = Field(description="Bottom edge")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


class OcrWord(BaseModel):
    """A single recognized word."""

    text: str
    confidence: float = Field(
        ge=-1.0,
        le=1.0,
        description="0.0-1.0 recognition confidence; -1.0 if unavailable",
    )
    bbox: BoundingBox


class OcrLine(BaseModel):
    """A text line containing one or more words."""

    text: str
    words: list[OcrWord] = Field(default_factory=list)
    bbox: BoundingBox


class OcrBlock(BaseModel):
    """A layout block (paragraph, table, figure, etc.)."""

    block_type: str = Field(
        default="text",
        description="text | table | figure | header | footer | title",
    )
    lines: list[OcrLine] = Field(default_factory=list)
    bbox: BoundingBox


class OcrPage(BaseModel):
    """OCR results for a single page."""

    page_number: int = Field(ge=0)
    width: float = Field(gt=0, description="Page width in pixels")
    height: float = Field(gt=0, description="Page height in pixels")
    blocks: list[OcrBlock] = Field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n".join(
            line.text for block in self.blocks for line in block.lines
        )


class OcrResult(BaseModel):
    """Top-level OCR response returned by every service."""

    provider: str = Field(description="surya | doctr | paddle | tesseract")
    pages: list[OcrPage] = Field(default_factory=list)
    raw: dict | None = Field(
        default=None,
        description="Original provider output preserved for debugging",
    )
    latency_ms: float = Field(default=0.0, ge=0.0)
