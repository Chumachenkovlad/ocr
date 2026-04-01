"""Convert Surya OCR output to the canonical ``OcrResult`` schema.

Surya returns ``OCRResult`` objects whose ``text_lines`` are ``TextLine``
instances with ``.text``, ``.confidence``, and ``.bbox`` (``[x1, y1, x2, y2]``
in pixel coordinates).  No denormalisation is required.

Because Surya provides only line-level granularity we:
1. Split each line's text into word-level bounding boxes (see ``word_splitter``).
2. Group lines into blocks by vertical (Y) proximity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ocr_schema.models import (
    BoundingBox,
    OcrBlock,
    OcrLine,
    OcrPage,
    OcrResult,
)
from src.word_splitter import split_line_to_words

if TYPE_CHECKING:
    from PIL import Image


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def adapt_surya_result(
    ocr_results: list,
    images: list[Image.Image],
    latency_ms: float,
) -> OcrResult:
    """Map a list of Surya ``OCRResult`` objects to the shared ``OcrResult``.

    Args:
        ocr_results: One ``OCRResult`` per page, returned by
            ``SuryaEngine.ocr``.
        images: Corresponding PIL images (used to read page dimensions).
        latency_ms: Total inference time in milliseconds.
    """
    pages: list[OcrPage] = []
    for page_idx, (result, image) in enumerate(zip(ocr_results, images)):
        w, h = image.size

        lines: list[OcrLine] = []
        for text_line in result.text_lines:
            bbox = BoundingBox(
                x1=text_line.bbox[0],
                y1=text_line.bbox[1],
                x2=text_line.bbox[2],
                y2=text_line.bbox[3],
            )
            words = split_line_to_words(
                text_line.text, bbox, text_line.confidence
            )
            lines.append(
                OcrLine(text=text_line.text, words=words, bbox=bbox)
            )

        blocks = group_lines_into_blocks(lines)

        pages.append(
            OcrPage(
                page_number=page_idx,
                width=w,
                height=h,
                blocks=blocks,
            )
        )

    return OcrResult(provider="surya", pages=pages, latency_ms=latency_ms)


# ---------------------------------------------------------------------------
# Block grouping helpers
# ---------------------------------------------------------------------------

def group_lines_into_blocks(
    lines: list[OcrLine],
    gap_threshold: float = 20.0,
) -> list[OcrBlock]:
    """Group lines into blocks by vertical (Y) proximity.

    Lines whose vertical gap exceeds *gap_threshold* pixels start a new
    block.
    """
    if not lines:
        return []

    sorted_lines = sorted(lines, key=lambda ln: (ln.bbox.y1, ln.bbox.x1))
    blocks: list[OcrBlock] = []
    current_block_lines: list[OcrLine] = [sorted_lines[0]]

    for line in sorted_lines[1:]:
        prev = current_block_lines[-1]
        if line.bbox.y1 - prev.bbox.y2 > gap_threshold:
            blocks.append(_make_block(current_block_lines))
            current_block_lines = [line]
        else:
            current_block_lines.append(line)

    blocks.append(_make_block(current_block_lines))
    return blocks


def _make_block(lines: list[OcrLine]) -> OcrBlock:
    """Create an ``OcrBlock`` that encloses all *lines*."""
    x1 = min(ln.bbox.x1 for ln in lines)
    y1 = min(ln.bbox.y1 for ln in lines)
    x2 = max(ln.bbox.x2 for ln in lines)
    y2 = max(ln.bbox.y2 for ln in lines)
    return OcrBlock(
        block_type="text",
        lines=lines,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )
