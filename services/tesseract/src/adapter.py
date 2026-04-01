"""Adapter: converts parsed hOCR output into the shared OcrResult schema."""

from __future__ import annotations

from ocr_schema import BoundingBox, OcrBlock, OcrLine, OcrPage, OcrResult, OcrWord

from src.hocr_parser import HocrBlock, HocrLine, HocrPage, HocrWord


def _bbox_from_hocr(word_or_block: HocrWord | HocrLine | HocrBlock) -> BoundingBox:
    """Convert an hOCR BBox to the shared BoundingBox (x1/y1/x2/y2 pixels)."""
    b = word_or_block.bbox
    return BoundingBox(x1=float(b.x1), y1=float(b.y1), x2=float(b.x2), y2=float(b.y2))


def _adapt_word(word: HocrWord) -> OcrWord:
    return OcrWord(
        text=word.text,
        confidence=round(word.confidence, 4),
        bbox=_bbox_from_hocr(word),
    )


def _adapt_line(line: HocrLine) -> OcrLine:
    words = [_adapt_word(w) for w in line.words]
    text = " ".join(w.text for w in line.words)
    return OcrLine(
        text=text,
        words=words,
        bbox=_bbox_from_hocr(line),
    )


def _adapt_block(block: HocrBlock) -> OcrBlock:
    lines = [_adapt_line(ln) for ln in block.lines]
    return OcrBlock(
        block_type="text",
        lines=lines,
        bbox=_bbox_from_hocr(block),
    )


def adapt_page(hocr_page: HocrPage, page_number: int) -> OcrPage:
    """Convert a parsed HocrPage into the shared OcrPage schema.

    Args:
        hocr_page: Parsed hOCR page from ``hocr_parser.parse_hocr``.
        page_number: Zero-based page index.

    Returns:
        An ``OcrPage`` conforming to the shared ocr_schema.
    """
    blocks = [_adapt_block(b) for b in hocr_page.blocks]

    # Guard against zero dimensions (empty pages)
    width = float(max(hocr_page.width, 1))
    height = float(max(hocr_page.height, 1))

    return OcrPage(
        page_number=page_number,
        width=width,
        height=height,
        blocks=blocks,
    )


def build_ocr_result(
    pages: list[OcrPage],
    *,
    raw_hocr: list[str] | None = None,
    latency_ms: float = 0.0,
) -> OcrResult:
    """Build the top-level OcrResult envelope.

    Args:
        pages: Adapted OcrPage objects.
        raw_hocr: Optional list of raw hOCR strings (one per page) for debugging.
        latency_ms: Total processing latency in milliseconds.

    Returns:
        An ``OcrResult`` conforming to the shared ocr_schema.
    """
    raw: dict | None = None
    if raw_hocr is not None:
        raw = {"hocr": raw_hocr}

    return OcrResult(
        provider="tesseract",
        pages=pages,
        raw=raw,
        latency_ms=latency_ms,
    )
