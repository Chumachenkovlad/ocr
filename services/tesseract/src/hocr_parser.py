"""hOCR HTML parser — faithful port of the TypeScript hocr-parser.service.ts.

Parses Tesseract hOCR output into a structured hierarchy:
  page -> blocks (ocr_carea / ocr_par) -> lines -> words

Uses lxml.html for parsing and XPath for element selection by CSS class.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

import structlog
from lxml import html

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Data classes (immutable intermediate representation)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BBox:
    """Pixel-coordinate bounding box: (x1, y1) top-left, (x2, y2) bottom-right."""

    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclass(frozen=True)
class TitleProps:
    """Parsed hOCR title attribute properties."""

    bbox: BBox | None = None
    baseline: tuple[float, float] | None = None
    x_wconf: int | None = None


@dataclass(frozen=True)
class HocrWord:
    """A single recognized word."""

    text: str
    bbox: BBox
    confidence: float
    font_size_estimate: float


@dataclass(frozen=True)
class HocrLine:
    """A text line containing one or more words."""

    bbox: BBox
    baseline: int
    words: tuple[HocrWord, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class HocrBlock:
    """A layout block (paragraph or content area)."""

    bbox: BBox
    lines: tuple[HocrLine, ...] = field(default_factory=tuple)
    block_index: int = 0


@dataclass(frozen=True)
class HocrPage:
    """Parsed OCR results for a single page."""

    width: int = 0
    height: int = 0
    dpi: int = 300
    blocks: tuple[HocrBlock, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Title attribute parser
# ---------------------------------------------------------------------------

_BBOX_RE = re.compile(r"bbox\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)")
_WCONF_RE = re.compile(r"x_wconf\s+(-?\d+)")
_BASELINE_RE = re.compile(r"baseline\s+([-\d.]+)\s+([-\d.]+)")


def parse_title(title: str | None) -> TitleProps:
    """Parse an hOCR title attribute.

    Format: ``"bbox 100 200 300 250; x_wconf 95; baseline 0 -5"``
    """
    if not title:
        return TitleProps()

    bbox: BBox | None = None
    x_wconf: int | None = None
    baseline: tuple[float, float] | None = None

    bbox_m = _BBOX_RE.search(title)
    if bbox_m:
        bbox = BBox(
            x1=int(bbox_m.group(1)),
            y1=int(bbox_m.group(2)),
            x2=int(bbox_m.group(3)),
            y2=int(bbox_m.group(4)),
        )

    wconf_m = _WCONF_RE.search(title)
    if wconf_m:
        x_wconf = int(wconf_m.group(1))

    base_m = _BASELINE_RE.search(title)
    if base_m:
        baseline = (float(base_m.group(1)), float(base_m.group(2)))

    return TitleProps(bbox=bbox, x_wconf=x_wconf, baseline=baseline)


# ---------------------------------------------------------------------------
# Element finders (XPath-based)
# ---------------------------------------------------------------------------

def _find_all_by_class(root: html.HtmlElement, class_name: str) -> list[html.HtmlElement]:
    """Find all descendant elements whose class attribute contains *class_name*."""
    return root.xpath(f".//*[contains(@class, '{class_name}')]")


def _find_by_class(root: html.HtmlElement, class_name: str) -> html.HtmlElement | None:
    """Find the first descendant element with the given class."""
    results = _find_all_by_class(root, class_name)
    return results[0] if results else None


def _extract_text(element: html.HtmlElement) -> str:
    """Extract the concatenated text content of an element (non-recursive, immediate)."""
    # lxml's text_content() recurses into children which is what we want for
    # leaf words, but for words we only need the direct text.
    return (element.text_content() or "").strip()


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _parse_words(line_el: html.HtmlElement, dpi: int) -> tuple[HocrWord, ...]:
    """Parse all ocrx_word elements inside a line."""
    word_elements = _find_all_by_class(line_el, "ocrx_word")
    words: list[HocrWord] = []

    for word_el in word_elements:
        title_props = parse_title(word_el.get("title"))
        bbox = title_props.bbox or BBox()
        confidence = title_props.x_wconf / 100.0 if title_props.x_wconf is not None else 0.0
        font_size_estimate = (bbox.height * 72) / dpi if bbox.height > 0 else 0.0
        text = _extract_text(word_el)

        words.append(
            HocrWord(
                text=text,
                bbox=bbox,
                confidence=confidence,
                font_size_estimate=font_size_estimate,
            )
        )

    return tuple(words)


def _parse_lines(parent: html.HtmlElement, dpi: int) -> tuple[HocrLine, ...]:
    """Parse all line elements (ocr_line, ocr_textfloat, ocr_header) inside a parent."""
    line_elements = _find_all_by_class(parent, "ocr_line")
    text_floats = _find_all_by_class(parent, "ocr_textfloat")
    headers = _find_all_by_class(parent, "ocr_header")
    all_lines = [*line_elements, *text_floats, *headers]

    lines: list[HocrLine] = []

    for line_el in all_lines:
        title_props = parse_title(line_el.get("title"))
        line_bbox = title_props.bbox or BBox()

        # Baseline: same logic as the TS version
        if title_props.baseline is not None:
            baseline = line_bbox.y1 + line_bbox.height + int(title_props.baseline[1])
        else:
            baseline = line_bbox.y1 + line_bbox.height

        words = _parse_words(line_el, dpi)
        lines.append(HocrLine(bbox=line_bbox, baseline=baseline, words=words))

    return tuple(lines)


def _parse_blocks(page_div: html.HtmlElement, dpi: int) -> tuple[HocrBlock, ...]:
    """Parse blocks: ocr_carea -> ocr_par hierarchy (mirrors TS parseBlocks)."""
    blocks: list[HocrBlock] = []
    block_index = 0

    careas = _find_all_by_class(page_div, "ocr_carea")

    for carea in careas:
        carea_title = parse_title(carea.get("title"))
        pars = _find_all_by_class(carea, "ocr_par")

        if not pars:
            # Lines directly under carea (no paragraph wrapper)
            lines = _parse_lines(carea, dpi)
            if lines:
                blocks.append(
                    HocrBlock(
                        bbox=carea_title.bbox or BBox(),
                        lines=lines,
                        block_index=block_index,
                    )
                )
                block_index += 1
            continue

        for par in pars:
            par_title = parse_title(par.get("title"))
            lines = _parse_lines(par, dpi)
            if lines:
                blocks.append(
                    HocrBlock(
                        bbox=par_title.bbox or BBox(),
                        lines=lines,
                        block_index=block_index,
                    )
                )
                block_index += 1

    # Fallback: paragraphs directly under page (no carea wrapper) — mirrors TS
    if not careas:
        pars = _find_all_by_class(page_div, "ocr_par")
        for par in pars:
            par_title = parse_title(par.get("title"))
            lines = _parse_lines(par, dpi)
            if lines:
                blocks.append(
                    HocrBlock(
                        bbox=par_title.bbox or BBox(),
                        lines=lines,
                        block_index=block_index,
                    )
                )
                block_index += 1

    return tuple(blocks)


def parse_hocr(hocr_html: str, dpi: int = 300) -> HocrPage:
    """Parse Tesseract hOCR HTML output into the internal HocrPage structure.

    Args:
        hocr_html: Raw hOCR HTML string produced by Tesseract.
        dpi: DPI used during recognition (needed for font size estimation).

    Returns:
        An immutable ``HocrPage`` containing the full block/line/word hierarchy.
    """
    logger.info("parsing_hocr", chars=len(hocr_html), dpi=dpi)

    # lxml rejects Unicode strings with an XML encoding declaration.
    # Encode to bytes so the parser can handle the declaration properly.
    hocr_bytes = hocr_html.encode("utf-8") if isinstance(hocr_html, str) else hocr_html
    doc = html.fromstring(hocr_bytes)

    # Find <body> — lxml usually returns the <html> root; body is a child.
    body_elements = doc.xpath("//body")
    if not body_elements:
        return HocrPage(dpi=dpi)

    body = body_elements[0]

    page_div = _find_by_class(body, "ocr_page")
    if page_div is None:
        return HocrPage(dpi=dpi)

    page_title = parse_title(page_div.get("title"))
    page_bbox = page_title.bbox or BBox()

    parsed_blocks = _parse_blocks(page_div, dpi)

    return HocrPage(
        width=page_bbox.width,
        height=page_bbox.height,
        dpi=dpi,
        blocks=parsed_blocks,
    )
