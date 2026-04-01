"""Tests for the adapter module."""

from __future__ import annotations

import pytest

from src.adapter import adapt_page, build_ocr_result
from src.hocr_parser import BBox, HocrBlock, HocrLine, HocrPage, HocrWord


class TestAdaptPage:
    """Tests for adapt_page()."""

    def test_basic_conversion(self) -> None:
        hocr_page = HocrPage(
            width=800,
            height=600,
            dpi=300,
            blocks=(
                HocrBlock(
                    bbox=BBox(x1=10, y1=20, x2=400, y2=200),
                    lines=(
                        HocrLine(
                            bbox=BBox(x1=10, y1=20, x2=400, y2=50),
                            baseline=45,
                            words=(
                                HocrWord(
                                    text="Hello",
                                    bbox=BBox(x1=10, y1=20, x2=100, y2=50),
                                    confidence=0.95,
                                    font_size_estimate=7.2,
                                ),
                                HocrWord(
                                    text="World",
                                    bbox=BBox(x1=120, y1=20, x2=250, y2=50),
                                    confidence=0.88,
                                    font_size_estimate=7.2,
                                ),
                            ),
                        ),
                    ),
                    block_index=0,
                ),
            ),
        )

        page = adapt_page(hocr_page, page_number=0)

        assert page.page_number == 0
        assert page.width == 800.0
        assert page.height == 600.0
        assert len(page.blocks) == 1

        block = page.blocks[0]
        assert block.block_type == "text"
        assert block.bbox.x1 == 10.0
        assert block.bbox.y1 == 20.0
        assert block.bbox.x2 == 400.0
        assert block.bbox.y2 == 200.0

        line = block.lines[0]
        assert line.text == "Hello World"
        assert len(line.words) == 2
        assert line.words[0].text == "Hello"
        assert line.words[0].confidence == pytest.approx(0.95)
        assert line.words[0].bbox.x1 == 10.0
        assert line.words[1].text == "World"

    def test_empty_page_gets_minimum_dimensions(self) -> None:
        """Pages with 0 width/height get clamped to 1."""
        hocr_page = HocrPage(width=0, height=0, dpi=300)
        page = adapt_page(hocr_page, page_number=5)
        assert page.page_number == 5
        assert page.width == 1.0
        assert page.height == 1.0
        assert len(page.blocks) == 0

    def test_multiple_blocks(self) -> None:
        hocr_page = HocrPage(
            width=1000,
            height=800,
            dpi=300,
            blocks=(
                HocrBlock(
                    bbox=BBox(x1=10, y1=10, x2=500, y2=100),
                    lines=(
                        HocrLine(
                            bbox=BBox(x1=10, y1=10, x2=500, y2=50),
                            baseline=45,
                            words=(
                                HocrWord(text="A", bbox=BBox(x1=10, y1=10, x2=50, y2=50),
                                         confidence=0.9, font_size_estimate=9.6),
                            ),
                        ),
                    ),
                    block_index=0,
                ),
                HocrBlock(
                    bbox=BBox(x1=10, y1=200, x2=500, y2=300),
                    lines=(
                        HocrLine(
                            bbox=BBox(x1=10, y1=200, x2=500, y2=240),
                            baseline=235,
                            words=(
                                HocrWord(text="B", bbox=BBox(x1=10, y1=200, x2=50, y2=240),
                                         confidence=0.8, font_size_estimate=9.6),
                            ),
                        ),
                    ),
                    block_index=1,
                ),
            ),
        )

        page = adapt_page(hocr_page, page_number=2)
        assert len(page.blocks) == 2
        assert page.blocks[0].lines[0].words[0].text == "A"
        assert page.blocks[1].lines[0].words[0].text == "B"


class TestBuildOcrResult:
    """Tests for build_ocr_result()."""

    def test_basic_result(self) -> None:
        from ocr_schema import BoundingBox, OcrBlock, OcrLine, OcrPage, OcrWord

        page = OcrPage(
            page_number=0,
            width=800.0,
            height=600.0,
            blocks=[
                OcrBlock(
                    block_type="text",
                    lines=[
                        OcrLine(
                            text="Hello",
                            words=[
                                OcrWord(
                                    text="Hello",
                                    confidence=0.95,
                                    bbox=BoundingBox(x1=10, y1=20, x2=100, y2=50),
                                ),
                            ],
                            bbox=BoundingBox(x1=10, y1=20, x2=100, y2=50),
                        ),
                    ],
                    bbox=BoundingBox(x1=10, y1=20, x2=400, y2=200),
                ),
            ],
        )

        result = build_ocr_result([page], raw_hocr=["<html>...</html>"], latency_ms=123.4)
        assert result.provider == "tesseract"
        assert len(result.pages) == 1
        assert result.latency_ms == 123.4
        assert result.raw is not None
        assert "hocr" in result.raw
        assert len(result.raw["hocr"]) == 1

    def test_no_raw(self) -> None:
        result = build_ocr_result([], latency_ms=0.0)
        assert result.provider == "tesseract"
        assert result.raw is None
        assert len(result.pages) == 0
