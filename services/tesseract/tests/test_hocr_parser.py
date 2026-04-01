"""Tests for the hOCR parser module."""

from __future__ import annotations

import pytest

from src.hocr_parser import BBox, TitleProps, parse_hocr, parse_title


class TestParseTitle:
    """Tests for parse_title()."""

    def test_empty_title(self) -> None:
        result = parse_title(None)
        assert result == TitleProps()

    def test_empty_string(self) -> None:
        result = parse_title("")
        assert result == TitleProps()

    def test_bbox_only(self) -> None:
        result = parse_title("bbox 100 200 300 250")
        assert result.bbox == BBox(x1=100, y1=200, x2=300, y2=250)
        assert result.x_wconf is None
        assert result.baseline is None

    def test_bbox_and_wconf(self) -> None:
        result = parse_title("bbox 100 200 300 250; x_wconf 95")
        assert result.bbox == BBox(x1=100, y1=200, x2=300, y2=250)
        assert result.x_wconf == 95

    def test_full_title(self) -> None:
        result = parse_title("bbox 10 20 30 40; baseline 0 -5; x_wconf 88")
        assert result.bbox == BBox(x1=10, y1=20, x2=30, y2=40)
        assert result.baseline == (0.0, -5.0)
        assert result.x_wconf == 88

    def test_image_title(self) -> None:
        """Page-level title with image metadata."""
        result = parse_title('image "test.png"; bbox 0 0 2550 3300; ppageno 0')
        assert result.bbox == BBox(x1=0, y1=0, x2=2550, y2=3300)

    def test_negative_baseline(self) -> None:
        result = parse_title("bbox 0 0 100 50; baseline 0.002 -7")
        assert result.baseline == (0.002, -7.0)


class TestBBox:
    """Tests for BBox dataclass."""

    def test_width_height(self) -> None:
        bbox = BBox(x1=100, y1=200, x2=400, y2=350)
        assert bbox.width == 300
        assert bbox.height == 150

    def test_zero_bbox(self) -> None:
        bbox = BBox()
        assert bbox.width == 0
        assert bbox.height == 0


class TestParseHocr:
    """Tests for parse_hocr()."""

    def test_full_document(self, sample_hocr: str) -> None:
        page = parse_hocr(sample_hocr, dpi=300)

        assert page.width == 2550
        assert page.height == 3300
        assert page.dpi == 300
        assert len(page.blocks) == 2

        # First block: 2 lines
        block0 = page.blocks[0]
        assert block0.block_index == 0
        assert len(block0.lines) == 2
        assert block0.bbox == BBox(x1=100, y1=100, x2=2450, y2=500)

        # First line
        line0 = block0.lines[0]
        assert len(line0.words) == 2
        assert line0.words[0].text == "Hello"
        assert line0.words[0].confidence == pytest.approx(0.95)
        assert line0.words[0].bbox == BBox(x1=100, y1=100, x2=250, y2=140)
        assert line0.words[1].text == "World"
        assert line0.words[1].confidence == pytest.approx(0.92)

        # Baseline calculation: y1 + height + baseline_offset
        # line bbox = (100, 100, 800, 140), height=40, baseline_offset=-5
        assert line0.baseline == 100 + 40 + (-5)  # 135

        # Second block: 1 line
        block1 = page.blocks[1]
        assert block1.block_index == 1
        assert len(block1.lines) == 1
        assert block1.lines[0].words[0].text == "Another"
        assert block1.lines[0].words[1].text == "block"

    def test_empty_page(self, sample_hocr_empty_page: str) -> None:
        page = parse_hocr(sample_hocr_empty_page, dpi=300)

        assert page.width == 800
        assert page.height == 600
        assert len(page.blocks) == 0

    def test_no_carea_wrapper(self, sample_hocr_no_carea: str) -> None:
        """Paragraphs directly under ocr_page (no ocr_carea wrapper)."""
        page = parse_hocr(sample_hocr_no_carea, dpi=300)

        assert page.width == 1000
        assert page.height == 800
        assert len(page.blocks) == 1
        assert page.blocks[0].lines[0].words[0].text == "Direct"
        assert page.blocks[0].lines[0].words[1].text == "paragraph"

    def test_missing_confidence(self, sample_hocr_missing_conf: str) -> None:
        """Words without x_wconf should get confidence=0.0."""
        page = parse_hocr(sample_hocr_missing_conf, dpi=300)

        assert len(page.blocks) == 1
        word = page.blocks[0].lines[0].words[0]
        assert word.text == "NoConf"
        assert word.confidence == 0.0

    def test_no_body(self) -> None:
        """HTML without a body element returns an empty page."""
        page = parse_hocr("<html><head></head></html>", dpi=150)
        assert page.width == 0
        assert page.height == 0
        assert len(page.blocks) == 0

    def test_no_page_div(self) -> None:
        """Body without ocr_page div returns an empty page."""
        page = parse_hocr("<html><body><div>nothing</div></body></html>", dpi=150)
        assert page.width == 0
        assert page.height == 0
        assert len(page.blocks) == 0

    def test_font_size_estimate(self, sample_hocr: str) -> None:
        """Font size estimate = (height * 72) / dpi."""
        page = parse_hocr(sample_hocr, dpi=300)
        word = page.blocks[0].lines[0].words[0]
        # bbox height = 140 - 100 = 40
        expected = (40 * 72) / 300
        assert word.font_size_estimate == pytest.approx(expected)
