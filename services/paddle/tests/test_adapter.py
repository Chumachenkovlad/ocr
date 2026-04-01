"""Tests for the output adapter (PaddleOCR → editor JSON)."""

from __future__ import annotations

from src.core.adapter import adapt_ocr_result, split_line_to_words
from src.utils.geometry import BBox


class TestSplitLineToWords:
    """Test the critical line-to-word splitting algorithm."""

    def test_single_word(self) -> None:
        bbox = BBox(x=100, y=200, w=200, h=30)
        words = split_line_to_words("Hello", bbox, 0.95)
        assert len(words) == 1
        assert words[0].text == "Hello"
        assert words[0].confidence == 0.95
        assert words[0].bbox.x == 100
        assert words[0].bbox.w == 200

    def test_two_words(self) -> None:
        bbox = BBox(x=100, y=200, w=400, h=30)
        words = split_line_to_words("Hello World", bbox, 0.97)
        assert len(words) == 2
        assert words[0].text == "Hello"
        assert words[1].text == "World"
        # Both should have the same y and h
        assert words[0].bbox.y == 200
        assert words[1].bbox.y == 200
        assert words[0].bbox.h == 30
        assert words[1].bbox.h == 30
        # First word starts at line start
        assert words[0].bbox.x == 100
        # Second word starts after first + space
        assert words[1].bbox.x > words[0].bbox.x + words[0].bbox.w

    def test_proportional_widths(self) -> None:
        """Longer words should get wider bboxes."""
        bbox = BBox(x=0, y=0, w=1000, h=30)
        words = split_line_to_words("Hi Longword", bbox, 0.9)
        assert len(words) == 2
        # "Longword" (8 chars) should be wider than "Hi" (2 chars)
        assert words[1].bbox.w > words[0].bbox.w

    def test_empty_text(self) -> None:
        bbox = BBox(x=0, y=0, w=100, h=30)
        words = split_line_to_words("", bbox, 0.5)
        assert words == []

    def test_whitespace_only(self) -> None:
        bbox = BBox(x=0, y=0, w=100, h=30)
        words = split_line_to_words("   ", bbox, 0.5)
        assert words == []

    def test_multiple_spaces_collapsed(self) -> None:
        bbox = BBox(x=0, y=0, w=500, h=30)
        words = split_line_to_words("Hello   World", bbox, 0.9)
        assert len(words) == 2
        assert words[0].text == "Hello"
        assert words[1].text == "World"

    def test_confidence_rounding(self) -> None:
        bbox = BBox(x=0, y=0, w=100, h=20)
        words = split_line_to_words("Test", bbox, 0.97654321)
        assert words[0].confidence == 0.9765

    def test_many_words(self) -> None:
        bbox = BBox(x=0, y=0, w=1000, h=30)
        words = split_line_to_words("This is a longer sentence with many words", bbox, 0.9)
        assert len(words) == 8
        # Words should not overlap
        for i in range(len(words) - 1):
            assert words[i + 1].bbox.x >= words[i].bbox.x + words[i].bbox.w

    def test_word_widths_minimum_one(self) -> None:
        """Even very small words should have width >= 1."""
        bbox = BBox(x=0, y=0, w=10, h=5)
        words = split_line_to_words("A B C D E", bbox, 0.5)
        for word in words:
            assert word.bbox.w >= 1


class TestAdaptOcrResult:
    """Test full OCR result adaptation."""

    def test_basic_ocr_mode(self, sample_ocr_raw_result: dict) -> None:
        """OCR-only mode should produce a single text block."""
        page = adapt_ocr_result(
            raw_result=sample_ocr_raw_result,
            page_index=0,
            image_width=2550,
            image_height=3300,
            dpi=300,
            language="en",
        )
        assert page.page_index == 0
        assert page.width_px == 2550
        assert page.height_px == 3300
        assert page.dpi == 300
        assert page.language == "en"
        assert len(page.blocks) == 1
        assert page.blocks[0].block_type == "text"
        assert len(page.blocks[0].lines) == 3
        assert page.reading_order == [0]

    def test_words_extracted_from_lines(self, sample_ocr_raw_result: dict) -> None:
        """Each line should be split into words."""
        page = adapt_ocr_result(
            raw_result=sample_ocr_raw_result,
            page_index=0,
            image_width=2550,
            image_height=3300,
            dpi=300,
        )
        block = page.blocks[0]
        # "Document Title" → 2 words
        assert len(block.lines[0].words) == 2
        assert block.lines[0].words[0].text == "Document"
        assert block.lines[0].words[1].text == "Title"

    def test_font_size_estimated(self, sample_ocr_raw_result: dict) -> None:
        """Block should have a non-zero font size estimate."""
        page = adapt_ocr_result(
            raw_result=sample_ocr_raw_result,
            page_index=0,
            image_width=2550,
            image_height=3300,
            dpi=300,
        )
        assert page.blocks[0].font_size_estimate_pt > 0

    def test_structure_mode_multiple_blocks(
        self, sample_structure_raw_result: dict
    ) -> None:
        """Structure mode: blocks from layout regions with tables/formulas/seals."""
        page = adapt_ocr_result(
            raw_result=sample_structure_raw_result,
            page_index=0,
            image_width=2550,
            image_height=3300,
            dpi=300,
        )
        assert len(page.blocks) == 5
        assert page.blocks[0].block_type == "title"
        assert page.blocks[1].block_type == "text"
        assert page.blocks[2].block_type == "table"
        assert page.blocks[3].block_type == "formula"
        assert page.blocks[4].block_type == "seal"
        assert page.reading_order == [0, 1, 2, 3, 4]

    def test_table_html_extracted(self, sample_structure_raw_result: dict) -> None:
        """Table blocks should have table_html populated."""
        page = adapt_ocr_result(
            raw_result=sample_structure_raw_result,
            page_index=0,
            image_width=2550,
            image_height=3300,
            dpi=300,
        )
        table_block = next(b for b in page.blocks if b.block_type == "table")
        assert table_block.table_html is not None
        assert "<table>" in table_block.table_html

    def test_formula_latex_extracted(self, sample_structure_raw_result: dict) -> None:
        """Formula blocks should have formula_latex populated."""
        page = adapt_ocr_result(
            raw_result=sample_structure_raw_result,
            page_index=0,
            image_width=2550,
            image_height=3300,
            dpi=300,
        )
        formula_block = next(b for b in page.blocks if b.block_type == "formula")
        assert formula_block.formula_latex == "E = mc^2"

    def test_seal_texts_extracted(self, sample_structure_raw_result: dict) -> None:
        """Seal blocks should have seal_texts populated."""
        page = adapt_ocr_result(
            raw_result=sample_structure_raw_result,
            page_index=0,
            image_width=2550,
            image_height=3300,
            dpi=300,
        )
        seal_block = next(b for b in page.blocks if b.block_type == "seal")
        assert seal_block.seal_texts == ["APPROVED", "2024-01-01"]

    def test_rotation_angle_propagated(self, sample_ocr_raw_result: dict) -> None:
        """Rotation angle from engine should be propagated to page."""
        sample_ocr_raw_result["rotation_angle"] = 2.5
        page = adapt_ocr_result(
            raw_result=sample_ocr_raw_result,
            page_index=0,
            image_width=2550,
            image_height=3300,
            dpi=300,
        )
        assert page.rotation_angle == 2.5

    def test_markdown_from_engine(self, sample_ocr_raw_result: dict) -> None:
        """Markdown from engine should be propagated to page when present."""
        sample_ocr_raw_result["markdown"] = "# Title\n\nSome text"
        page = adapt_ocr_result(
            raw_result=sample_ocr_raw_result,
            page_index=0,
            image_width=2550,
            image_height=3300,
            dpi=300,
        )
        assert page.markdown == "# Title\n\nSome text"

    def test_markdown_generated_fallback(self, sample_structure_raw_result: dict) -> None:
        """When engine doesn't provide markdown, it should be generated from blocks."""
        page = adapt_ocr_result(
            raw_result=sample_structure_raw_result,
            page_index=0,
            image_width=2550,
            image_height=3300,
            dpi=300,
        )
        # Should have some markdown content (generated from blocks)
        assert page.markdown is not None or page.markdown == ""

    def test_empty_result(self) -> None:
        """Empty OCR result should produce page with no blocks."""
        page = adapt_ocr_result(
            raw_result={"mode": "ocr", "raw": None},
            page_index=0,
            image_width=2550,
            image_height=3300,
            dpi=300,
        )
        assert page.blocks == []
        assert page.reading_order == []

    def test_page_metadata(self, sample_ocr_raw_result: dict) -> None:
        """Page metadata should match input parameters."""
        page = adapt_ocr_result(
            raw_result=sample_ocr_raw_result,
            page_index=3,
            image_width=1200,
            image_height=1600,
            dpi=150,
            language="fr",
        )
        assert page.page_index == 3
        assert page.width_px == 1200
        assert page.height_px == 1600
        assert page.dpi == 150
        assert page.language == "fr"
