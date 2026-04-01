"""Tests for font size estimation."""

from __future__ import annotations

import pytest
from src.utils.font_estimator import estimate_block_font_size, estimate_font_size_pt


class TestEstimateFontSizePt:
    """Test single-line font size estimation."""

    def test_standard_12pt_at_300dpi(self) -> None:
        """12pt text at 300 DPI should have ~50px bbox height."""
        # 12pt = (50 / 300) * 72 = 12.0
        result = estimate_font_size_pt(50, 300)
        assert result == 12.0

    def test_standard_10pt_at_300dpi(self) -> None:
        # 10pt ≈ (42 / 300) * 72 = 10.08 → 10.1
        result = estimate_font_size_pt(42, 300)
        assert result == pytest.approx(10.1, abs=0.05)

    def test_72dpi_identity(self) -> None:
        """At 72 DPI, pixels equal points."""
        assert estimate_font_size_pt(12, 72) == 12.0
        assert estimate_font_size_pt(24, 72) == 24.0

    def test_zero_height(self) -> None:
        assert estimate_font_size_pt(0, 300) == 0.0

    def test_invalid_dpi_raises(self) -> None:
        with pytest.raises(ValueError, match="DPI must be positive"):
            estimate_font_size_pt(50, 0)
        with pytest.raises(ValueError, match="DPI must be positive"):
            estimate_font_size_pt(50, -1)

    def test_negative_height_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            estimate_font_size_pt(-10, 300)


class TestEstimateBlockFontSize:
    """Test block-level font size from multiple line heights."""

    def test_single_line(self) -> None:
        result = estimate_block_font_size([50], 300)
        assert result == 12.0

    def test_median_of_odd_count(self) -> None:
        # Heights: [40, 50, 60] → median = 50
        result = estimate_block_font_size([40, 50, 60], 300)
        assert result == 12.0

    def test_median_of_even_count(self) -> None:
        # Heights: [40, 50] → median = 45 → (45/300)*72 = 10.8
        result = estimate_block_font_size([40, 50], 300)
        assert result == 10.8

    def test_empty_list(self) -> None:
        assert estimate_block_font_size([], 300) == 0.0

    def test_unsorted_input(self) -> None:
        """Input order shouldn't affect median calculation."""
        result_sorted = estimate_block_font_size([30, 40, 50], 300)
        result_unsorted = estimate_block_font_size([50, 30, 40], 300)
        assert result_sorted == result_unsorted
