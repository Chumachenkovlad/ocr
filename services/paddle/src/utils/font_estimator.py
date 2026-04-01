"""Font size estimation from bounding box dimensions and DPI."""

from __future__ import annotations


def estimate_font_size_pt(bbox_height_px: int, dpi: int) -> float:
    """Estimate font size in points from a bbox height and rendering DPI.

    Formula: font_size_pt = (bbox_height_px / dpi) * 72

    Args:
        bbox_height_px: Height of the text bounding box in pixels.
        dpi: Dots per inch used when rendering the page image.

    Returns:
        Estimated font size in typographic points.
    """
    if dpi <= 0:
        msg = f"DPI must be positive, got {dpi}"
        raise ValueError(msg)
    if bbox_height_px < 0:
        msg = f"Bbox height must be non-negative, got {bbox_height_px}"
        raise ValueError(msg)
    return round((bbox_height_px / dpi) * 72, 1)


def estimate_block_font_size(line_heights: list[int], dpi: int) -> float:
    """Estimate font size for a block using the median line height.

    Args:
        line_heights: List of line bbox heights in pixels.
        dpi: Dots per inch.

    Returns:
        Estimated font size in points, or 0.0 if no line heights.
    """
    if not line_heights:
        return 0.0

    sorted_heights = sorted(line_heights)
    mid = len(sorted_heights) // 2
    if len(sorted_heights) % 2 == 0:
        median = (sorted_heights[mid - 1] + sorted_heights[mid]) / 2
    else:
        median = sorted_heights[mid]

    return estimate_font_size_pt(int(median), dpi)
