"""Tests for geometry utilities."""

from __future__ import annotations

import pytest
from src.utils.geometry import BBox, bbox_contains, compute_iou, polygon_to_bbox


class TestBBox:
    """Test BBox dataclass."""

    def test_basic_properties(self) -> None:
        bbox = BBox(x=10, y=20, w=100, h=50)
        assert bbox.x2 == 110
        assert bbox.y2 == 70
        assert bbox.area == 5000

    def test_to_dict(self) -> None:
        bbox = BBox(x=1, y=2, w=3, h=4)
        assert bbox.to_dict() == {"x": 1, "y": 2, "w": 3, "h": 4}

    def test_frozen(self) -> None:
        """BBox should be immutable."""
        bbox = BBox(x=0, y=0, w=10, h=10)
        with pytest.raises(AttributeError):
            bbox.x = 5  # type: ignore[misc]

    def test_zero_area(self) -> None:
        bbox = BBox(x=0, y=0, w=0, h=0)
        assert bbox.area == 0


class TestPolygonToBBox:
    """Test polygon to bounding box conversion."""

    def test_rectangle_polygon(self) -> None:
        poly = [[100, 200], [500, 200], [500, 230], [100, 230]]
        bbox = polygon_to_bbox(poly)
        assert bbox == BBox(x=100, y=200, w=400, h=30)

    def test_skewed_polygon(self) -> None:
        """Skewed polygons should still produce axis-aligned bbox."""
        poly = [[105, 200], [500, 195], [498, 230], [100, 235]]
        bbox = polygon_to_bbox(poly)
        assert bbox == BBox(x=100, y=195, w=400, h=40)

    def test_triangle_polygon(self) -> None:
        poly = [[0, 0], [100, 0], [50, 50]]
        bbox = polygon_to_bbox(poly)
        assert bbox == BBox(x=0, y=0, w=100, h=50)

    def test_too_few_points_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 3 points"):
            polygon_to_bbox([[0, 0], [1, 1]])


class TestComputeIoU:
    """Test intersection over union."""

    def test_identical_boxes(self) -> None:
        bbox = BBox(x=0, y=0, w=100, h=100)
        assert compute_iou(bbox, bbox) == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        a = BBox(x=0, y=0, w=50, h=50)
        b = BBox(x=100, y=100, w=50, h=50)
        assert compute_iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        a = BBox(x=0, y=0, w=100, h=100)
        b = BBox(x=50, y=50, w=100, h=100)
        # intersection: 50*50 = 2500
        # union: 10000 + 10000 - 2500 = 17500
        assert compute_iou(a, b) == pytest.approx(2500 / 17500)

    def test_zero_area(self) -> None:
        a = BBox(x=0, y=0, w=0, h=0)
        b = BBox(x=0, y=0, w=0, h=0)
        assert compute_iou(a, b) == 0.0


class TestBBoxContains:
    """Test containment check."""

    def test_fully_contained(self) -> None:
        outer = BBox(x=0, y=0, w=200, h=200)
        inner = BBox(x=50, y=50, w=50, h=50)
        assert bbox_contains(outer, inner) is True

    def test_not_contained(self) -> None:
        outer = BBox(x=0, y=0, w=50, h=50)
        inner = BBox(x=100, y=100, w=50, h=50)
        assert bbox_contains(outer, inner) is False

    def test_zero_area_inner(self) -> None:
        outer = BBox(x=0, y=0, w=100, h=100)
        inner = BBox(x=50, y=50, w=0, h=0)
        assert bbox_contains(outer, inner) is False
