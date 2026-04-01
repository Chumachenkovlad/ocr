"""Geometry utilities for bounding box operations and polygon conversion."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box (x, y, width, height)."""

    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def area(self) -> int:
        return self.w * self.h

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


def polygon_to_bbox(polygon: Sequence[Sequence[float]]) -> BBox:
    """Convert a detection polygon (4+ points) to an axis-aligned BBox.

    Args:
        polygon: List of [x, y] coordinate pairs.

    Returns:
        Minimum bounding rectangle as BBox.
    """
    if len(polygon) < 3:
        msg = f"Polygon must have at least 3 points, got {len(polygon)}"
        raise ValueError(msg)

    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]

    x_min = int(min(xs))
    y_min = int(min(ys))
    x_max = int(max(xs))
    y_max = int(max(ys))

    return BBox(x=x_min, y=y_min, w=x_max - x_min, h=y_max - y_min)


def compute_iou(a: BBox, b: BBox) -> float:
    """Compute Intersection over Union between two bounding boxes."""
    x_overlap = max(0, min(a.x2, b.x2) - max(a.x, b.x))
    y_overlap = max(0, min(a.y2, b.y2) - max(a.y, b.y))
    intersection = x_overlap * y_overlap

    union = a.area + b.area - intersection
    if union == 0:
        return 0.0
    return intersection / union


def bbox_contains(outer: BBox, inner: BBox, threshold: float = 0.8) -> bool:
    """Check if inner bbox is mostly contained within outer bbox.

    Args:
        outer: The containing bounding box.
        inner: The potentially contained bounding box.
        threshold: Minimum overlap ratio (0.0-1.0) of inner's area that
                   must fall within outer.
    """
    if inner.area == 0:
        return False

    x_overlap = max(0, min(outer.x2, inner.x2) - max(outer.x, inner.x))
    y_overlap = max(0, min(outer.y2, inner.y2) - max(outer.y, inner.y))
    intersection = x_overlap * y_overlap

    return (intersection / inner.area) >= threshold
