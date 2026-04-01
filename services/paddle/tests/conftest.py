"""Shared test fixtures."""

from __future__ import annotations

from enum import Enum
from unittest.mock import AsyncMock, MagicMock

import pytest
from src.config import Settings


@pytest.fixture
def test_settings() -> Settings:
    """Settings configured for testing."""
    return Settings(
        dpi=300,
        max_file_size_mb=10,
        max_pages=5,
        device="cpu",  # type: ignore[arg-type]
        mode="structure",  # type: ignore[arg-type]
        request_timeout_s=30,
        max_concurrent=2,
        log_level="DEBUG",
        model_dir="/var/tmp/test-models",  # noqa: S108
        lang="en",
        host="127.0.0.1",
        port=8080,
    )


class _FakeBackend(str, Enum):
    rapid = "rapid_ocr"


@pytest.fixture
def mock_ocr_engine() -> MagicMock:
    """Mock OCR engine that returns realistic-looking results."""
    engine = MagicMock()
    engine.models_loaded = True
    engine.backend = _FakeBackend.rapid
    engine.process_image = AsyncMock(
        return_value={
            "mode": "ocr",
            "raw": [
                {
                    "dt_polys": [
                        [[100, 200], [500, 200], [500, 230], [100, 230]],
                        [[100, 250], [600, 250], [600, 280], [100, 280]],
                    ],
                    "rec_texts": ["Hello World", "This is a test document"],
                    "rec_scores": [0.97, 0.95],
                }
            ],
        }
    )
    return engine


@pytest.fixture
def sample_ocr_raw_result() -> dict:
    """Realistic PaddleOCR raw result for adapter tests."""
    return {
        "mode": "ocr",
        "raw": [
            {
                "dt_polys": [
                    [[100, 100], [400, 100], [400, 130], [100, 130]],
                    [[100, 150], [500, 150], [500, 180], [100, 180]],
                    [[100, 200], [350, 200], [350, 230], [100, 230]],
                ],
                "rec_texts": [
                    "Document Title",
                    "This is the first paragraph of text.",
                    "Second paragraph here.",
                ],
                "rec_scores": [0.99, 0.96, 0.94],
            }
        ],
    }


@pytest.fixture
def sample_structure_raw_result() -> dict:
    """Realistic PP-StructureV3 result with layout regions."""
    return {
        "mode": "structure",
        "raw": [
            {
                "dt_polys": [
                    [[100, 100], [400, 100], [400, 130], [100, 130]],
                    [[100, 200], [500, 200], [500, 230], [100, 230]],
                    [[100, 250], [500, 250], [500, 280], [100, 280]],
                ],
                "rec_texts": ["Title Text", "Body paragraph one.", "Body paragraph two."],
                "rec_scores": [0.99, 0.95, 0.93],
                "layout_result": [
                    {
                        "bbox": [80, 80, 420, 140],
                        "label": "title",
                    },
                    {
                        "bbox": [80, 180, 520, 300],
                        "label": "text",
                    },
                    {
                        "bbox": [80, 320, 520, 500],
                        "label": "table",
                        "table_html": "<table><tr><td>A</td><td>B</td></tr></table>",
                    },
                    {
                        "bbox": [80, 520, 520, 600],
                        "label": "formula",
                        "latex": "E = mc^2",
                    },
                    {
                        "bbox": [80, 620, 520, 700],
                        "label": "seal",
                        "seal_texts": ["APPROVED", "2024-01-01"],
                    },
                ],
            }
        ],
    }
