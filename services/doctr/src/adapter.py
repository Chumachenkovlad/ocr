"""Convert docTR export dict to canonical OcrResult.

docTR outputs normalised coordinates (0-1).  This adapter denormalises
them into pixel coordinates using the per-page ``dimensions`` field.

Key detail: ``dimensions`` is ``[height, width]`` (height first).
"""

from ocr_schema.models import (
    BoundingBox,
    OcrBlock,
    OcrLine,
    OcrPage,
    OcrResult,
    OcrWord,
)


def denorm_bbox(
    geometry: list[list[float]],
    page_height: float,
    page_width: float,
) -> BoundingBox:
    """Convert docTR normalised geometry to pixel BoundingBox.

    ``geometry`` is ``[[x_min, y_min], [x_max, y_max]]`` with values in 0-1.
    """
    (x_min, y_min), (x_max, y_max) = geometry
    return BoundingBox(
        x1=round(x_min * page_width, 1),
        y1=round(y_min * page_height, 1),
        x2=round(x_max * page_width, 1),
        y2=round(y_max * page_height, 1),
    )


def adapt_doctr_result(exported: dict, latency_ms: float) -> OcrResult:
    """Transform the dict from ``result.export()`` into an ``OcrResult``."""
    pages: list[OcrPage] = []

    for page_idx, page_data in enumerate(exported.get("pages", [])):
        # dimensions is [height, width]
        dims = page_data["dimensions"]
        page_h: float = dims[0]
        page_w: float = dims[1]

        blocks: list[OcrBlock] = []
        for block_data in page_data.get("blocks", []):
            lines: list[OcrLine] = []
            for line_data in block_data.get("lines", []):
                words: list[OcrWord] = []
                for word_data in line_data.get("words", []):
                    words.append(
                        OcrWord(
                            text=word_data["value"],
                            confidence=round(
                                word_data.get("confidence", -1.0), 4
                            ),
                            bbox=denorm_bbox(
                                word_data["geometry"], page_h, page_w
                            ),
                        )
                    )

                line_text = " ".join(w.text for w in words)
                lines.append(
                    OcrLine(
                        text=line_text,
                        words=words,
                        bbox=denorm_bbox(
                            line_data["geometry"], page_h, page_w
                        ),
                    )
                )

            blocks.append(
                OcrBlock(
                    block_type="text",
                    lines=lines,
                    bbox=denorm_bbox(
                        block_data["geometry"], page_h, page_w
                    ),
                )
            )

        pages.append(
            OcrPage(
                page_number=page_idx,
                width=page_w,
                height=page_h,
                blocks=blocks,
            )
        )

    return OcrResult(
        provider="doctr",
        pages=pages,
        raw=exported,
        latency_ms=latency_ms,
    )
