"""Split a recognised text line into individual word bounding boxes.

Uses a character-proportional width distribution (monospace approximation):
each word's width is proportional to its character count relative to the
total characters in the line.  Ported from the PaddleOCR service's
``split_line_to_words`` and adapted to the shared ``BoundingBox`` model.
"""

from __future__ import annotations

from ocr_schema.models import BoundingBox, OcrWord


def split_line_to_words(
    text: str,
    line_bbox: BoundingBox,
    confidence: float,
) -> list[OcrWord]:
    """Split *text* into whitespace-delimited words with estimated bboxes.

    Args:
        text: Full recognised text for the line.
        line_bbox: Bounding box of the entire text line (pixel coords).
        confidence: Recognition confidence propagated to every word.

    Returns:
        List of ``OcrWord`` objects with computed bounding boxes.
    """
    raw_words = text.split()
    if not raw_words:
        return []

    # Single word -- reuse the full line bbox.
    if len(raw_words) == 1:
        return [
            OcrWord(
                text=raw_words[0],
                confidence=round(confidence, 4),
                bbox=line_bbox,
            )
        ]

    total_chars = sum(len(w) for w in raw_words)
    if total_chars == 0:
        return []

    space_width = line_bbox.width * 0.02  # approximate inter-word gap
    usable_width = line_bbox.width - space_width * (len(raw_words) - 1)
    current_x = line_bbox.x1

    words: list[OcrWord] = []
    for word_text in raw_words:
        word_ratio = len(word_text) / total_chars
        word_width = usable_width * word_ratio
        words.append(
            OcrWord(
                text=word_text,
                confidence=round(confidence, 4),
                bbox=BoundingBox(
                    x1=round(current_x, 1),
                    y1=line_bbox.y1,
                    x2=round(current_x + word_width, 1),
                    y2=line_bbox.y2,
                ),
            )
        )
        current_x += word_width + space_width

    return words
