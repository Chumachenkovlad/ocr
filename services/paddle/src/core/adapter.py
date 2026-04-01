"""Output adapter: transforms PaddleOCR results into the editor's canonical JSON schema."""

from __future__ import annotations

from typing import Any

import structlog

from src.api.schemas import (
    BBoxSchema,
    BlockSchema,
    LineSchema,
    PageSchema,
    WordSchema,
)
from src.utils.font_estimator import estimate_block_font_size
from src.utils.geometry import BBox, polygon_to_bbox

logger = structlog.get_logger()


def split_line_to_words(
    line_text: str,
    line_bbox: BBox,
    line_confidence: float,
) -> list[WordSchema]:
    """Split a recognized text line into individual words.

    Uses character-proportional width distribution (monospace approximation).
    Each word's width is proportional to its character count relative to the
    total characters in the line.

    Args:
        line_text: Full recognized text for the line.
        line_bbox: Bounding box of the text line.
        line_confidence: Confidence score for the line (propagated to words).

    Returns:
        List of WordSchema objects with computed bounding boxes.
    """
    raw_words = line_text.split()
    if not raw_words:
        return []

    # Single word: use the full line bbox
    if len(raw_words) == 1:
        return [
            WordSchema(
                text=raw_words[0],
                confidence=round(line_confidence, 4),
                bbox=BBoxSchema(**line_bbox.to_dict()),
            )
        ]

    total_chars = sum(len(w) for w in raw_words)
    if total_chars == 0:
        return []

    words: list[WordSchema] = []
    space_width = line_bbox.w * 0.02  # approximate inter-word gap
    usable_width = line_bbox.w - space_width * (len(raw_words) - 1)
    current_x = float(line_bbox.x)

    for word_text in raw_words:
        word_ratio = len(word_text) / total_chars
        word_width = usable_width * word_ratio

        words.append(
            WordSchema(
                text=word_text,
                confidence=round(line_confidence, 4),
                bbox=BBoxSchema(
                    x=round(current_x),
                    y=line_bbox.y,
                    w=max(1, round(word_width)),
                    h=line_bbox.h,
                ),
            )
        )
        current_x += word_width + space_width

    return words


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    """Get a value from a dict-like or attribute-based PaddleX result object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_ocr_fields(obj: Any) -> tuple[list, list, list]:
    """Extract (dt_polys, rec_texts, rec_scores) from an OCR result object."""
    polys = _safe_get(obj, "dt_polys", [])
    texts = _safe_get(obj, "rec_texts", [])
    scores = _safe_get(obj, "rec_scores", [])
    # Handle numpy arrays
    if hasattr(polys, "tolist"):
        polys = polys.tolist()
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    return polys or [], texts or [], scores or []


def _extract_lines_from_ocr_result(
    raw_result: Any,
) -> list[tuple[list[list[float]], str, float]]:
    """Extract (polygon, text, confidence) tuples from raw PaddleOCR output.

    Handles both PaddleOCR predict() and PPStructureV3 LayoutParsingResultV2.
    """
    lines: list[tuple[list[list[float]], str, float]] = []

    if not raw_result:
        return lines

    results = raw_result if isinstance(raw_result, list) else [raw_result]

    for result in results:
        # Try direct OCR fields (PaddleOCR predict format)
        polys, texts, scores = _get_ocr_fields(result)

        # If not at top level, try inside overall_ocr_res (PPStructureV3 format)
        if not polys:
            ocr_res = _safe_get(result, "overall_ocr_res")
            if ocr_res is not None:
                polys, texts, scores = _get_ocr_fields(ocr_res)

        for i in range(min(len(polys), len(texts), len(scores))):
            poly = polys[i]
            text = texts[i] if i < len(texts) else ""
            score = scores[i] if i < len(scores) else 0.0
            if text.strip():
                # Convert numpy arrays to lists
                if hasattr(poly, "tolist"):
                    poly = poly.tolist()
                lines.append((poly, text, float(score)))

    return lines


# Map PPStructureV3 label names to our schema block types
_LABEL_MAP: dict[str, str] = {
    "figure_title": "title",
    "table_title": "title",
    "doc_title": "title",
    "paragraph_title": "title",
    "abstract": "text",
    "content": "text",
    "paragraph": "text",
    "text": "text",
    "table": "table",
    "figure": "figure",
    "formula": "formula",
    "seal": "seal",
    "chart": "chart",
    "header": "header",
    "footer": "footer",
    "caption": "caption",
    "page_number": "footer",
    "reference": "text",
    "equation": "formula",
}


def _scale_bbox(bbox: list, scale_x: float, scale_y: float) -> list:
    """Scale a bbox ([x1,y1,x2,y2] or polygon) by the given factors."""
    if not bbox:
        return bbox
    if len(bbox) == 4 and not isinstance(bbox[0], (list, tuple)):
        return [bbox[0] * scale_x, bbox[1] * scale_y, bbox[2] * scale_x, bbox[3] * scale_y]
    return [[pt[0] * scale_x, pt[1] * scale_y] for pt in bbox]


def _normalize_label(raw_label: str) -> str:
    """Normalize PPStructureV3 labels to our canonical block types."""
    return _LABEL_MAP.get(raw_label, raw_label)


def _extract_layout_regions(raw_result: Any) -> list[dict[str, Any]]:
    """Extract layout regions from PP-StructureV3 output.

    Tries sources in priority order:
    1. parsing_res_list — list of LayoutBlock objects (PPStructureV3 primary)
    2. layout_det_res — DetResult with boxes/labels (if labels present)
    3. layout_result — list of region dicts (legacy format)
    """
    regions: list[dict[str, Any]] = []

    if not raw_result:
        return regions

    results = raw_result if isinstance(raw_result, list) else [raw_result]

    for result in results:
        # Primary: parsing_res_list (list of LayoutBlock objects)
        parsing_list = _safe_get(result, "parsing_res_list", [])
        if parsing_list:
            for block_obj in parsing_list:
                bbox = _safe_get(block_obj, "bbox")
                label = _safe_get(block_obj, "label", "text")
                content = _safe_get(block_obj, "content", "")
                index = _safe_get(block_obj, "index")
                if bbox is None:
                    continue
                if hasattr(bbox, "tolist"):
                    bbox = bbox.tolist()
                regions.append({
                    "bbox": list(bbox),
                    "label": str(label),
                    "content": str(content) if content else "",
                    "index": index,
                })
            logger.info(
                "layout_regions_extracted",
                source="parsing_res_list",
                count=len(regions),
                labels=[r["label"] for r in regions],
            )
            continue

        # Fallback: layout_det_res (DetResult with parallel arrays)
        layout_det = _safe_get(result, "layout_det_res")
        if layout_det is not None:
            boxes = _safe_get(layout_det, "boxes", [])
            labels = (
                _safe_get(layout_det, "labels", [])
                or _safe_get(layout_det, "label_names", [])
                or []
            )
            det_scores = _safe_get(layout_det, "scores", []) or []
            if hasattr(boxes, "tolist"):
                boxes = boxes.tolist()
            if hasattr(labels, "tolist"):
                labels = labels.tolist()
            if hasattr(det_scores, "tolist"):
                det_scores = det_scores.tolist()
            if boxes and labels:
                for i in range(min(len(boxes), len(labels))):
                    regions.append({
                        "bbox": list(boxes[i]) if hasattr(boxes[i], "tolist") else boxes[i],
                        "label": str(labels[i]),
                        "score": float(det_scores[i]) if i < len(det_scores) else 0.0,
                    })
                logger.info(
                    "layout_regions_extracted",
                    source="layout_det_res",
                    count=len(regions),
                    labels=[r["label"] for r in regions],
                )
                continue

        # Legacy: layout_result (list of region dicts)
        layout = _safe_get(result, "layout_result")
        if isinstance(layout, list) and layout:
            regions.extend(layout)
            continue

    return regions


def _extract_rich_content(raw_result: Any) -> dict[str, list[dict[str, Any]]]:
    """Extract tables, formulas, seals from PPStructureV3 *_res_list fields.

    Each item gets its bbox (for spatial matching to layout blocks) plus content.
    """
    content: dict[str, list[dict[str, Any]]] = {
        "tables": [],
        "formulas": [],
        "seals": [],
    }
    if not raw_result:
        return content

    results = raw_result if isinstance(raw_result, list) else [raw_result]
    for result in results:
        for table in _safe_get(result, "table_res_list", []) or []:
            # PPStructureV3: pred_html is the primary key for table HTML
            html = (
                _safe_get(table, "pred_html")
                or _safe_get(table, "html")
                or _safe_get(table, "table_html")
            )
            # Try bbox from table, or from table_ocr_pred sub-result
            bbox = _safe_get(table, "bbox") or _safe_get(table, "box")
            region_id = _safe_get(table, "table_region_id")
            if hasattr(bbox, "tolist"):
                bbox = bbox.tolist()
            content["tables"].append({
                "html": html,
                "bbox": bbox,
                "region_id": region_id,
            })
            logger.info(
                "rich_table_found",
                has_html=html is not None,
                html_len=len(html) if html else 0,
                bbox=bbox,
                region_id=region_id,
            )

        for formula in _safe_get(result, "formula_res_list", []) or []:
            latex = (
                _safe_get(formula, "pred_formula")
                or _safe_get(formula, "latex")
                or _safe_get(formula, "formula")
            )
            bbox = _safe_get(formula, "bbox") or _safe_get(formula, "box")
            region_id = _safe_get(formula, "formula_region_id")
            if hasattr(bbox, "tolist"):
                bbox = bbox.tolist()
            content["formulas"].append({
                "latex": latex,
                "bbox": bbox,
                "region_id": region_id,
            })

        for seal in _safe_get(result, "seal_res_list", []) or []:
            texts = _safe_get(seal, "rec_texts") or _safe_get(seal, "texts")
            bbox = _safe_get(seal, "bbox") or _safe_get(seal, "box")
            if hasattr(bbox, "tolist"):
                bbox = bbox.tolist()
            if texts:
                content["seals"].append({"texts": texts, "bbox": bbox})

    return content


def _attach_rich_content(
    blocks: list[BlockSchema],
    rich: dict[str, list[dict[str, Any]]],
    layout_regions: list[dict[str, Any]],
) -> None:
    """Attach table/formula/seal content to blocks by region_id or bbox overlap."""
    for table_info in rich.get("tables", []):
        html = table_info.get("html")
        if not html:
            continue
        region_id = table_info.get("region_id")
        block = _find_block_by_region_id(blocks, layout_regions, region_id)
        if block is None:
            bbox = table_info.get("bbox")
            block = _find_matching_block(blocks, bbox, "table")
        if block is not None:
            block.table_html = html
            logger.info("table_html_attached", block_index=block.block_index)

    for formula_info in rich.get("formulas", []):
        latex = formula_info.get("latex")
        if not latex:
            continue
        region_id = formula_info.get("region_id")
        block = _find_block_by_region_id(blocks, layout_regions, region_id)
        if block is None:
            bbox = formula_info.get("bbox")
            block = _find_matching_block(blocks, bbox, "formula")
        if block is not None:
            block.formula_latex = latex

    for seal_info in rich.get("seals", []):
        texts = seal_info.get("texts")
        if not texts:
            continue
        bbox = seal_info.get("bbox")
        block = _find_matching_block(blocks, bbox, "seal")
        if block is not None:
            block.seal_texts = [str(t) for t in texts if str(t).strip()]


def _find_block_by_region_id(
    blocks: list[BlockSchema],
    layout_regions: list[dict[str, Any]],
    region_id: Any,
) -> BlockSchema | None:
    """Find a block by matching its index to a table/formula region_id."""
    if region_id is None or not blocks or not layout_regions:
        return None
    # region_id corresponds to the index in parsing_res_list
    for i, region in enumerate(layout_regions):
        if region.get("index") == region_id and i < len(blocks):
            return blocks[i]
    # Direct index match
    if isinstance(region_id, int) and 0 <= region_id < len(blocks):
        return blocks[region_id]
    return None


def _find_matching_block(
    blocks: list[BlockSchema],
    bbox: Any,
    preferred_type: str,
) -> BlockSchema | None:
    """Find the block whose bbox best overlaps with the given bbox."""
    if bbox is None or not blocks:
        return None
    try:
        if len(bbox) == 4 and not isinstance(bbox[0], (list, tuple)):
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
        else:
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
    except (TypeError, IndexError):
        return None

    # Prefer blocks matching the type, then any block containing the center
    for block in blocks:
        b = block.bbox
        if (b.x <= cx <= b.x + b.w and b.y <= cy <= b.y + b.h
                and block.block_type == preferred_type):
            return block
    # Fallback: any block containing center
    for block in blocks:
        b = block.bbox
        if b.x <= cx <= b.x + b.w and b.y <= cy <= b.y + b.h:
            return block
    return None


def adapt_ocr_result(
    raw_result: dict[str, Any],
    page_index: int,
    image_width: int,
    image_height: int,
    dpi: int,
    language: str = "en",
    page_image_base64: str | None = None,
) -> PageSchema:
    """Transform raw PaddleOCR output into the editor's PageSchema."""
    mode = raw_result.get("mode", "ocr")
    raw = raw_result.get("raw")
    rotation_angle = float(raw_result.get("rotation_angle", 0.0))
    page_markdown = raw_result.get("markdown")

    lines_data = _extract_lines_from_ocr_result(raw)
    layout_regions = _extract_layout_regions(raw) if mode == "structure" else []
    rich_content = _extract_rich_content(raw) if mode == "structure" else {}

    logger.info(
        "adapter_extraction",
        mode=mode,
        lines_count=len(lines_data),
        regions_count=len(layout_regions),
        tables=len(rich_content.get("tables", [])),
        formulas=len(rich_content.get("formulas", [])),
        seals=len(rich_content.get("seals", [])),
    )

    # Log coordinate ranges for debugging
    if lines_data:
        all_xs = [pt[0] for poly, _, _ in lines_data for pt in poly]
        all_ys = [pt[1] for poly, _, _ in lines_data for pt in poly]
        if all_xs:
            logger.info(
                "coordinate_range_debug",
                image_w=image_width,
                image_h=image_height,
                coord_max_x=round(max(all_xs), 1),
                coord_max_y=round(max(all_ys), 1),
            )

    if layout_regions:
        blocks = _build_blocks_from_layout(layout_regions, lines_data, dpi)
        _attach_rich_content(blocks, rich_content, layout_regions)
        reading_order = list(range(len(blocks)))
    elif lines_data:
        blocks = _build_single_block(lines_data, dpi)
        reading_order = [0] if blocks else []
    else:
        blocks = []
        reading_order = []

    # Generate simple markdown from text if not provided by engine
    if page_markdown is None and blocks:
        page_markdown = _generate_markdown_from_blocks(blocks)

    return PageSchema(
        page_index=page_index,
        width_px=image_width,
        height_px=image_height,
        dpi=dpi,
        language=language,
        rotation_angle=rotation_angle,
        blocks=blocks,
        reading_order=reading_order,
        markdown=page_markdown,
        page_image_base64=page_image_base64,
    )


def _build_single_block(
    lines_data: list[tuple[list[list[float]], str, float]],
    dpi: int,
) -> list[BlockSchema]:
    """Build a single text block from all detected lines (OCR-only mode)."""
    lines: list[LineSchema] = []
    line_heights: list[int] = []
    all_bboxes: list[BBox] = []

    for i, (poly, text, score) in enumerate(lines_data):
        try:
            bbox = polygon_to_bbox(poly)
        except ValueError:
            continue

        words = split_line_to_words(text, bbox, score)
        lines.append(
            LineSchema(
                line_index=i,
                bbox=BBoxSchema(**bbox.to_dict()),
                words=words,
            )
        )
        line_heights.append(bbox.h)
        all_bboxes.append(bbox)

    if not lines:
        return []

    # Compute enclosing bbox for the block
    x_min = min(b.x for b in all_bboxes)
    y_min = min(b.y for b in all_bboxes)
    x_max = max(b.x2 for b in all_bboxes)
    y_max = max(b.y2 for b in all_bboxes)

    block_bbox = BBox(x=x_min, y=y_min, w=x_max - x_min, h=y_max - y_min)
    font_size = estimate_block_font_size(line_heights, dpi)

    return [
        BlockSchema(
            block_index=0,
            block_type="text",
            bbox=BBoxSchema(**block_bbox.to_dict()),
            lines=lines,
            font_size_estimate_pt=font_size,
        )
    ]


def _build_blocks_from_layout(
    layout_regions: list[dict[str, Any]],
    lines_data: list[tuple[list[list[float]], str, float]],
    dpi: int,
) -> list[BlockSchema]:
    """Build blocks from layout analysis regions, assigning lines to regions."""
    blocks: list[BlockSchema] = []

    for block_idx, region in enumerate(layout_regions):
        # Extract region bbox and type
        region_bbox_raw = region.get("bbox", region.get("box", []))
        region_type = _normalize_label(
            region.get("label", region.get("type", "text"))
        )

        if not region_bbox_raw:
            continue

        # Layout bbox can be [x1, y1, x2, y2] or [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        if len(region_bbox_raw) == 4 and not isinstance(region_bbox_raw[0], (list, tuple)):
            x1, y1, x2, y2 = [int(v) for v in region_bbox_raw]
            region_bbox = BBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1)
        else:
            try:
                region_bbox = polygon_to_bbox(region_bbox_raw)
            except ValueError:
                continue

        # Find lines that fall within this region
        block_lines: list[LineSchema] = []
        line_heights: list[int] = []
        line_idx = 0

        for poly, text, score in lines_data:
            try:
                line_bbox = polygon_to_bbox(poly)
            except ValueError:
                continue

            # Check if line center falls within region
            line_cx = line_bbox.x + line_bbox.w // 2
            line_cy = line_bbox.y + line_bbox.h // 2
            if (
                region_bbox.x <= line_cx <= region_bbox.x2
                and region_bbox.y <= line_cy <= region_bbox.y2
            ):
                words = split_line_to_words(text, line_bbox, score)
                block_lines.append(
                    LineSchema(
                        line_index=line_idx,
                        bbox=BBoxSchema(**line_bbox.to_dict()),
                        words=words,
                    )
                )
                line_heights.append(line_bbox.h)
                line_idx += 1

        font_size = estimate_block_font_size(line_heights, dpi) if line_heights else 0.0

        # Extract rich content for special block types
        table_html = _extract_table_html(region)
        formula_latex = _extract_formula_latex(region)
        seal_texts = _extract_seal_texts(region)

        blocks.append(
            BlockSchema(
                block_index=block_idx,
                block_type=str(region_type),
                bbox=BBoxSchema(**region_bbox.to_dict()),
                lines=block_lines,
                font_size_estimate_pt=font_size,
                table_html=table_html,
                formula_latex=formula_latex,
                seal_texts=seal_texts,
            )
        )

    return blocks


def _extract_table_html(region: dict[str, Any]) -> str | None:
    """Extract HTML table from a layout region if present."""
    # PP-StructureV3 stores table results under various keys
    for key in ("table_html", "html", "table_result"):
        val = region.get(key)
        if isinstance(val, str) and val.strip():
            return val
        if isinstance(val, dict):
            html = val.get("html", val.get("table_html"))
            if isinstance(html, str) and html.strip():
                return html
    return None


def _extract_formula_latex(region: dict[str, Any]) -> str | None:
    """Extract LaTeX formula from a layout region if present."""
    for key in ("formula", "latex", "formula_result", "rec_formula"):
        val = region.get(key)
        if isinstance(val, str) and val.strip():
            return val
        if isinstance(val, dict):
            latex = val.get("latex", val.get("formula"))
            if isinstance(latex, str) and latex.strip():
                return latex
    return None


def _extract_seal_texts(region: dict[str, Any]) -> list[str] | None:
    """Extract recognized seal/stamp texts from a layout region."""
    for key in ("seal_texts", "seal_result", "rec_texts"):
        val = region.get(key)
        if isinstance(val, list) and val:
            return [str(t) for t in val if str(t).strip()]
    return None


def _generate_markdown_from_blocks(blocks: list[BlockSchema]) -> str:
    """Generate a simple Markdown representation from structured blocks."""
    parts: list[str] = []
    for block in blocks:
        if block.block_type == "title":
            text = _block_text(block)
            if text:
                parts.append(f"## {text}\n")
        elif block.block_type == "table" and block.table_html:
            parts.append(block.table_html + "\n")
        elif block.block_type in ("formula", "equation") and block.formula_latex:
            parts.append(f"$$\n{block.formula_latex}\n$$\n")
        elif block.block_type == "seal" and block.seal_texts:
            parts.append("[Seal: " + " | ".join(block.seal_texts) + "]\n")
        elif block.block_type == "table":
            # Table without HTML — use text from lines
            text = _block_text(block)
            if text:
                parts.append(text + "\n")
        else:
            text = _block_text(block)
            if text:
                parts.append(text + "\n")
    return "\n".join(parts) if parts else ""


def _block_text(block: BlockSchema) -> str:
    """Extract plain text from a block's lines."""
    return "\n".join(
        " ".join(w.text for w in line.words) for line in block.lines
    )
