"""Generate visual overlay images and HTML report from OCR benchmark.

Supports both images and PDFs (renders PDF pages to images).
Shows bbox overlays, extracted text, raw JSON, and summary statistics.

Usage:
    python3 benchmark/visualize.py --fixtures-dir test-fixtures/ --host 52.6.114.209
    python3 benchmark/visualize.py --fixtures-dir test-fixtures/pdfs/ --host 52.6.114.209
"""

from __future__ import annotations

import argparse
import html
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import httpx
from PIL import Image, ImageDraw, ImageFont

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".pdf"}

DEFAULT_HOST = "localhost"
DEFAULT_PORTS = {
    "paddle": 8081,
    "tesseract": 8082,
    "surya": 8083,
    "doctr": 8084,
}


def _get_font(size: int = 10) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_overlay(
    image: Image.Image,
    ocr_page: dict,
) -> Image.Image:
    """Draw bounding boxes on a copy of the image for a single page."""
    img = image.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    page_w = ocr_page.get("width", img.width)
    page_h = ocr_page.get("height", img.height)
    scale_x = img.width / page_w if page_w else 1
    scale_y = img.height / page_h if page_h else 1

    for block in ocr_page.get("blocks", []):
        bb = block.get("bbox", {})
        if bb:
            x1 = bb["x1"] * scale_x
            y1 = bb["y1"] * scale_y
            x2 = bb["x2"] * scale_x
            y2 = bb["y2"] * scale_y
            draw.rectangle([x1, y1, x2, y2], fill=(220, 40, 40, 35), outline=(220, 40, 40, 120), width=2)

        for line in block.get("lines", []):
            lb = line.get("bbox", {})
            if lb:
                x1 = lb["x1"] * scale_x
                y1 = lb["y1"] * scale_y
                x2 = lb["x2"] * scale_x
                y2 = lb["y2"] * scale_y
                draw.rectangle([x1, y1, x2, y2], fill=(0, 200, 80, 30), outline=(0, 200, 80, 100), width=1)

            for word in line.get("words", []):
                wb = word.get("bbox", {})
                if not wb:
                    continue
                x1 = wb["x1"] * scale_x
                y1 = wb["y1"] * scale_y
                x2 = wb["x2"] * scale_x
                y2 = wb["y2"] * scale_y
                conf = word.get("confidence", 0)
                if conf >= 0.9:
                    fill = (0, 180, 0, 50)
                    outline = (0, 140, 0, 180)
                elif conf >= 0.7:
                    fill = (255, 180, 0, 50)
                    outline = (200, 140, 0, 180)
                else:
                    fill = (255, 40, 40, 50)
                    outline = (200, 0, 0, 180)
                draw.rectangle([x1, y1, x2, y2], fill=fill, outline=outline, width=1)

    return Image.alpha_composite(img, overlay).convert("RGB")


def pdf_to_images(pdf_path: Path, dpi: int = 150) -> list[Image.Image]:
    """Render PDF pages to PIL images."""
    try:
        from pdf2image import convert_from_path
        return convert_from_path(str(pdf_path), dpi=dpi)
    except ImportError:
        return []
    except Exception:
        return []


def _has_words(ocr_data: dict) -> bool:
    for p in ocr_data.get("pages", []):
        for b in p.get("blocks", []):
            for line in b.get("lines", []):
                if line.get("words"):
                    return True
    return False


def fetch_ocr_result(
    client: httpx.Client,
    host: str,
    provider: str,
    file_bytes: bytes,
    filename: str,
    timeout: float = 300.0,
    retries: int = 1,
) -> dict | None:
    """Send file to a single OCR service with optional retries."""
    port = DEFAULT_PORTS.get(provider)
    if not port:
        return None
    url = f"http://{host}:{port}/api/v1/ocr"
    for attempt in range(1, retries + 1):
        try:
            resp = client.post(url, files={"file": (filename, file_bytes)}, data={"dpi": "300"}, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            if _has_words(data) or attempt == retries:
                return data
            print(f"    {provider}: 0 words on attempt {attempt}/{retries}, retrying...")
        except Exception as e:
            if attempt == retries:
                print(f"    {provider}: ERROR (attempt {attempt}/{retries}) - {e}")
                return None
            print(f"    {provider}: ERROR on attempt {attempt}/{retries}, retrying...")
    return None


def _extract_stats(ocr_data: dict) -> dict:
    """Extract stats from an OCR result."""
    pages = ocr_data.get("pages", [])
    words = []
    for p in pages:
        for b in p.get("blocks", []):
            for line in b.get("lines", []):
                for w in line.get("words", []):
                    words.append(w)
    confs = [w.get("confidence", 0) for w in words if w.get("confidence", -1) >= 0]
    return {
        "page_count": len(pages),
        "word_count": len(words),
        "block_count": sum(len(p.get("blocks", [])) for p in pages),
        "line_count": sum(len(b.get("lines", [])) for p in pages for b in p.get("blocks", [])),
        "avg_confidence": round(statistics.mean(confs), 3) if confs else 0,
        "min_confidence": round(min(confs), 3) if confs else 0,
        "max_confidence": round(max(confs), 3) if confs else 0,
        "latency_ms": round(ocr_data.get("latency_ms", 0)),
        "latency_s": round(ocr_data.get("latency_ms", 0) / 1000, 1),
    }


def _extract_text(ocr_data: dict) -> str:
    """Extract full text from OCR result."""
    lines = []
    for p in ocr_data.get("pages", []):
        for b in p.get("blocks", []):
            for line in b.get("lines", []):
                lines.append(line.get("text", ""))
    return "\n".join(lines)


CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 20px; background: #f5f5f5; color: #333; }
h1 { color: #2c3e50; margin-bottom: 5px; }
h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 8px; margin-top: 40px; }
h3 { color: #2c3e50; margin: 20px 0 10px; }
.subtitle { color: #888; margin-top: 0; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 16px; margin: 16px 0; }
.card { background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); overflow: hidden; }
.card-header { padding: 12px 16px; background: #2c3e50; color: white; display: flex; justify-content: space-between; align-items: center; }
.card-header .provider { font-weight: bold; font-size: 16px; }
.card-header .badge { font-size: 11px; padding: 2px 8px; border-radius: 10px; background: rgba(255,255,255,0.2); }
.card img { width: 100%; height: auto; display: block; cursor: zoom-in; }
.card img.zoomed { cursor: zoom-out; max-width: none; width: auto; }
.card-body { padding: 0; }
.card-stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0; border-bottom: 1px solid #eee; }
.stat { padding: 8px 12px; text-align: center; border-right: 1px solid #eee; }
.stat:last-child { border-right: none; }
.stat-value { font-size: 18px; font-weight: bold; color: #2c3e50; }
.stat-label { font-size: 10px; color: #888; text-transform: uppercase; }
.card-footer { padding: 8px 16px; font-size: 12px; color: #666; background: #fafafa; }
.error-card { background: #fee; }
.error-card .card-header { background: #c0392b; }
details { margin: 4px 0; }
details summary { cursor: pointer; padding: 6px 0; font-weight: 500; color: #3498db; }
pre { background: #f8f9fa; padding: 12px; border-radius: 4px; overflow-x: auto; font-size: 11px; max-height: 300px; overflow-y: auto; white-space: pre-wrap; word-break: break-all; }
.legend { display: flex; flex-wrap: wrap; gap: 16px; margin: 10px 0; padding: 12px; background: white; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
.legend-item { display: flex; align-items: center; gap: 6px; font-size: 13px; }
.legend-box { width: 20px; height: 14px; border-radius: 2px; }
table.summary { border-collapse: collapse; width: 100%; margin: 16px 0; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
table.summary th { background: #2c3e50; color: white; padding: 10px 14px; text-align: left; font-size: 13px; }
table.summary td { padding: 8px 14px; border-bottom: 1px solid #eee; font-size: 13px; }
table.summary tr:hover { background: #f0f8ff; }
table.summary .num { text-align: right; font-variant-numeric: tabular-nums; }
.best { background: #e8f5e9 !important; font-weight: bold; }
.worst { background: #fce4ec !important; }
.page-sep { margin: 12px 0; padding: 6px 12px; background: #ecf0f1; border-radius: 4px; font-size: 12px; color: #666; }
"""


def generate_report(
    fixtures_dir: Path,
    host: str,
    output_dir: Path,
    providers: list[str],
    timeout: float = 300.0,
) -> Path:
    """Generate full HTML report with overlays, stats, text, and raw JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    fixtures = sorted(
        f for f in fixtures_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not fixtures:
        print(f"No fixtures found in {fixtures_dir}")
        return output_dir / "visual-report.html"

    all_stats: list[dict] = []
    sections_html: list[str] = []

    with httpx.Client(timeout=timeout) as client:
        for fixture in fixtures:
            print(f"\n  {fixture.name}:")
            is_pdf = fixture.suffix.lower() == ".pdf"
            file_bytes = fixture.read_bytes()

            # Get source images for overlay
            if is_pdf:
                source_images = pdf_to_images(fixture, dpi=150)
                if not source_images:
                    print("    WARNING: Could not render PDF pages (pdf2image/poppler missing)")
                    source_images = []
            else:
                source_images = [Image.open(fixture)]

            sections_html.append(f"<h2>{html.escape(fixture.name)}</h2>")
            if source_images:
                sections_html.append(f"<p>{len(source_images)} page(s), source: {fixture.suffix}</p>")
            sections_html.append("<div class='grid'>")

            for provider in providers:
                pdf_retries = 3 if is_pdf else 1
                ocr_data = fetch_ocr_result(client, host, provider, file_bytes, fixture.name, timeout, retries=pdf_retries)

                if ocr_data is None:
                    sections_html.append(
                        "<div class='card error-card'>"
                        f"<div class='card-header'><span class='provider'>{provider}</span>"
                        "<span class='badge'>ERROR</span></div>"
                        "<div class='card-footer'>Service returned an error or timed out</div></div>"
                    )
                    continue

                stats = _extract_stats(ocr_data)
                full_text = _extract_text(ocr_data)

                # Save raw JSON
                raw_filename = f"{fixture.stem}_{provider}.json"
                raw_path = raw_dir / raw_filename
                try:
                    raw_path.write_text(json.dumps(ocr_data, indent=2, ensure_ascii=False, default=str))
                except Exception:
                    raw_path.write_text(json.dumps({"error": "Could not serialize"}, indent=2))

                # Draw overlays per page
                overlay_filenames: list[str] = []
                pages = ocr_data.get("pages", [])
                for page_idx, page_data in enumerate(pages):
                    if page_idx < len(source_images):
                        src_img = source_images[page_idx]
                    elif source_images:
                        src_img = source_images[0]
                    else:
                        # Create blank image from page dimensions
                        w = int(page_data.get("width", 800))
                        h = int(page_data.get("height", 1100))
                        src_img = Image.new("RGB", (w, h), "white")

                    overlay_img = draw_overlay(src_img, page_data)
                    fname = f"{fixture.stem}_{provider}_p{page_idx}.png"
                    overlay_img.save(images_dir / fname, quality=85)
                    overlay_filenames.append(fname)

                all_stats.append({
                    "fixture": fixture.name,
                    "provider": provider,
                    **stats,
                })

                print(f"    {provider}: {stats['word_count']} words, {stats['latency_s']}s, conf={stats['avg_confidence']}")

                # Build card HTML
                card = [
                    "<div class='card'>",
                    f"<div class='card-header'><span class='provider'>{provider}</span>"
                    f"<span class='badge'>{stats['latency_s']}s</span></div>",
                    "<div class='card-stats'>",
                    f"<div class='stat'><div class='stat-value'>{stats['word_count']}</div><div class='stat-label'>Words</div></div>",
                    f"<div class='stat'><div class='stat-value'>{stats['block_count']}</div><div class='stat-label'>Blocks</div></div>",
                    f"<div class='stat'><div class='stat-value'>{stats['latency_s']}s</div><div class='stat-label'>Latency</div></div>",
                    f"<div class='stat'><div class='stat-value'>{stats['avg_confidence']:.2f}</div><div class='stat-label'>Confidence</div></div>",
                    "</div>",
                    "<div class='card-body'>",
                ]

                for i, fname in enumerate(overlay_filenames):
                    if len(overlay_filenames) > 1:
                        card.append(f"<div class='page-sep'>Page {i + 1}</div>")
                    card.append(f"<img src='images/{fname}' alt='{provider} page {i}' onclick='this.classList.toggle(\"zoomed\")'>")

                card.append("</div>")  # card-body

                card.append("<div class='card-footer'>")
                card.append(f"<details><summary>Extracted text ({stats['word_count']} words)</summary>"
                            f"<pre>{html.escape(full_text[:5000])}</pre></details>")
                card.append(f"<details><summary>Raw JSON response</summary>"
                            f"<pre>{html.escape(json.dumps(ocr_data, indent=2, default=str)[:10000])}</pre></details>")
                card.append("</div>")  # card-footer
                card.append("</div>")  # card

                sections_html.append("\n".join(card))

            sections_html.append("</div>")  # grid

    # Build summary tables
    summary_html = _build_summary_html(all_stats, providers)

    # Assemble full HTML
    report_html = "\n".join([
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>OCR Comparison Report</title>",
        f"<style>{CSS}</style></head><body>",
        "<h1>OCR Comparison Report</h1>",
        f"<p class='subtitle'>Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} | "
        f"{len(fixtures)} fixtures | {len(providers)} providers</p>",
        "<div class='legend'>",
        "<span class='legend-item'><span class='legend-box' style='background:rgba(0,180,0,0.4);border:1px solid #090'></span> High conf (&gt;0.9)</span>",
        "<span class='legend-item'><span class='legend-box' style='background:rgba(255,180,0,0.4);border:1px solid #b80'></span> Medium (0.7-0.9)</span>",
        "<span class='legend-item'><span class='legend-box' style='background:rgba(255,40,40,0.4);border:1px solid #c00'></span> Low (&lt;0.7)</span>",
        "<span class='legend-item'><span class='legend-box' style='background:rgba(0,200,80,0.2);border:1px solid #0c8050'></span> Line</span>",
        "<span class='legend-item'><span class='legend-box' style='background:rgba(220,40,40,0.15);border:1px solid #dc2828'></span> Block</span>",
        "</div>",
        summary_html,
        "\n".join(sections_html),
        "</body></html>",
    ])

    report_path = output_dir / "visual-report.html"
    report_path.write_text(report_html)
    print(f"\nReport: {report_path}")
    return report_path


def _build_summary_html(all_stats: list[dict], providers: list[str]) -> str:
    """Build pivot summary tables: latency, words, confidence — with best/worst highlighting."""
    fixtures = sorted(set(s["fixture"] for s in all_stats))
    by_fixture_provider = {(s["fixture"], s["provider"]): s for s in all_stats}

    def _pivot_table(title: str, metric: str, fmt: str = "d", lower_is_better: bool = True) -> str:
        rows = []
        for fixture in fixtures:
            vals = {}
            for p in providers:
                key = (fixture, p)
                if key in by_fixture_provider:
                    vals[p] = by_fixture_provider[key][metric]
            if not vals:
                continue
            numeric_vals = [v for v in vals.values() if isinstance(v, (int, float)) and v > 0]
            best_val = min(numeric_vals) if lower_is_better and numeric_vals else (max(numeric_vals) if numeric_vals else None)
            worst_val = max(numeric_vals) if lower_is_better and numeric_vals else (min(numeric_vals) if numeric_vals else None)

            cells = [f"<td>{fixture}</td>"]
            for p in providers:
                v = vals.get(p)
                if v is None:
                    cells.append("<td class='num'>-</td>")
                else:
                    css = ""
                    if v == best_val and len(numeric_vals) > 1:
                        css = " class='num best'"
                    elif v == worst_val and len(numeric_vals) > 1:
                        css = " class='num worst'"
                    else:
                        css = " class='num'"
                    cells.append(f"<td{css}>{v:{fmt}}</td>")
            rows.append("<tr>" + "".join(cells) + "</tr>")

        # Aggregate row
        agg_cells = ["<td><b>Average</b></td>"]
        for p in providers:
            vals = [by_fixture_provider[(f, p)][metric] for f in fixtures if (f, p) in by_fixture_provider]
            numeric = [v for v in vals if isinstance(v, (int, float)) and v > 0]
            if numeric:
                avg = statistics.mean(numeric)
                agg_cells.append(f"<td class='num'><b>{avg:{fmt}}</b></td>")
            else:
                agg_cells.append("<td class='num'>-</td>")
        rows.append("<tr style='border-top:2px solid #2c3e50'>" + "".join(agg_cells) + "</tr>")

        header = "<tr><th>Fixture</th>" + "".join(f"<th>{p}</th>" for p in providers) + "</tr>"
        return f"<h3>{title}</h3><table class='summary'>{header}{''.join(rows)}</table>"

    parts = [
        "<h2>Summary</h2>",
        _pivot_table("Latency (seconds)", "latency_s", ".1f", lower_is_better=True),
        _pivot_table("Word Count", "word_count", ".0f", lower_is_better=False),
        _pivot_table("Avg Confidence", "avg_confidence", ".3f", lower_is_better=False),
    ]
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visual OCR comparison report")
    parser.add_argument("--fixtures-dir", type=Path, default=Path("test-fixtures/"))
    parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark/results/visual"))
    parser.add_argument("--providers", type=str, default="paddle,tesseract,surya,doctr")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    providers = [p.strip() for p in args.providers.split(",")]
    generate_report(args.fixtures_dir, args.host, args.output_dir, providers, args.timeout)


if __name__ == "__main__":
    main()
