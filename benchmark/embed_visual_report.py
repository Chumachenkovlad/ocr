#!/usr/bin/env python3
"""
Embed visual-pdfs/visual-report.html images as base64 JPEG data URIs.
Produces a self-contained docs/visual-report.html suitable for GitHub Pages.
"""
import base64
import io
import os
import re
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Pillow required: pip install Pillow")
    sys.exit(1)

SRC_HTML = Path("benchmark/results/visual-pdfs/visual-report.html")
SRC_IMG_DIR = Path("benchmark/results/visual-pdfs/images")
OUT_HTML = Path("docs/visual-report.html")

# Resize to this max width (keeps aspect ratio); reduces file size dramatically
MAX_WIDTH = 1400
JPEG_QUALITY = 72


def compress_image(png_path: Path) -> str:
    """Return base64-encoded JPEG data URI for the given PNG."""
    img = Image.open(png_path).convert("RGB")
    if img.width > MAX_WIDTH:
        ratio = MAX_WIDTH / img.width
        new_h = int(img.height * ratio)
        img = img.resize((MAX_WIDTH, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def main():
    html = SRC_HTML.read_text(encoding="utf-8")

    # Find all img src references like images/foo.png
    refs = re.findall(r"src='(images/[^']+)'", html)
    refs = list(dict.fromkeys(refs))  # deduplicate, preserve order
    print(f"Found {len(refs)} unique image references")

    for i, ref in enumerate(refs):
        img_path = SRC_IMG_DIR / Path(ref).name
        if not img_path.exists():
            print(f"  MISSING: {img_path}")
            continue
        size_mb = img_path.stat().st_size / 1_048_576
        print(f"  [{i+1}/{len(refs)}] {img_path.name} ({size_mb:.1f} MB) ...", end="", flush=True)
        data_uri = compress_image(img_path)
        html = html.replace(f"src='{ref}'", f"src='{data_uri}'", 1)
        compressed_kb = len(data_uri) * 3 // 4 // 1024
        print(f" → {compressed_kb} KB")

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    out_mb = OUT_HTML.stat().st_size / 1_048_576
    print(f"\nWrote {OUT_HTML} ({out_mb:.1f} MB)")


if __name__ == "__main__":
    main()
