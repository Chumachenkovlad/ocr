#!/usr/bin/env python3
"""Normalize all docs/*.html pages to a consistent design system.

- Shared CSS variables, reset, topbar, hero, container
- Consistent active-nav class
- Content wrapped in .container for max-width alignment
- Hero section with dark background for page title

Idempotent: safe to run multiple times.
"""
import re
from pathlib import Path

DOCS_DIR = Path(__file__).parent.parent / "docs"

# ── Shared CSS (single source of truth) ──────────────────────────────

SHARED_CSS = """\
:root { --accent: #e94560; --teal: #4ecca3; --dark: #1a1a2e; --mid: #16213e; }
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #f5f5f5; color: #333; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

.topbar { background: var(--dark); color: #eee; padding: 12px 32px;
          display: flex; align-items: center; gap: 16px; }
.topbar h1 { font-size: 18px; font-weight: 700; }
.topbar .badge { font-size: 11px; padding: 2px 8px; border-radius: 10px;
                 background: var(--teal); color: #111; font-weight: 600; }
.topbar nav { margin-left: auto; display: flex; gap: 20px; }
.topbar nav a { color: #ccc; font-size: 13px; text-decoration: none; }
.topbar nav a:hover, .topbar nav a.active { color: #fff; font-weight: 700; }

.hero { background: var(--mid); color: #eee; padding: 28px 32px 20px; }
.hero h2 { font-size: 22px; font-weight: 700; margin-bottom: 6px; }
.hero .subtitle { color: #aaa; font-size: 13px; margin: 0; }

.container { max-width: 1100px; margin: 0 auto; padding: 28px 32px; }
"""

# ── Per-page CSS (supplements shared, not overrides) ─────────────────

# Tables, cards, etc. that vary per report type
REPORT_TABLES_CSS = """\
table { border-collapse: collapse; width: 100%; margin: 16px 0;
        background: white; border-radius: 8px; overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
th { background: var(--dark); color: #eee; padding: 9px 14px;
     text-align: left; font-size: 13px; font-weight: 600; }
td { padding: 8px 14px; border-bottom: 1px solid #eee; font-size: 13px; }
tr:hover td { background: #fafafa; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.best { background: #e8f5e9 !important; font-weight: bold; }
.worst { background: #fce4ec !important; }
h2 { color: var(--dark); font-size: 16px; font-weight: 700;
     margin: 28px 0 12px; padding-bottom: 6px;
     border-bottom: 2px solid var(--accent); }
h3 { color: #444; font-size: 14px; margin: 18px 0 8px; }
"""

VISUAL_EXTRA_CSS = """\
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
        gap: 16px; margin: 16px 0; }
.card { background: white; border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1); overflow: hidden; }
.card-header { padding: 12px 16px; background: var(--dark); color: white;
               display: flex; justify-content: space-between; align-items: center; }
.card-header .provider { font-weight: bold; font-size: 16px; }
.card-header .badge { font-size: 11px; padding: 2px 8px; border-radius: 10px;
                      background: rgba(255,255,255,0.2); }
.card img { width: 100%; height: auto; display: block; cursor: zoom-in; }
.card img.zoomed { cursor: zoom-out; max-width: none; width: auto; }
.card-body { padding: 0; }
.card-stats { display: grid; grid-template-columns: repeat(4, 1fr);
              gap: 0; border-bottom: 1px solid #eee; }
.stat { padding: 8px 12px; text-align: center; border-right: 1px solid #eee; }
.stat:last-child { border-right: none; }
.stat-value { font-size: 18px; font-weight: bold; color: var(--dark); }
.stat-label { font-size: 10px; color: #888; text-transform: uppercase; }
.card-footer { padding: 8px 16px; font-size: 12px; color: #666; background: #fafafa; }
.error-card { background: #fee; }
.error-card .card-header { background: #c0392b; }
details { margin: 4px 0; }
details summary { cursor: pointer; padding: 6px 0; font-weight: 500; color: var(--accent); }
pre { background: #f8f9fa; padding: 12px; border-radius: 4px; overflow-x: auto;
      font-size: 11px; max-height: 300px; overflow-y: auto;
      white-space: pre-wrap; word-break: break-all; }
.legend { display: flex; flex-wrap: wrap; gap: 16px; margin: 10px 0; padding: 12px;
          background: white; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
.legend-item { display: flex; align-items: center; gap: 6px; font-size: 13px; }
.legend-box { width: 20px; height: 14px; border-radius: 2px; }
table.summary th { background: var(--dark); }
table.summary td { padding: 8px 14px; border-bottom: 1px solid #eee; font-size: 13px; }
table.summary tr:hover { background: #f0f8ff; }
table.summary .num { text-align: right; font-variant-numeric: tabular-nums; }
.page-sep { margin: 12px 0; padding: 6px 12px; background: #ecf0f1;
            border-radius: 4px; font-size: 12px; color: #666; }
"""

RESOURCE_EXTRA_CSS = """\
.card { background: white; border-radius: 8px; padding: 20px; margin: 16px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
.bar-row { display: flex; align-items: center; gap: 8px; margin: 6px 0; }
.bar-label { width: 120px; font-size: 13px; color: #555; }
.optimal { background: #27ae60; color: white; padding: 2px 8px;
           border-radius: 4px; font-size: 11px; font-weight: bold; }
"""

SATURATION_EXTRA_CSS = """\
.card { background: white; border-radius: 8px; padding: 20px; margin: 16px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
.bar { display: inline-block; height: 16px; border-radius: 3px; min-width: 2px; }
.optimal { background: #27ae60; color: white; padding: 2px 8px;
           border-radius: 4px; font-size: 11px; font-weight: bold; }
"""

# ── Nav ──────────────────────────────────────────────────────────────

NAV_LINKS = [
    ("index.html", "Overview"),
    ("report.html", "Full Report"),
    ("resource-report.html", "Resources"),
    ("saturation-report.html", "Saturation"),
    ("visual-report.html", "Visual"),
]


def build_nav(active_href: str) -> str:
    parts = []
    for href, label in NAV_LINKS:
        cls = ' class="active"' if href == active_href else ""
        parts.append(f'<a href="{href}"{cls}>{label}</a>')
    return "<nav>" + " ".join(parts) + "</nav>"


def build_topbar(active_href: str) -> str:
    return (
        '<div class="topbar">\n'
        '  <h1>OCR Comparison</h1>\n'
        '  <span class="badge">Benchmark Results</span>\n'
        f"  {build_nav(active_href)}\n"
        "</div>"
    )


# ── Page configs ─────────────────────────────────────────────────────

PAGES = {
    "resource-report.html": {
        "active": "resource-report.html",
        "title": "Resource Consumption Report",
        "subtitle_pattern": r"Instance:.*?</p>\s*<p class='subtitle'>Generated:.*?</p>",
        "extra_css": RESOURCE_EXTRA_CSS,
    },
    "saturation-report.html": {
        "active": "saturation-report.html",
        "title": "GPU Saturation Benchmark",
        "subtitle_pattern": r"Instance:.*?</p>",
        "extra_css": SATURATION_EXTRA_CSS,
    },
    "visual-report.html": {
        "active": "visual-report.html",
        "title": "Visual Accuracy Report",
        "subtitle_pattern": r"Generated.*?providers</p>",
        "extra_css": VISUAL_EXTRA_CSS,
    },
}


def extract_subtitle(html: str, pattern: str) -> str:
    """Extract subtitle text from the old h1-adjacent <p> tags."""
    m = re.search(pattern, html, re.DOTALL)
    if m:
        # Strip tags, keep text
        text = re.sub(r"<[^>]+>", " ", m.group(0)).strip()
        return text
    return ""


def normalize_page(filename: str, config: dict) -> None:
    path = DOCS_DIR / filename
    if not path.exists():
        print(f"SKIP: {filename} not found")
        return

    html = path.read_text(encoding="utf-8")

    # Extract subtitle before we rewrite
    subtitle = extract_subtitle(html, config["subtitle_pattern"])

    # ── Strip old <style> and everything up to and including old topbar ──
    # Find the content after the topbar closing div
    # Pattern: everything from <body> through the topbar, then the old <h1> + subtitles
    body_match = re.search(
        r"</div>\s*\n?"  # end of topbar
        r"(?:<h1>.*?</h1>\s*)?"  # old h1
        r"(?:<p\s+class='subtitle'>.*?</p>\s*)*",  # old subtitles
        html,
        re.DOTALL,
    )

    if not body_match:
        print(f"SKIP: {filename} — could not find content boundary")
        return

    # Everything after the old header is the actual content
    content_start = body_match.end()
    content_end = html.rfind("</body>")
    if content_end == -1:
        content_end = len(html)

    content = html[content_start:content_end].strip()

    # ── Build new page ──
    extra_css = REPORT_TABLES_CSS + config.get("extra_css", "")

    new_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OCR Benchmark — {config['title']}</title>
  <style>
{SHARED_CSS}
{extra_css}
  </style>
</head>
<body>
{build_topbar(config['active'])}
<div class="hero">
  <h2>{config['title']}</h2>
  <p class="subtitle">{subtitle}</p>
</div>
<div class="container">
{content}
</div>
</body>
</html>"""

    path.write_text(new_html, encoding="utf-8")
    print(f"NORMALIZED: {filename}")


def normalize_index_report() -> None:
    """Ensure index.html and report.html use class-based active nav (not inline style)."""
    for filename, active in [("index.html", "index.html"), ("report.html", "report.html")]:
        path = DOCS_DIR / filename
        if not path.exists():
            continue
        html = path.read_text(encoding="utf-8")
        original = html

        # Replace any nav block with the canonical one
        nav_html = build_nav(active)
        html = re.sub(r"<nav>.*?</nav>", nav_html, html, count=1, flags=re.DOTALL)

        # Ensure .active CSS rule exists
        if ".topbar nav a.active" not in html and "topbar nav a:hover" in html:
            html = html.replace(
                ".topbar nav a:hover { color: #fff; }",
                ".topbar nav a:hover, .topbar nav a.active { color: #fff; font-weight: 700; }",
            )

        if html != original:
            path.write_text(html, encoding="utf-8")
            print(f"FIXED NAV: {filename}")
        else:
            print(f"OK: {filename}")


def main() -> None:
    print("Normalizing HTML reports...\n")

    # Fix index.html and report.html nav
    normalize_index_report()

    # Normalize the 3 non-conforming pages
    for filename, config in PAGES.items():
        normalize_page(filename, config)

    print("\nDone.")


if __name__ == "__main__":
    main()
