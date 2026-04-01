#!/usr/bin/env python3
"""
Ensure consistent topbar nav across all docs/*.html pages.
- Adds missing topbar CSS to resource-report.html and saturation-report.html
- Adds missing Visual link to report.html, resource-report.html, saturation-report.html
- Standardises active link to use inline style (matches original generator output)
"""
import re
from pathlib import Path

TOPBAR_CSS = """\
  .topbar { background: var(--dark); color: #eee; padding: 12px 32px;
            display: flex; align-items: center; gap: 16px; }
  .topbar h1 { font-size: 18px; font-weight: 700; }
  .topbar .badge { font-size: 11px; padding: 2px 8px; border-radius: 10px;
                   background: var(--teal); color: #111; font-weight: 600; }
  .topbar nav { margin-left: auto; display: flex; gap: 20px; }
  .topbar nav a { color: #ccc; font-size: 13px; text-decoration: none; }
  .topbar nav a:hover, .topbar nav a.active { color: #fff; font-weight: 700; }"""

NAV_LINKS = [
    ("index.html",            "Overview"),
    ("report.html",           "Full Report"),
    ("resource-report.html",  "Resources"),
    ("saturation-report.html","Saturation"),
    ("visual-report.html",    "Visual"),
]

PAGES = {
    "docs/index.html":            "index.html",
    "docs/report.html":           "report.html",
    "docs/resource-report.html":  "resource-report.html",
    "docs/saturation-report.html":"saturation-report.html",
    "docs/visual-report.html":    "visual-report.html",
}


def build_nav(active_href: str) -> str:
    parts = []
    for href, label in NAV_LINKS:
        if href == active_href:
            parts.append(f'<a href="{href}" class="active">{label}</a>')
        else:
            parts.append(f'<a href="{href}">{label}</a>')
    return "<nav>" + " ".join(parts) + "</nav>"


def ensure_topbar_css(html: str) -> str:
    """Inject topbar CSS if missing."""
    if ".topbar" in html:
        return html
    # Insert before </style>
    return html.replace("</style>", TOPBAR_CSS + "\n</style>", 1)


def ensure_root_vars(html: str) -> str:
    """Ensure :root CSS vars are present for the topbar colours."""
    if "--dark" in html:
        return html
    root_vars = "  :root { --accent: #e94560; --teal: #4ecca3; --dark: #1a1a2e; }\n"
    # Insert after <style>
    html = re.sub(r"(<style[^>]*>)", r"\1\n" + root_vars, html, count=1)
    return html


def fix_nav_links(html: str, active_href: str) -> str:
    """Replace any <nav>...</nav> inside .topbar with the canonical nav."""
    nav_html = build_nav(active_href)
    # Match the nav element (non-greedy)
    return re.sub(r"<nav>.*?</nav>", nav_html, html, count=1, flags=re.DOTALL)


def remove_body_margin(html: str) -> str:
    """Remove `margin: 20px` from body rule (topbar needs full-width flush layout)."""
    return re.sub(r"(body\s*\{[^}]*?)margin:\s*20px;?\s*", r"\1", html, count=1)


for path, active_href in PAGES.items():
    p = Path(path)
    if not p.exists():
        print(f"SKIP (missing): {path}")
        continue

    original = p.read_text(encoding="utf-8")
    html = original

    html = ensure_root_vars(html)
    html = ensure_topbar_css(html)
    html = remove_body_margin(html)
    html = fix_nav_links(html, active_href)

    if html != original:
        p.write_text(html, encoding="utf-8")
        print(f"UPDATED: {path}")
    else:
        print(f"OK (no change): {path}")
