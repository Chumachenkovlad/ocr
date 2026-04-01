"""
Generate a static GitHub Pages site from benchmark results.

Output structure (written to docs/ at repo root):
  docs/index.html          — landing page with summary cards + links
  docs/report.html         — full analysis report (converted from markdown)
  docs/resource-report.html — resource / throughput HTML (copied)
  docs/saturation-report.html — saturation HTML (copied)
  docs/data/latest.json    — latest benchmark JSON (trimmed, no full_text)

Usage:
  python3 benchmark/generate_site.py
  python3 benchmark/generate_site.py --results benchmark/results/2026-03-31_071432.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "benchmark" / "results"
DOCS_DIR = REPO_ROOT / "docs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def latest_json() -> Path | None:
    files = sorted(
        p for p in RESULTS_DIR.glob("*.json")
        if not p.name.endswith("_resources.json")
        and not p.name.endswith("_saturation.json")
    )
    return files[-1] if files else None


def md_to_html(md: str) -> str:
    """Minimal markdown → HTML (tables, headings, bold, lists, code, hr)."""
    lines = md.split("\n")
    html_lines: list[str] = []
    in_table = False
    in_ul = False

    for line in lines:
        # Headings
        m = re.match(r"^(#{1,3})\s+(.*)", line)
        if m:
            if in_table: html_lines.append("</table>"); in_table = False
            if in_ul: html_lines.append("</ul>"); in_ul = False
            level = len(m.group(1))
            text = _inline(m.group(2))
            html_lines.append(f"<h{level}>{text}</h{level}>")
            continue

        # HR
        if re.match(r"^---+$", line.strip()):
            if in_table: html_lines.append("</table>"); in_table = False
            if in_ul: html_lines.append("</ul>"); in_ul = False
            html_lines.append("<hr>")
            continue

        # Table row
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if re.match(r"^\s*[-:]+\s*$", cells[0]):
                # separator row — skip
                continue
            if not in_table:
                html_lines.append('<table>')
                in_table = True
            tag = "th" if not any("<td>" in l for l in html_lines[-5:]) and not in_table else "td"
            # Always use td; first row after table open is th
            row_html = "".join(f"<td>{_inline(c)}</td>" for c in cells)
            html_lines.append(f"<tr>{row_html}</tr>")
            continue

        if in_table:
            html_lines.append("</table>")
            in_table = False

        # List item
        if re.match(r"^[-*]\s+", line):
            if not in_ul:
                html_lines.append("<ul>")
                in_ul = True
            html_lines.append(f"<li>{_inline(line[2:].strip())}</li>")
            continue

        if in_ul and line.strip():
            if not re.match(r"^[-*]\s+", line):
                html_lines.append("</ul>")
                in_ul = False

        if in_ul and not line.strip():
            html_lines.append("</ul>")
            in_ul = False
            html_lines.append("")
            continue

        # Blank line
        if not line.strip():
            html_lines.append("<br>")
            continue

        # Normal paragraph / blockquote
        if line.startswith("> "):
            html_lines.append(f"<blockquote>{_inline(line[2:])}</blockquote>")
        else:
            html_lines.append(f"<p>{_inline(line)}</p>")

    if in_table: html_lines.append("</table>")
    if in_ul: html_lines.append("</ul>")

    return "\n".join(html_lines)


def _inline(text: str) -> str:
    """Process inline markdown: bold, italic, code, links."""
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    return text


# ---------------------------------------------------------------------------
# Build summary data from benchmark JSON
# ---------------------------------------------------------------------------

def build_summary(data: dict) -> dict:
    results = [r for r in data["results"] if r["status"] == "ok"]
    providers = sorted(set(r["provider"] for r in results))

    summary: dict[str, dict] = {}
    for p in providers:
        pr = [r for r in results if r["provider"] == p]
        times = [r["wall_ms"] for r in pr]
        words = [r.get("word_count", 0) for r in pr]
        confs = [r.get("avg_confidence", 0) for r in pr if r.get("avg_confidence", 0) > 0]
        summary[p] = {
            "p50_ms": sorted(times)[len(times) // 2] if times else 0,
            "mean_ms": statistics.mean(times) if times else 0,
            "mean_words": statistics.mean(words) if words else 0,
            "mean_conf": statistics.mean(confs) if confs else 0,
            "fixtures": len(set(r["fixture"] for r in pr)),
            "errors": sum(1 for r in data["results"] if r["provider"] == p and r["status"] == "error"),
        }

    return {"providers": providers, "stats": summary, "timestamp": data["timestamp"]}


# ---------------------------------------------------------------------------
# HTML templates
# ---------------------------------------------------------------------------

PAGE_CSS = """
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
  .topbar nav a { color: #ccc; font-size: 13px; }
  .topbar nav a:hover { color: #fff; }

  .hero { background: var(--mid); color: #eee; padding: 40px 32px 32px; }
  .hero h2 { font-size: 28px; font-weight: 700; margin-bottom: 8px; }
  .hero p { color: #aaa; font-size: 14px; max-width: 600px; margin-bottom: 20px; }
  .hero .ts { font-size: 12px; color: #666; }

  .container { max-width: 1100px; margin: 0 auto; padding: 28px 32px; }

  /* Cards */
  .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 16px;
           margin-bottom: 28px; }
  .card { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 16px 18px; }
  .card-label { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.5px;
                margin-bottom: 4px; }
  .card-value { font-size: 26px; font-weight: 700; color: var(--dark); }
  .card-sub { font-size: 12px; color: #888; margin-top: 4px; }
  .card-winner .card-value { color: var(--accent); text-transform: capitalize; }

  /* Metrics table */
  .metrics-table { width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 24px; }
  .metrics-table th { background: var(--dark); color: #eee; padding: 9px 14px; text-align: left; font-weight: 600; }
  .metrics-table td { padding: 8px 14px; border-bottom: 1px solid #eee; }
  .metrics-table tr:hover td { background: #fafafa; }
  .best { color: #155724; font-weight: 700; }
  .worst { color: #721c24; }

  /* Bar */
  .bar-wrap { display: flex; align-items: center; gap: 8px; }
  .bar-track { flex: 1; height: 8px; background: #eee; border-radius: 4px; min-width: 80px; }
  .bar-fill { height: 100%; border-radius: 4px; }
  .bar-teal { background: var(--teal); }
  .bar-red { background: var(--accent); }
  .bar-val { font-size: 11px; color: #666; white-space: nowrap; }

  /* Links section */
  .links-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px;
                margin-bottom: 32px; }
  .link-card { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 16px 18px;
               display: flex; flex-direction: column; gap: 6px; }
  .link-card h3 { font-size: 14px; font-weight: 700; color: var(--dark); }
  .link-card p { font-size: 12px; color: #666; }
  .link-card a.btn { display: inline-block; margin-top: 6px; padding: 6px 14px;
                     background: var(--accent); color: #fff; border-radius: 4px;
                     font-size: 12px; font-weight: 600; width: fit-content; }

  /* Report page */
  .report-body { background: #fff; border: 1px solid #ddd; border-radius: 8px;
                 padding: 28px 32px; max-width: 900px; }
  .report-body h1 { font-size: 22px; font-weight: 700; margin-bottom: 16px; color: var(--dark); }
  .report-body h2 { font-size: 16px; font-weight: 700; color: var(--dark); margin: 24px 0 10px;
                    padding-bottom: 6px; border-bottom: 2px solid var(--accent); }
  .report-body h3 { font-size: 14px; font-weight: 700; color: #444; margin: 18px 0 8px; }
  .report-body p { font-size: 13px; line-height: 1.7; color: #444; margin-bottom: 8px; }
  .report-body table { width: 100%; border-collapse: collapse; font-size: 12px; margin: 12px 0; }
  .report-body table th { background: #f0f4ff; padding: 7px 12px; text-align: left;
                          font-weight: 600; color: #3949ab; border-bottom: 2px solid #c5cae9; }
  .report-body table td { padding: 6px 12px; border-bottom: 1px solid #eee; }
  .report-body table tr:hover td { background: #fafafa; }
  .report-body ul { padding-left: 20px; margin: 8px 0; font-size: 13px; }
  .report-body li { margin-bottom: 4px; }
  .report-body code { background: #f0f0f0; padding: 1px 5px; border-radius: 3px;
                      font-size: 11px; font-family: monospace; }
  .report-body blockquote { border-left: 3px solid var(--teal); padding: 8px 14px;
                             background: #f9f9f9; color: #555; font-size: 12px; margin: 12px 0; }
  .report-body hr { border: none; border-top: 1px solid #eee; margin: 20px 0; }
  .report-body strong { font-weight: 700; }
  .report-body em { font-style: italic; }
"""


def make_topbar(active: str) -> str:
    pages = [
        ("index.html", "Overview"),
        ("report.html", "Full Report"),
        ("resource-report.html", "Resources"),
        ("saturation-report.html", "Saturation"),
    ]
    links = " ".join(
        f'<a href="{href}" {"style=color:#fff;font-weight:700" if name == active else ""}>{name}</a>'
        for href, name in pages
    )
    return f"""
    <div class="topbar">
      <h1>OCR Comparison</h1>
      <span class="badge">Benchmark Results</span>
      <nav>{links}</nav>
    </div>"""


def build_index_html(summary: dict, report_md: str | None) -> str:
    stats = summary["stats"]
    providers = summary["providers"]

    # Winners
    if stats:
        fastest = min(providers, key=lambda p: stats[p]["mean_ms"])
        most_words = max(providers, key=lambda p: stats[p]["mean_words"])
        highest_conf = max(providers, key=lambda p: stats[p]["mean_conf"])
    else:
        fastest = most_words = highest_conf = "—"

    # Cards
    cards = f"""
    <div class="cards">
      <div class="card card-winner">
        <div class="card-label">Fastest (mean latency)</div>
        <div class="card-value">{fastest}</div>
        <div class="card-sub">{stats.get(fastest, {}).get('mean_ms', 0):.0f} ms average</div>
      </div>
      <div class="card card-winner">
        <div class="card-label">Most words extracted</div>
        <div class="card-value">{most_words}</div>
        <div class="card-sub">{stats.get(most_words, {}).get('mean_words', 0):.0f} words average</div>
      </div>
      <div class="card card-winner">
        <div class="card-label">Highest confidence</div>
        <div class="card-value">{highest_conf}</div>
        <div class="card-sub">{stats.get(highest_conf, {}).get('mean_conf', 0)*100:.1f}% average</div>
      </div>
      <div class="card">
        <div class="card-label">Providers tested</div>
        <div class="card-value">{len(providers)}</div>
        <div class="card-sub">{", ".join(providers)}</div>
      </div>
    </div>"""

    # Metrics table
    max_ms = max((s["mean_ms"] for s in stats.values()), default=1)
    max_words = max((s["mean_words"] for s in stats.values()), default=1)
    rows = ""
    sorted_providers = sorted(providers, key=lambda p: stats[p]["mean_ms"])
    for i, p in enumerate(sorted_providers):
        s = stats[p]
        lat_cls = "best" if i == 0 else ("worst" if i == len(sorted_providers) - 1 else "")
        rows += f"""
        <tr>
          <td><strong style="text-transform:capitalize">{p}</strong></td>
          <td class="{lat_cls}">
            <div class="bar-wrap">
              <div class="bar-track"><div class="bar-fill bar-red" style="width:{s['mean_ms']/max_ms*100:.1f}%"></div></div>
              <span class="bar-val">{s['mean_ms']:.0f} ms</span>
            </div>
          </td>
          <td>
            <div class="bar-wrap">
              <div class="bar-track"><div class="bar-fill bar-teal" style="width:{s['mean_words']/max_words*100:.1f}%"></div></div>
              <span class="bar-val">{s['mean_words']:.0f}</span>
            </div>
          </td>
          <td>{s['mean_conf']*100:.1f}%</td>
          <td>{'<span style="color:#e94560">'+str(s['errors'])+'</span>' if s['errors'] else '—'}</td>
        </tr>"""

    table = f"""
    <h2 style="font-size:16px;font-weight:700;color:#1a1a2e;margin:0 0 12px;padding-bottom:6px;border-bottom:2px solid #e94560">Provider summary</h2>
    <table class="metrics-table">
      <thead><tr><th>Provider</th><th>Mean latency</th><th>Mean words</th><th>Avg confidence</th><th>Errors</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>"""

    links = """
    <h2 style="font-size:16px;font-weight:700;color:#1a1a2e;margin:0 0 12px;padding-bottom:6px;border-bottom:2px solid #e94560">Reports</h2>
    <div class="links-grid">
      <div class="link-card">
        <h3>Full Analysis Report</h3>
        <p>Latency, word counts, confidence distributions, text similarity matrix, per-doc-type breakdown, and conclusion.</p>
        <a class="btn" href="report.html">View report →</a>
      </div>
      <div class="link-card">
        <h3>Resource Consumption</h3>
        <p>CPU, memory, and GPU utilization per service at steady-state load. Instance sizing recommendations.</p>
        <a class="btn" href="resource-report.html">View report →</a>
      </div>
      <div class="link-card">
        <h3>Saturation Benchmark</h3>
        <p>RPS vs concurrency sweep (1–16). Throughput curves, GPU utilization under load, queue saturation points.</p>
        <a class="btn" href="saturation-report.html">View report →</a>
      </div>
    </div>"""

    ts = summary["timestamp"].replace("_", " ").replace("-", "-")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OCR Benchmark — Results</title>
  <style>{PAGE_CSS}</style>
</head>
<body>
  {make_topbar("Overview")}
  <div class="hero">
    <h2>OCR Engine Benchmark Results</h2>
    <p>Side-by-side comparison of PaddleOCR, Tesseract, Surya, and docTR across 10 document types.</p>
    <div class="ts">Last run: {ts} UTC</div>
  </div>
  <div class="container">
    {cards}
    {table}
    {links}
  </div>
</body>
</html>"""


def build_report_html(report_md: str) -> str:
    body = md_to_html(report_md)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OCR Benchmark — Full Report</title>
  <style>{PAGE_CSS}</style>
</head>
<body>
  {make_topbar("Full Report")}
  <div class="container">
    <div class="report-body">
      {body}
    </div>
  </div>
</body>
</html>"""


def wrap_existing_html(html_path: Path, active: str) -> str:
    """Inject topbar into an existing standalone HTML report."""
    src = html_path.read_text()
    # Inject topbar after <body>
    topbar = make_topbar(active)
    injected = re.sub(r"(<body[^>]*>)", r"\1" + topbar, src, count=1)
    return injected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(results_file: Path | None = None, docs_dir: Path | None = None) -> None:
    out = docs_dir or DOCS_DIR
    out.mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(exist_ok=True)

    # Load benchmark JSON
    results_file = results_file or latest_json()
    if not results_file:
        print("No benchmark JSON found. Run benchmarks first.")
        return

    print(f"Loading results from {results_file}")
    data = json.loads(results_file.read_text())
    summary = build_summary(data)

    # Save trimmed JSON for site use
    slim = {k: v for k, v in data.items() if k != "results"}
    slim["results"] = [
        {k: v for k, v in r.items() if k != "full_text"}
        for r in data["results"]
    ]
    (out / "data" / "latest.json").write_text(json.dumps(slim, indent=2))

    # Generate/load analysis markdown report
    report_md_path = results_file.with_suffix(".md")
    if not report_md_path.exists():
        print("Running analyze.py to generate report...")
        import subprocess
        subprocess.run(
            ["python3", str(REPO_ROOT / "benchmark" / "analyze.py"), str(results_file)],
            check=True,
        )
    report_md = report_md_path.read_text() if report_md_path.exists() else None

    # Build index.html
    (out / "index.html").write_text(build_index_html(summary, report_md))
    print(f"  docs/index.html")

    # Build report.html
    if report_md:
        (out / "report.html").write_text(build_report_html(report_md))
        print(f"  docs/report.html")

    # Copy / wrap existing HTML reports
    for fname, label in [
        ("resource-report.html", "Resources"),
        ("saturation-report.html", "Saturation"),
    ]:
        src = RESULTS_DIR / fname
        dst = out / fname
        if src.exists():
            dst.write_text(wrap_existing_html(src, label))
            print(f"  docs/{fname}")
        else:
            print(f"  [skip] {fname} not found")

    print(f"\nSite generated at: {out}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate static GitHub Pages site from benchmark results")
    parser.add_argument("--results", type=Path, default=None, help="Benchmark JSON file")
    parser.add_argument("--docs-dir", type=Path, default=None, help="Output directory (default: docs/)")
    args = parser.parse_args()
    generate(args.results, args.docs_dir)


if __name__ == "__main__":
    main()
