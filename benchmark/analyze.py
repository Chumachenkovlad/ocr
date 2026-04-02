"""Analyze benchmark results and generate a Markdown report."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


# ---------------------------------------------------------------------------
# Document type inference
# ---------------------------------------------------------------------------

_DOC_TYPE_KEYWORDS: list[tuple[str, str]] = [
    ("table", "table"),
    ("form", "form"),
    ("multi", "multipage"),
    ("noisy", "noisy-scan"),
    ("low-dpi", "low-dpi"),
    ("low_dpi", "low-dpi"),
    ("dense", "dense-layout"),
    ("mixed", "mixed-content"),
    ("multilingual", "multilingual"),
    ("simple", "simple-typed"),
    ("handwrite", "handwriting"),
    ("invoice", "invoice"),
    ("receipt", "receipt"),
    ("scan", "scanned"),
]


def doc_type(fixture_name: str) -> str:
    name = fixture_name.lower()
    for kw, label in _DOC_TYPE_KEYWORDS:
        if kw in name:
            return label
    return "general"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_results(path: Path) -> dict:
    return json.loads(path.read_text())


def _ok_results(data: dict) -> list[dict]:
    return [r for r in data["results"] if r["status"] == "ok"]


def _providers(data: dict) -> list[str]:
    return sorted(set(r["provider"] for r in data["results"]))


def _fixtures(data: dict) -> list[str]:
    return sorted(set(r["fixture"] for r in data["results"]))


def _subset(results: list[dict], fixture: str, provider: str) -> list[dict]:
    return [r for r in results if r["fixture"] == fixture and r["provider"] == provider]


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[len(s) // 2]


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(int(len(s) * p), len(s) - 1)]


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _section_latency(results: list[dict], providers: list[str], fixtures: list[str]) -> list[str]:
    lines = [
        "## Latency (ms)",
        "",
        f"| Fixture | {' | '.join(providers)} |",
        f"| --- | {' | '.join(['---'] * len(providers))} |",
    ]
    for fixture in fixtures:
        row = [fixture]
        for p in providers:
            times = [r["wall_ms"] for r in _subset(results, fixture, p)]
            row.append(f"{_median(times):.0f}" if times else "ERR")
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "### Aggregate latency",
        "",
        "| Provider | p50 | p95 | mean |",
        "| --- | --- | --- | --- |",
    ]
    for p in providers:
        times = [r["wall_ms"] for r in results if r["provider"] == p]
        if times:
            lines.append(
                f"| {p} | {_percentile(times, 0.50):.0f}"
                f" | {_percentile(times, 0.95):.0f}"
                f" | {statistics.mean(times):.0f} |"
            )
        else:
            lines.append(f"| {p} | - | - | - |")
    lines.append("")
    return lines


def _section_word_counts(results: list[dict], providers: list[str], fixtures: list[str]) -> list[str]:
    lines = [
        "## Word / line / block counts",
        "",
        f"| Fixture | {' | '.join(f'{p} (W/L/B)' for p in providers)} |",
        f"| --- | {' | '.join(['---'] * len(providers))} |",
    ]
    for fixture in fixtures:
        row = [fixture]
        for p in providers:
            rr = _subset(results, fixture, p)
            if rr:
                r = rr[0]
                row.append(f"{r.get('word_count',0)} / {r.get('line_count',0)} / {r.get('block_count',0)}")
            else:
                row.append("ERR")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def _section_confidence(results: list[dict], providers: list[str], fixtures: list[str]) -> list[str]:
    lines = [
        "## Confidence distribution",
        "",
        f"| Fixture | {' | '.join(providers)} |",
        f"| --- | {' | '.join(['---'] * len(providers))} |",
    ]
    for fixture in fixtures:
        row = [fixture]
        for p in providers:
            rr = _subset(results, fixture, p)
            if rr:
                r = rr[0]
                avg = r.get("avg_confidence", 0)
                std = r.get("std_confidence", 0)
                p10 = r.get("p10_confidence", 0)
                p90 = r.get("p90_confidence", 0)
                row.append(f"{avg:.3f} ±{std:.3f} [{p10:.2f}–{p90:.2f}]")
            else:
                row.append("ERR")
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "*Format: mean ±std [p10–p90]*",
        "",
    ]

    # Aggregate per provider
    lines += [
        "### Aggregate confidence",
        "",
        "| Provider | mean | p10 | p50 | p90 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for p in providers:
        avgs = [r["avg_confidence"] for r in results if r["provider"] == p and "avg_confidence" in r]
        if avgs:
            lines.append(
                f"| {p} | {statistics.mean(avgs):.3f}"
                f" | {_percentile(avgs, 0.10):.3f}"
                f" | {_percentile(avgs, 0.50):.3f}"
                f" | {_percentile(avgs, 0.90):.3f} |"
            )
        else:
            lines.append(f"| {p} | - | - | - | - |")
    lines.append("")
    return lines


def _section_accuracy(accuracy: list[dict], providers: list[str]) -> list[str]:
    """Render CER/WER accuracy table from majority-vote pseudo-ground-truth."""
    if not accuracy:
        return []

    lines = [
        "## Text accuracy (vs. majority-vote reference)",
        "",
        "*CER = Character Error Rate, WER = Word Error Rate. Lower is better.*",
        "*Reference text is the provider output most similar to all others (centroid).*",
        "",
    ]

    # Per-fixture table
    cer_keys = [f"{p}_cer" for p in providers]
    wer_keys = [f"{p}_wer" for p in providers]

    lines += [
        "### Character Error Rate (CER)",
        "",
        f"| Fixture | {' | '.join(providers)} |",
        f"| --- | {' | '.join(['---'] * len(providers))} |",
    ]
    for rec in sorted(accuracy, key=lambda x: x["fixture"]):
        row = [rec["fixture"]]
        for p in providers:
            val = rec.get(f"{p}_cer")
            row.append(f"{val:.3f}" if val is not None else "—")
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "### Word Error Rate (WER)",
        "",
        f"| Fixture | {' | '.join(providers)} |",
        f"| --- | {' | '.join(['---'] * len(providers))} |",
    ]
    for rec in sorted(accuracy, key=lambda x: x["fixture"]):
        row = [rec["fixture"]]
        for p in providers:
            val = rec.get(f"{p}_wer")
            row.append(f"{val:.3f}" if val is not None else "—")
        lines.append("| " + " | ".join(row) + " |")

    # Aggregate per provider
    lines += [
        "",
        "### Aggregate accuracy",
        "",
        "| Provider | Mean CER | Mean WER |",
        "| --- | --- | --- |",
    ]
    for p in providers:
        cers = [rec[f"{p}_cer"] for rec in accuracy if f"{p}_cer" in rec]
        wers = [rec[f"{p}_wer"] for rec in accuracy if f"{p}_wer" in rec]
        mean_cer = statistics.mean(cers) if cers else 0
        mean_wer = statistics.mean(wers) if wers else 0
        lines.append(f"| {p} | {mean_cer:.3f} | {mean_wer:.3f} |")
    lines.append("")
    return lines


def _section_similarity(similarities: list[dict], providers: list[str]) -> list[str]:
    if not similarities:
        return []

    # Collect all pair keys
    pair_keys: list[str] = []
    for s in similarities:
        for k in s:
            if k not in ("fixture", "dpi") and "_vs_" in k and k not in pair_keys:
                pair_keys.append(k)
    pair_keys.sort()

    if not pair_keys:
        return []

    lines = [
        "## Text similarity (SequenceMatcher, 0.0–1.0)",
        "",
        f"| Fixture | {' | '.join(pair_keys)} |",
        f"| --- | {' | '.join(['---'] * len(pair_keys))} |",
    ]
    for s in sorted(similarities, key=lambda x: x["fixture"]):
        row = [s["fixture"]]
        for pk in pair_keys:
            val = s.get(pk)
            row.append(f"{val:.3f}" if val is not None else "—")
        lines.append("| " + " | ".join(row) + " |")

    # Aggregate mean per pair
    lines += ["", "### Mean similarity", ""]
    for pk in pair_keys:
        vals = [s[pk] for s in similarities if pk in s]
        if vals:
            lines.append(f"- **{pk}**: {statistics.mean(vals):.3f}")
    lines.append("")
    return lines


def _section_doc_types(results: list[dict], providers: list[str]) -> list[str]:
    """Group fixtures by inferred document type and show per-type winner."""
    type_map: dict[str, list[str]] = {}
    for r in results:
        t = doc_type(r["fixture"])
        type_map.setdefault(t, [])
        if r["fixture"] not in type_map[t]:
            type_map[t].append(r["fixture"])

    if len(type_map) <= 1:
        return []

    lines = [
        "## Per document-type analysis",
        "",
        "| Type | Fixtures | Speed winner | Throughput winner | Confidence winner |",
        "| --- | --- | --- | --- | --- |",
    ]

    for dtype, fixtures in sorted(type_map.items()):
        type_results = [r for r in results if r["fixture"] in fixtures]

        # Speed: lowest mean latency
        speed: dict[str, float] = {}
        for p in providers:
            times = [r["wall_ms"] for r in type_results if r["provider"] == p]
            if times:
                speed[p] = statistics.mean(times)
        speed_winner = min(speed, key=speed.__getitem__) if speed else "—"

        # Throughput: highest mean word count
        words: dict[str, float] = {}
        for p in providers:
            counts = [r["word_count"] for r in type_results if r["provider"] == p and "word_count" in r]
            if counts:
                words[p] = statistics.mean(counts)
        words_winner = max(words, key=words.__getitem__) if words else "—"

        # Confidence: highest mean avg_confidence
        confs: dict[str, float] = {}
        for p in providers:
            vals = [r["avg_confidence"] for r in type_results if r["provider"] == p and "avg_confidence" in r]
            if vals:
                confs[p] = statistics.mean(vals)
        conf_winner = max(confs, key=confs.__getitem__) if confs else "—"

        fixture_names = ", ".join(f"`{f}`" for f in fixtures)
        lines.append(
            f"| {dtype} | {fixture_names} | {speed_winner} | {words_winner} | {conf_winner} |"
        )

    lines.append("")
    return lines


def _section_variance(results: list[dict], providers: list[str], fixtures: list[str]) -> list[str]:
    """Show run-to-run variance when --runs > 1."""
    multi_run = any(
        len([r for r in results if r["fixture"] == f and r["provider"] == p]) > 1
        for f in fixtures
        for p in providers
    )
    if not multi_run:
        return []

    lines = [
        "## Run-to-run variance (multi-run mode)",
        "",
        f"| Fixture | {' | '.join(providers)} |",
        f"| --- | {' | '.join(['---'] * len(providers))} |",
    ]
    for fixture in fixtures:
        row = [fixture]
        for p in providers:
            times = [r["wall_ms"] for r in _subset(results, fixture, p)]
            if len(times) >= 2:
                cv = statistics.stdev(times) / statistics.mean(times) if statistics.mean(times) else 0
                row.append(f"σ={statistics.stdev(times):.0f}ms CV={cv:.2%}")
            elif times:
                row.append("single run")
            else:
                row.append("ERR")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def _section_conclusion(
    results: list[dict],
    providers: list[str],
    similarities: list[dict],
) -> list[str]:
    """Produce an evidence-based conclusion section."""

    ok = [r for r in results if r["status"] == "ok"]
    if not ok:
        return []

    # Rankings per metric
    latency: dict[str, float] = {}
    words: dict[str, float] = {}
    confs: dict[str, float] = {}

    for p in providers:
        pr = [r for r in ok if r["provider"] == p]
        if not pr:
            continue
        latency[p] = statistics.mean(r["wall_ms"] for r in pr)
        words[p] = statistics.mean(r.get("word_count", 0) for r in pr)
        confs[p] = statistics.mean(r.get("avg_confidence", 0) for r in pr)

    speed_rank = sorted(latency, key=latency.__getitem__)
    words_rank = sorted(words, key=words.__getitem__, reverse=True)
    conf_rank = sorted(confs, key=confs.__getitem__, reverse=True)

    def rank_str(lst: list[str]) -> str:
        return " > ".join(f"**{p}** ({_metric_val(lst, p, latency, words, confs)})" for p in lst)

    def _metric_val(rank_list, p, lat, wds, cfs):
        if rank_list is speed_rank:
            return f"{lat.get(p, 0):.0f}ms"
        if rank_list is words_rank:
            return f"{wds.get(p, 0):.0f}w"
        return f"{cfs.get(p, 0):.3f}"

    # Agreement
    if similarities:
        all_pairs: dict[str, list[float]] = {}
        for s in similarities:
            for k, v in s.items():
                if "_vs_" in k:
                    all_pairs.setdefault(k, []).append(v)
        avg_pairs = {k: statistics.mean(v) for k, v in all_pairs.items()}
        most_similar = max(avg_pairs, key=avg_pairs.__getitem__) if avg_pairs else None
        least_similar = min(avg_pairs, key=avg_pairs.__getitem__) if avg_pairs else None
        agreement_text = (
            f"- **Most agreeing pair**: {most_similar} ({avg_pairs[most_similar]:.3f})\n"
            f"- **Least agreeing pair**: {least_similar} ({avg_pairs[least_similar]:.3f})"
            if most_similar else ""
        )
    else:
        agreement_text = ""

    # Build per-use-case recommendations based on rankings and known characteristics
    recs: list[tuple[str, str, str]] = []

    fastest = speed_rank[0] if speed_rank else "—"
    most_words = words_rank[0] if words_rank else "—"
    highest_conf = conf_rank[0] if conf_rank else "—"

    recs.append(("Real-time / latency-sensitive", fastest, f"lowest mean latency ({latency.get(fastest, 0):.0f}ms)"))
    recs.append(("Maximum text extraction (recall)", most_words, f"most words on average ({words.get(most_words, 0):.0f})"))
    recs.append(("Highest confidence / precision", highest_conf, f"best avg confidence ({confs.get(highest_conf, 0):.3f})"))

    # Table-heavy documents: paddle wins on table detection support; tesseract is weakest
    if "paddle" in providers:
        recs.append(("Table extraction", "paddle", "PaddleOCR has native table HTML output"))
    if "surya" in providers:
        recs.append(("Multi-language documents (90+ langs)", "surya", "Surya supports 90+ languages"))
    if "tesseract" in providers:
        recs.append(("Air-gap / CPU-only / no GPU infra", "tesseract", "pure CPU, Apache 2.0, zero model downloads at runtime"))
    if "doctr" in providers:
        recs.append(("Word-level bounding boxes", "doctr", "docTR provides per-word bbox + confidence"))

    lines = [
        "## Conclusion",
        "",
        "### Observed rankings",
        "",
        f"| Metric | Ranking |",
        "| --- | --- |",
        f"| Speed | {' > '.join(f'{p} ({latency.get(p,0):.0f}ms)' for p in speed_rank)} |",
        f"| Words extracted | {' > '.join(f'{p} ({words.get(p,0):.0f})' for p in words_rank)} |",
        f"| Avg confidence | {' > '.join(f'{p} ({confs.get(p,0):.3f})' for p in conf_rank)} |",
        "",
    ]

    if agreement_text:
        lines += [
            "### Text agreement between providers",
            "",
            agreement_text,
            "",
        ]

    lines += [
        "### Use-case recommendations",
        "",
        "| Use case | Recommended | Reason |",
        "| --- | --- | --- |",
    ]
    for use_case, rec, reason in recs:
        lines.append(f"| {use_case} | **{rec}** | {reason} |")

    lines += [
        "",
        "> Rankings derived from benchmark data. For document types not covered by fixtures, "
        "validate with representative samples before choosing a provider.",
        "",
    ]
    return lines


# ---------------------------------------------------------------------------
# Main analyze function
# ---------------------------------------------------------------------------

def analyze(data: dict) -> str:
    ok = _ok_results(data)
    providers = _providers(data)
    fixtures = _fixtures(data)
    similarities = data.get("similarities", [])
    accuracy = data.get("accuracy", [])

    lines: list[str] = [
        "# OCR Benchmark Report",
        "",
        f"- **Date**: {data['timestamp']}",
        f"- **DPI**: {data.get('dpi_sweep') or data['dpi']}",
        f"- **Runs per fixture**: {data['runs_per_fixture']}",
        f"- **Providers**: {', '.join(providers)}",
        f"- **Fixtures**: {len(fixtures)}",
        "",
    ]

    lines += _section_latency(ok, providers, fixtures)
    lines += _section_word_counts(ok, providers, fixtures)
    lines += _section_confidence(ok, providers, fixtures)
    lines += _section_accuracy(accuracy, providers)
    lines += _section_similarity(similarities, providers)
    lines += _section_variance(ok, providers, fixtures)
    lines += _section_doc_types(ok, providers)
    lines += _section_conclusion(ok, providers, similarities)

    errors = [r for r in data["results"] if r["status"] == "error"]
    if errors:
        lines += [
            "## Errors",
            "",
        ]
        for err in errors:
            lines.append(f"- **{err['provider']}** on `{err['fixture']}`: {err.get('error', 'unknown')}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze OCR benchmark results")
    parser.add_argument("results_file", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    data = load_results(args.results_file)
    report = analyze(data)

    output_path = args.output or args.results_file.with_suffix(".md")
    output_path.write_text(report)
    print(f"Report written to {output_path}")
    print(report)


if __name__ == "__main__":
    main()
