"""Extract and analyze embedded text layers from PDF fixtures.

Compares embedded OCR text layers against fresh OCR output to assess
whether existing text layers are usable for the editor pipeline.

Usage:
    python3 benchmark/text_layer.py --pdfs-dir test-fixtures/pdfs/
    python3 benchmark/text_layer.py --pdfs-dir test-fixtures/pdfs/ --raw-dir benchmark/results/visual/raw/
"""

from __future__ import annotations

import argparse
import difflib
import json
import statistics
from pathlib import Path


def extract_text_layer(pdf_path: Path) -> dict:
    """Extract embedded text layer from a PDF using PyMuPDF.

    Returns:
        Dict with per-page text and metadata.
    """
    import pymupdf

    doc = pymupdf.open(str(pdf_path))
    pages: list[dict] = []
    total_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        blocks = page.get_text("dict")["blocks"]

        text_blocks = [b for b in blocks if b["type"] == 0]  # text blocks only
        word_count = sum(
            len(line["spans"])
            for b in text_blocks
            for line in b.get("lines", [])
        )

        pages.append({
            "page_number": page_num,
            "text": text,
            "word_count": word_count,
            "block_count": len(text_blocks),
            "has_text": bool(text.strip()),
        })
        total_text.append(text)

    doc.close()

    full_text = "\n".join(total_text)
    return {
        "file": pdf_path.name,
        "page_count": len(pages),
        "pages": pages,
        "full_text": full_text,
        "has_text_layer": any(p["has_text"] for p in pages),
        "total_chars": len(full_text.strip()),
        "total_words": len(full_text.split()),
    }


def _sequence_similarity(a: str, b: str) -> float:
    """SequenceMatcher ratio (0.0–1.0)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return round(
        difflib.SequenceMatcher(None, a[:5000], b[:5000]).ratio(), 4
    )


def _word_error_rate(hypothesis: str, reference: str) -> float:
    """Word Error Rate between hypothesis and reference."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    n, m = len(ref_words), len(hyp_words)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[j], prev = min(dp[j] + 1, dp[j - 1] + 1, prev + cost), dp[j]
    return round(dp[m] / max(len(ref_words), 1), 4)


def load_ocr_text(raw_json_path: Path) -> str:
    """Extract full text from a saved OCR result JSON."""
    data = json.loads(raw_json_path.read_text())
    lines = []
    for page in data.get("pages", []):
        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                lines.append(line.get("text", ""))
    return "\n".join(lines)


def analyze_text_layers(
    pdfs_dir: Path,
    raw_dir: Path | None = None,
) -> list[dict]:
    """Analyze text layers in all PDFs and compare against OCR output."""
    pdf_files = sorted(
        f for f in pdfs_dir.iterdir()
        if f.suffix.lower() == ".pdf"
    )

    if not pdf_files:
        print(f"No PDF files found in {pdfs_dir}")
        return []

    results: list[dict] = []

    for pdf_path in pdf_files:
        print(f"\n--- {pdf_path.name} ---")
        layer = extract_text_layer(pdf_path)

        result: dict = {
            "file": pdf_path.name,
            "has_text_layer": layer["has_text_layer"],
            "layer_words": layer["total_words"],
            "layer_chars": layer["total_chars"],
            "page_count": layer["page_count"],
        }

        if layer["has_text_layer"]:
            print(f"  Text layer: {layer['total_words']} words, {layer['total_chars']} chars")
        else:
            print("  No embedded text layer")

        # Compare against OCR output if raw results available
        if raw_dir and raw_dir.exists() and layer["has_text_layer"]:
            providers = ["paddle", "tesseract", "surya", "doctr"]
            comparisons: dict[str, dict] = {}

            for provider in providers:
                raw_path = raw_dir / f"{pdf_path.stem}_{provider}.json"
                if not raw_path.exists():
                    continue

                ocr_text = load_ocr_text(raw_path)
                if not ocr_text.strip():
                    continue

                similarity = _sequence_similarity(layer["full_text"], ocr_text)
                wer = _word_error_rate(ocr_text, layer["full_text"])

                comparisons[provider] = {
                    "similarity": similarity,
                    "wer_vs_layer": wer,
                    "ocr_words": len(ocr_text.split()),
                }
                print(f"  vs {provider}: similarity={similarity:.3f}, WER={wer:.3f}, ocr_words={len(ocr_text.split())}")

            result["comparisons"] = comparisons

        results.append(result)

    return results


def generate_report(results: list[dict]) -> str:
    """Generate a Markdown report from text layer analysis."""
    lines = [
        "# PDF Text Layer Analysis",
        "",
        "Comparison of embedded OCR text layers against fresh OCR output.",
        "",
        "## Summary",
        "",
        "| PDF | Has Layer | Layer Words | Pages |",
        "| --- | --- | --- | --- |",
    ]

    for r in results:
        has = "Yes" if r["has_text_layer"] else "No"
        lines.append(
            f"| {r['file']} | {has} | {r['layer_words']} | {r['page_count']} |"
        )
    lines.append("")

    # Comparison table (only PDFs with text layers and OCR comparisons)
    compared = [r for r in results if r.get("comparisons")]
    if compared:
        providers = sorted(
            set(p for r in compared for p in r["comparisons"])
        )
        lines += [
            "## Text layer vs. fresh OCR",
            "",
            "*Similarity: SequenceMatcher ratio (1.0 = identical). WER: Word Error Rate (0.0 = perfect match).*",
            "",
            f"| PDF | {' | '.join(f'{p} (sim / WER)' for p in providers)} |",
            f"| --- | {' | '.join(['---'] * len(providers))} |",
        ]
        for r in compared:
            row = [r["file"]]
            for p in providers:
                c = r["comparisons"].get(p)
                if c:
                    row.append(f"{c['similarity']:.3f} / {c['wer_vs_layer']:.3f}")
                else:
                    row.append("—")
            lines.append("| " + " | ".join(row) + " |")

        # Aggregate
        lines += ["", "### Aggregate"]
        for p in providers:
            sims = [
                r["comparisons"][p]["similarity"]
                for r in compared if p in r["comparisons"]
            ]
            wers = [
                r["comparisons"][p]["wer_vs_layer"]
                for r in compared if p in r["comparisons"]
            ]
            if sims:
                lines.append(
                    f"- **{p}**: mean similarity={statistics.mean(sims):.3f}, "
                    f"mean WER={statistics.mean(wers):.3f}"
                )
        lines.append("")

        # Assessment
        lines += [
            "## Assessment",
            "",
            "**Are embedded text layers usable?**",
            "",
        ]
        all_sims = [
            c["similarity"]
            for r in compared
            for c in r["comparisons"].values()
        ]
        avg_sim = statistics.mean(all_sims) if all_sims else 0
        if avg_sim > 0.85:
            lines.append(
                "The embedded text layers show high agreement with fresh OCR output "
                f"(mean similarity {avg_sim:.3f}). They may be usable as-is for the "
                "editor pipeline, potentially saving an OCR pass."
            )
        elif avg_sim > 0.6:
            lines.append(
                "The embedded text layers show moderate agreement with fresh OCR "
                f"(mean similarity {avg_sim:.3f}). They could serve as a starting "
                "point but may need correction for precise text placement."
            )
        else:
            lines.append(
                "The embedded text layers show low agreement with fresh OCR "
                f"(mean similarity {avg_sim:.3f}). Fresh OCR is recommended "
                "for the editor pipeline."
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze embedded text layers in PDF fixtures"
    )
    parser.add_argument(
        "--pdfs-dir", type=Path, default=Path("test-fixtures/pdfs/"),
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=None,
        help="Directory with saved OCR result JSONs (e.g. benchmark/results/visual/raw/)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("benchmark/results/text-layer-report.md"),
    )
    args = parser.parse_args()

    results = analyze_text_layers(args.pdfs_dir, args.raw_dir)
    report = generate_report(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"\nReport written to {args.output}")
    print(report)


if __name__ == "__main__":
    main()
