"""Benchmark runner: sends test fixtures to all OCR services and collects results."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

DEFAULT_HOST = "localhost"
DEFAULT_PORTS = {
    "paddle": 8081,
    "tesseract": 8082,
    "surya": 8083,
    "doctr": 8084,
}


def build_service_urls(host: str) -> dict[str, str]:
    return {name: f"http://{host}:{port}" for name, port in DEFAULT_PORTS.items()}


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".pdf"}


def collect_fixtures(fixtures_dir: Path) -> list[Path]:
    files = sorted(
        f
        for f in fixtures_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"No test fixtures found in {fixtures_dir}")
    return files


def extract_page_stats(body: dict) -> dict:
    """Pull per-word, per-line, per-block counts and full text from a response body."""
    word_count = 0
    line_count = 0
    block_count = 0
    confidences: list[float] = []
    text_lines: list[str] = []

    for page in body.get("pages", []):
        for block in page.get("blocks", []):
            block_count += 1
            for line in block.get("lines", []):
                line_count += 1
                line_text = line.get("text", "")
                words_in_line: list[str] = []
                for word in line.get("words", []):
                    word_count += 1
                    conf = word.get("confidence", -1)
                    if conf >= 0:
                        confidences.append(conf)
                    words_in_line.append(word.get("text", ""))
                if not line_text and words_in_line:
                    line_text = " ".join(words_in_line)
                if line_text:
                    text_lines.append(line_text)

    avg_confidence = statistics.mean(confidences) if confidences else 0.0
    std_confidence = statistics.stdev(confidences) if len(confidences) >= 2 else 0.0

    sorted_conf = sorted(confidences)
    def _percentile(data: list[float], p: float) -> float:
        if not data:
            return 0.0
        idx = int(len(data) * p)
        return data[min(idx, len(data) - 1)]

    full_text = "\n".join(text_lines)

    return {
        "word_count": word_count,
        "line_count": line_count,
        "block_count": block_count,
        "avg_confidence": round(avg_confidence, 4),
        "std_confidence": round(std_confidence, 4),
        "p10_confidence": round(_percentile(sorted_conf, 0.10), 4),
        "p50_confidence": round(_percentile(sorted_conf, 0.50), 4),
        "p90_confidence": round(_percentile(sorted_conf, 0.90), 4),
        "full_text": full_text[:8000],  # cap to avoid huge JSON
    }


def send_to_service(
    client: httpx.Client,
    url: str,
    file_path: Path,
    dpi: int = 300,
    timeout: float = 120.0,
) -> dict:
    """Send a file to an OCR service and return timing + rich result metadata."""
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    t0 = time.monotonic()
    try:
        resp = client.post(
            f"{url}/api/v1/ocr",
            files={"file": (file_path.name, file_bytes)},
            data={"dpi": str(dpi)},
            timeout=timeout,
        )
        wall_ms = (time.monotonic() - t0) * 1000
        resp.raise_for_status()
        body = resp.json()

        stats = extract_page_stats(body)
        return {
            "status": "ok",
            "wall_ms": round(wall_ms, 1),
            "service_latency_ms": body.get("latency_ms", 0),
            "page_count": len(body.get("pages", [])),
            **stats,
        }
    except httpx.HTTPStatusError as e:
        wall_ms = (time.monotonic() - t0) * 1000
        return {"status": "error", "error": str(e), "wall_ms": round(wall_ms, 1)}
    except Exception as e:
        wall_ms = (time.monotonic() - t0) * 1000
        return {"status": "error", "error": str(e), "wall_ms": round(wall_ms, 1)}


def _text_similarity(a: str, b: str) -> float:
    """Word-level Jaccard similarity between two texts (0.0–1.0)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    intersection = len(words_a & words_b)
    union = len(words_a | words_b)
    return round(intersection / union, 4) if union else 0.0


def _sequence_similarity(a: str, b: str) -> float:
    """SequenceMatcher ratio — order-sensitive similarity (0.0–1.0)."""
    import difflib
    return round(difflib.SequenceMatcher(None, a[:3000], b[:3000]).ratio(), 4)


def compute_fixture_similarities(
    fixture_results: list[dict],
) -> dict[str, float]:
    """Compute pairwise text similarities for all providers on a single fixture."""
    provider_texts: dict[str, str] = {}
    for r in fixture_results:
        if r["status"] == "ok":
            provider_texts[r["provider"]] = r.get("full_text", "")

    providers = sorted(provider_texts)
    pairs: dict[str, float] = {}
    for i, p1 in enumerate(providers):
        for p2 in providers[i + 1:]:
            key = f"{p1}_vs_{p2}"
            pairs[key] = _sequence_similarity(provider_texts[p1], provider_texts[p2])
    return pairs


def run_benchmark(
    fixtures_dir: Path,
    services: dict[str, str],
    dpi: int = 300,
    runs: int = 1,
    dpi_sweep: list[int] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Run all fixtures against all services.

    Returns:
        (results, similarity_records)
    """
    dpis = dpi_sweep if dpi_sweep else [dpi]
    fixtures = collect_fixtures(fixtures_dir)
    results: list[dict] = []

    with httpx.Client() as client:
        for current_dpi in dpis:
            dpi_label = f" [DPI={current_dpi}]" if len(dpis) > 1 else ""
            for fixture in fixtures:
                print(f"\n--- {fixture.name}{dpi_label} ---")
                for name, url in services.items():
                    for run_idx in range(runs):
                        label = f"run {run_idx + 1}/{runs}" if runs > 1 else ""
                        print(f"  {name}{' ' + label if label else ''}...", end=" ", flush=True)
                        result = send_to_service(client, url, fixture, dpi=current_dpi)
                        result["fixture"] = fixture.name
                        result["provider"] = name
                        result["run"] = run_idx + 1
                        result["dpi"] = current_dpi
                        results.append(result)

                        if result["status"] == "ok":
                            print(
                                f"{result['wall_ms']:.0f}ms  "
                                f"words={result['word_count']} "
                                f"blocks={result['block_count']} "
                                f"lines={result['line_count']} "
                                f"conf={result['avg_confidence']:.3f}±{result['std_confidence']:.3f}"
                            )
                        else:
                            print(f"ERROR: {result.get('error', 'unknown')}")

    # Compute per-fixture pairwise text similarities
    similarity_records: list[dict] = []
    fixture_dpi_pairs = set((r["fixture"], r["dpi"]) for r in results)
    for fixture_name, current_dpi in sorted(fixture_dpi_pairs):
        fixture_results = [
            r for r in results
            if r["fixture"] == fixture_name and r["dpi"] == current_dpi and r["run"] == 1
        ]
        if len(fixture_results) >= 2:
            pairs = compute_fixture_similarities(fixture_results)
            similarity_records.append({
                "fixture": fixture_name,
                "dpi": current_dpi,
                **pairs,
            })

    return results, similarity_records


def main():
    parser = argparse.ArgumentParser(description="OCR Benchmark Runner")
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=Path("test-fixtures"),
    )
    parser.add_argument(
        "--services",
        type=str,
        default=None,
        help="Comma-separated list of services to test (default: all)",
    )
    parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--dpi-sweep",
        type=str,
        default=None,
        help="Comma-separated DPI values to sweep, e.g. 150,300,600",
    )
    parser.add_argument("--runs", type=int, default=1, help="Runs per fixture per service")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark/results"),
    )
    args = parser.parse_args()

    services = build_service_urls(args.host)
    if args.services:
        selected = {s.strip() for s in args.services.split(",")}
        services = {k: v for k, v in services.items() if k in selected}

    dpi_sweep = None
    if args.dpi_sweep:
        dpi_sweep = [int(d.strip()) for d in args.dpi_sweep.split(",")]

    print(f"Fixtures:  {args.fixtures_dir}")
    print(f"Services:  {', '.join(services)}")
    print(f"DPI:       {dpi_sweep or args.dpi}")
    print(f"Runs:      {args.runs}")

    results, similarities = run_benchmark(
        args.fixtures_dir, services,
        dpi=args.dpi, runs=args.runs, dpi_sweep=dpi_sweep,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    output_path = args.output_dir / f"{timestamp}.json"

    # Strip full_text from saved JSON to keep file size reasonable;
    # keep it for similarity (already computed above).
    results_slim = [{k: v for k, v in r.items() if k != "full_text"} for r in results]

    output = {
        "timestamp": timestamp,
        "dpi": args.dpi,
        "dpi_sweep": dpi_sweep,
        "runs_per_fixture": args.runs,
        "services": list(services.keys()),
        "results": results_slim,
        "similarities": similarities,
    }
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
