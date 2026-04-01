"""GPU saturation benchmark — find max RPS per service.

Sweeps concurrency from 1 to max, measuring RPS, GPU utilization,
and error rate to find the optimal operating point.

Usage:
    python3 benchmark/saturation_bench.py --host 3.84.61.60
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

SSH_KEY = Path(__file__).parent.parent / "infra" / "ocr-comparison.pem"
SSH_USER = "ec2-user"

DEFAULT_PORTS = {
    "paddle": 8081,
    "tesseract": 8082,
    "surya": 8083,
    "doctr": 8084,
}

INSTANCE_COSTS = {
    "g5.2xlarge": 1.212,
    "g4dn.xlarge": 0.526,
    "c5.2xlarge": 0.340,
}

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}


def ssh_cmd(host: str, cmd: str, timeout: int = 15) -> str:
    result = subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
         "-i", str(SSH_KEY), f"{SSH_USER}@{host}", cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# GPU monitor
# ---------------------------------------------------------------------------

@dataclass
class GpuSample:
    timestamp: float
    vram_mb: float
    util_pct: float


class GpuMonitor:
    def __init__(self, host: str) -> None:
        self.host = host
        self.samples: list[GpuSample] = []
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        self._stop.clear()
        self.samples.clear()
        self._proc = subprocess.Popen(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-i", str(SSH_KEY),
             f"{SSH_USER}@{self.host}",
             "nvidia-smi --query-gpu=memory.used,utilization.gpu "
             "--format=csv,noheader,nounits --loop-ms=500"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    self.samples.append(GpuSample(
                        time.time(), float(parts[0].strip()), float(parts[1].strip()),
                    ))
                except ValueError:
                    pass

    def stop(self) -> None:
        self._stop.set()
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread:
            self._thread.join(timeout=5)

    def avg_util(self) -> float:
        if not self.samples:
            return 0
        return statistics.mean(s.util_pct for s in self.samples)

    def peak_vram(self) -> float:
        if not self.samples:
            return 0
        return max(s.vram_mb for s in self.samples)


# ---------------------------------------------------------------------------
# Async request sender
# ---------------------------------------------------------------------------

async def _send(
    client: httpx.AsyncClient,
    url: str,
    file_bytes: bytes,
    filename: str,
    sem: asyncio.Semaphore,
) -> dict:
    async with sem:
        t0 = time.monotonic()
        try:
            resp = await client.post(
                url,
                files={"file": (filename, file_bytes)},
                data={"dpi": "300"},
                timeout=120.0,
            )
            wall_ms = (time.monotonic() - t0) * 1000
            if resp.status_code != 200:
                return {"ok": False, "wall_ms": wall_ms, "words": 0}
            body = resp.json()
            words = sum(
                len(line.get("words", []))
                for p in body.get("pages", [])
                for b in p.get("blocks", [])
                for line in b.get("lines", [])
            )
            return {"ok": True, "wall_ms": wall_ms, "words": words}
        except Exception:
            wall_ms = (time.monotonic() - t0) * 1000
            return {"ok": False, "wall_ms": wall_ms, "words": 0}


@dataclass
class ConcurrencyPoint:
    concurrency: int
    total_requests: int
    ok_count: int
    error_count: int
    total_wall_s: float
    rps: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_avg_ms: float
    gpu_util_avg_pct: float
    gpu_vram_peak_mb: float
    avg_words: float
    cost_per_request: float  # in cents


async def run_concurrency_level(
    host: str,
    service: str,
    concurrency: int,
    file_bytes: bytes,
    filename: str,
    num_requests: int,
    gpu_mon: GpuMonitor,
    cost_per_second: float,
) -> ConcurrencyPoint:
    """Run a burst at a specific concurrency level."""
    port = DEFAULT_PORTS[service]
    url = f"http://{host}:{port}/api/v1/ocr"
    sem = asyncio.Semaphore(concurrency)

    gpu_mon.start()

    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=120) as client:
        tasks = [_send(client, url, file_bytes, filename, sem) for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
    total_wall = time.monotonic() - t0

    gpu_mon.stop()

    ok_results = [r for r in results if r["ok"]]
    latencies = sorted(r["wall_ms"] for r in ok_results) or [0]
    errors = len(results) - len(ok_results)
    rps = len(ok_results) / total_wall if total_wall > 0 else 0
    cost_per_req = (cost_per_second / rps * 100) if rps > 0 else 0  # cents

    return ConcurrencyPoint(
        concurrency=concurrency,
        total_requests=num_requests,
        ok_count=len(ok_results),
        error_count=errors,
        total_wall_s=round(total_wall, 1),
        rps=round(rps, 3),
        latency_p50_ms=round(latencies[len(latencies) // 2]),
        latency_p95_ms=round(latencies[int(len(latencies) * 0.95)]),
        latency_avg_ms=round(statistics.mean(latencies)),
        gpu_util_avg_pct=round(gpu_mon.avg_util(), 1),
        gpu_vram_peak_mb=round(gpu_mon.peak_vram()),
        avg_words=round(statistics.mean(r["words"] for r in ok_results)) if ok_results else 0,
        cost_per_request=round(cost_per_req, 3),
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_report(
    all_data: dict[str, list[ConcurrencyPoint]],
    instance_type: str,
    output_path: Path,
) -> None:
    cost_hr = INSTANCE_COSTS.get(instance_type, 1.212)
    cost_s = cost_hr / 3600

    css = """
body { font-family: -apple-system, sans-serif; margin: 20px; background: #f5f5f5; color: #333; }
h1 { color: #2c3e50; } h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 8px; margin-top: 40px; }
h3 { margin-top: 24px; color: #2c3e50; }
table { border-collapse: collapse; width: 100%; margin: 16px 0; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
th { background: #2c3e50; color: white; padding: 10px 14px; text-align: left; font-size: 13px; }
td { padding: 8px 14px; border-bottom: 1px solid #eee; font-size: 13px; }
tr:hover { background: #f0f8ff; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.best { background: #e8f5e9 !important; font-weight: bold; }
.card { background: white; border-radius: 8px; padding: 20px; margin: 16px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
.subtitle { color: #888; font-size: 14px; }
.bar { display: inline-block; height: 16px; border-radius: 3px; min-width: 2px; }
.optimal { background: #27ae60; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }
"""

    parts = [
        f"<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>GPU Saturation Benchmark</title><style>{css}</style></head><body>",
        f"<h1>GPU Saturation Benchmark</h1>",
        f"<p class='subtitle'>Instance: {instance_type} (${cost_hr:.3f}/hr = ${cost_s * 100:.4f} cents/sec) | "
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>",
    ]

    # --- Summary: optimal operating point per service ---
    parts.append("<h2>Optimal Operating Point</h2>")
    parts.append("<p class='subtitle'>The concurrency level that maximizes RPS before errors or diminishing returns</p>")
    parts.append("<table>")
    parts.append("<tr><th>Service</th><th>Optimal C</th><th>Max RPS</th><th>GPU Util</th>"
                 "<th>VRAM Peak</th><th>Cost/request</th><th>Avg Latency</th>"
                 "<th>RPS for 500K/mo</th><th>Instances needed</th></tr>")

    monthly_pages = 500_000
    seconds_per_month = 730 * 3600

    for svc, points in all_data.items():
        # Find optimal = highest RPS with 0 errors, or best RPS/error ratio
        valid = [p for p in points if p.error_count == 0]
        if not valid:
            valid = points
        optimal = max(valid, key=lambda p: p.rps)

        rps_needed = monthly_pages / seconds_per_month
        instances = rps_needed / optimal.rps if optimal.rps > 0 else 999

        parts.append(
            f"<tr><td><b>{svc}</b></td>"
            f"<td class='num'>{optimal.concurrency}</td>"
            f"<td class='num'><b>{optimal.rps:.2f}</b></td>"
            f"<td class='num'>{optimal.gpu_util_avg_pct:.0f}%</td>"
            f"<td class='num'>{optimal.gpu_vram_peak_mb:.0f} MB</td>"
            f"<td class='num'>{optimal.cost_per_request:.2f}c</td>"
            f"<td class='num'>{optimal.latency_avg_ms / 1000:.1f}s</td>"
            f"<td class='num'>{rps_needed:.3f}</td>"
            f"<td class='num'>{instances:.2f}</td></tr>"
        )
    parts.append("</table>")

    # --- 500K/month pricing at optimal ---
    parts.append("<h2>500K Pages/Month — Cost at Optimal RPS</h2>")
    parts.append("<table>")
    parts.append("<tr><th>Service</th><th>Optimal RPS</th><th>Instance</th>"
                 "<th>Instances needed</th><th>Monthly cost</th><th>Cost/1K pages</th></tr>")

    for svc, points in all_data.items():
        valid = [p for p in points if p.error_count == 0] or points
        optimal = max(valid, key=lambda p: p.rps)
        rps = optimal.rps
        if rps <= 0:
            continue

        # Try cheaper instances for services that don't need much VRAM
        vram = optimal.gpu_vram_peak_mb
        if svc == "tesseract":
            inst_type, inst_cost = "c5.2xlarge", 0.340
        elif vram < 8000:
            inst_type, inst_cost = "g4dn.xlarge", 0.526
        else:
            inst_type, inst_cost = instance_type, cost_hr

        instances = (monthly_pages / seconds_per_month) / rps
        monthly_cost = instances * inst_cost * 730
        cost_per_1k = monthly_cost / (monthly_pages / 1000)

        parts.append(
            f"<tr><td><b>{svc}</b></td>"
            f"<td class='num'>{rps:.2f}</td>"
            f"<td>{inst_type}</td>"
            f"<td class='num'>{instances:.2f}</td>"
            f"<td class='num'><b>${monthly_cost:.0f}</b></td>"
            f"<td class='num'>${cost_per_1k:.2f}</td></tr>"
        )
    parts.append("</table>")

    # --- Per-service detail tables ---
    for svc, points in all_data.items():
        parts.append(f"<h2>{svc} — Concurrency Sweep</h2>")

        # Find optimal
        valid = [p for p in points if p.error_count == 0] or points
        optimal_c = max(valid, key=lambda p: p.rps).concurrency

        parts.append("<table>")
        parts.append("<tr><th>C</th><th>RPS</th><th>GPU Util</th><th>VRAM Peak</th>"
                     "<th>Latency p50</th><th>Latency p95</th><th>Avg Latency</th>"
                     "<th>Errors</th><th>Cost/req</th><th>RPS bar</th></tr>")

        max_rps = max(p.rps for p in points) if points else 1

        for p in points:
            is_opt = p.concurrency == optimal_c
            bar_w = int((p.rps / max_rps) * 200) if max_rps > 0 else 0
            bar_color = "#27ae60" if is_opt else ("#e74c3c" if p.error_count > 0 else "#3498db")
            opt_badge = " <span class='optimal'>OPTIMAL</span>" if is_opt else ""

            parts.append(
                f"<tr{'  class=\"best\"' if is_opt else ''}>"
                f"<td class='num'>{p.concurrency}{opt_badge}</td>"
                f"<td class='num'><b>{p.rps:.2f}</b></td>"
                f"<td class='num'>{p.gpu_util_avg_pct:.0f}%</td>"
                f"<td class='num'>{p.gpu_vram_peak_mb:.0f}</td>"
                f"<td class='num'>{p.latency_p50_ms / 1000:.1f}s</td>"
                f"<td class='num'>{p.latency_p95_ms / 1000:.1f}s</td>"
                f"<td class='num'>{p.latency_avg_ms / 1000:.1f}s</td>"
                f"<td class='num'>{p.error_count}</td>"
                f"<td class='num'>{p.cost_per_request:.2f}c</td>"
                f"<td><span class='bar' style='width:{bar_w}px;background:{bar_color}'></span></td></tr>"
            )
        parts.append("</table>")

    parts.append("</body></html>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts))
    print(f"Report: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GPU Saturation Benchmark")
    parser.add_argument("--host", required=True)
    parser.add_argument("--services", default="paddle,tesseract,surya,doctr")
    parser.add_argument("--concurrency-levels", default="1,2,4,6,8,12,16")
    parser.add_argument("--requests-per-level", type=int, default=20)
    parser.add_argument("--fixtures-dir", type=Path, default=Path("test-fixtures/"))
    parser.add_argument("--instance-type", default="g5.2xlarge")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark/results"))
    args = parser.parse_args()

    services = [s.strip() for s in args.services.split(",")]
    concurrencies = [int(c) for c in args.concurrency_levels.split(",")]
    cost_hr = INSTANCE_COSTS.get(args.instance_type, 1.212)
    cost_s = cost_hr / 3600

    # Pick a representative fixture (medium complexity)
    fixtures = sorted(
        f for f in args.fixtures_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    # Use form-kvpairs as it's representative (text + tables + structure)
    fixture = next((f for f in fixtures if "form" in f.name), fixtures[0])
    file_bytes = fixture.read_bytes()

    print(f"Host: {args.host} | Instance: {args.instance_type}")
    print(f"Fixture: {fixture.name} ({len(file_bytes)} bytes)")
    print(f"Concurrency levels: {concurrencies}")
    print(f"Requests per level: {args.requests_per_level}")

    all_data: dict[str, list[ConcurrencyPoint]] = {}

    for svc in services:
        print(f"\n{'='*50}")
        print(f"  {svc.upper()}")
        print(f"{'='*50}")

        gpu_mon = GpuMonitor(args.host)
        points: list[ConcurrencyPoint] = []

        # Warmup
        print(f"  Warmup...")
        try:
            with httpx.Client(timeout=120) as client:
                client.post(
                    f"http://{args.host}:{DEFAULT_PORTS[svc]}/api/v1/ocr",
                    files={"file": (fixture.name, file_bytes)},
                    data={"dpi": "300"},
                )
        except Exception:
            pass
        time.sleep(2)

        for c in concurrencies:
            print(f"  C={c:2d}...", end=" ", flush=True)
            point = asyncio.run(run_concurrency_level(
                args.host, svc, c, file_bytes, fixture.name,
                args.requests_per_level, gpu_mon, cost_s,
            ))
            points.append(point)

            status = "OK" if point.error_count == 0 else f"{point.error_count} errors"
            print(f"RPS={point.rps:.2f} | GPU={point.gpu_util_avg_pct:.0f}% | "
                  f"VRAM={point.gpu_vram_peak_mb:.0f}MB | "
                  f"p50={point.latency_p50_ms / 1000:.1f}s | {status}")

            # Early stop: if more than 50% errors, no point going higher
            if point.error_count > args.requests_per_level * 0.5:
                print(f"  Stopping — too many errors at C={c}")
                break

        all_data[svc] = points

    # Save raw data
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    raw_path = args.output_dir / f"{timestamp}_saturation.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps({
        "timestamp": timestamp,
        "instance_type": args.instance_type,
        "fixture": fixture.name,
        "requests_per_level": args.requests_per_level,
        "data": {svc: [asdict(p) for p in pts] for svc, pts in all_data.items()},
    }, indent=2))
    print(f"\nRaw data: {raw_path}")

    # Generate report
    report_path = args.output_dir / "saturation-report.html"
    generate_report(all_data, args.instance_type, report_path)


if __name__ == "__main__":
    main()
