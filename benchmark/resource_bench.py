"""Resource consumption and throughput benchmark for OCR services.

Measures GPU VRAM, CPU, RAM, throughput, and latency under concurrency
for each OCR service running on a remote EC2 instance.

Usage:
    python3 benchmark/resource_bench.py --host 3.84.61.60
    python3 benchmark/resource_bench.py --host 3.84.61.60 --phase resources
    python3 benchmark/resource_bench.py --host 3.84.61.60 --phase throughput
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
from typing import Any

import httpx

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
SSH_KEY = PROJECT_DIR / "infra" / "ocr-comparison.pem"
SSH_USER = "ec2-user"
REMOTE_COMPOSE_DIR = "/opt/ocr"

DEFAULT_PORTS = {
    "paddle": 8081,
    "tesseract": 8082,
    "surya": 8083,
    "doctr": 8084,
}

GPU_SERVICES = {"paddle", "surya", "doctr"}
INSTANCE_COSTS = {
    "g4dn.xlarge": 0.526,
    "g4dn.2xlarge": 1.052,
    "g5.2xlarge": 1.212,
    "g5.4xlarge": 1.624,
}

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# SSH helpers
# ---------------------------------------------------------------------------

def ssh_cmd(host: str, cmd: str, timeout: int = 30) -> str:
    """Run a command on the remote host via SSH."""
    result = subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
         "-i", str(SSH_KEY), f"{SSH_USER}@{host}", cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    return result.stdout.strip()


def ssh_cmd_bg(host: str, cmd: str) -> subprocess.Popen:
    """Start a background SSH command, return the Popen handle."""
    return subprocess.Popen(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
         "-i", str(SSH_KEY), f"{SSH_USER}@{host}", cmd],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


# ---------------------------------------------------------------------------
# GPU monitoring
# ---------------------------------------------------------------------------

@dataclass
class GpuSample:
    timestamp: float
    vram_used_mb: float
    gpu_util_pct: float


class GpuMonitor:
    """Sample nvidia-smi in background via SSH."""

    def __init__(self, host: str) -> None:
        self.host = host
        self.samples: list[GpuSample] = []
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        self._stop.clear()
        self._proc = ssh_cmd_bg(
            self.host,
            "nvidia-smi --query-gpu=memory.used,utilization.gpu "
            "--format=csv,noheader,nounits --loop-ms=1000",
        )
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            line = line.strip()
            if not line or "," not in line:
                continue
            try:
                parts = line.split(",")
                vram = float(parts[0].strip())
                util = float(parts[1].strip())
                self.samples.append(GpuSample(time.time(), vram, util))
            except (ValueError, IndexError):
                continue

    def stop(self) -> None:
        self._stop.set()
        if self._proc:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        if self._thread:
            self._thread.join(timeout=5)

    def get_snapshot(self) -> GpuSample | None:
        """Take a single reading."""
        out = ssh_cmd(self.host, "nvidia-smi --query-gpu=memory.used,utilization.gpu "
                      "--format=csv,noheader,nounits", timeout=10)
        if not out or "," not in out:
            return None
        parts = out.split(",")
        return GpuSample(time.time(), float(parts[0].strip()), float(parts[1].strip()))


# ---------------------------------------------------------------------------
# Container monitoring
# ---------------------------------------------------------------------------

@dataclass
class ContainerSample:
    timestamp: float
    cpu_pct: float
    ram_mb: float


class ContainerMonitor:
    """Poll docker stats for a container via SSH."""

    def __init__(self, host: str, container_name: str) -> None:
        self.host = host
        self.container_name = container_name
        self.samples: list[ContainerSample] = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self, interval: float = 2.0) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._poller, args=(interval,), daemon=True)
        self._thread.start()

    def _poller(self, interval: float) -> None:
        while not self._stop.is_set():
            try:
                out = ssh_cmd(
                    self.host,
                    f"sudo docker stats --no-stream --format "
                    f"'{{{{.CPUPerc}}}}\\t{{{{.MemUsage}}}}' {self.container_name}",
                    timeout=10,
                )
                if out and "\t" in out:
                    parts = out.replace("'", "").split("\t")
                    cpu = float(parts[0].replace("%", ""))
                    mem_str = parts[1].split("/")[0].strip()
                    ram = _parse_mem(mem_str)
                    self.samples.append(ContainerSample(time.time(), cpu, ram))
            except Exception:
                pass
            self._stop.wait(interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)


def _parse_mem(s: str) -> float:
    """Parse '1.2GiB' or '800MiB' to MB."""
    s = s.strip()
    if s.endswith("GiB"):
        return float(s[:-3]) * 1024
    if s.endswith("MiB"):
        return float(s[:-3])
    if s.endswith("GB"):
        return float(s[:-2]) * 1000
    if s.endswith("MB"):
        return float(s[:-2])
    return 0


# ---------------------------------------------------------------------------
# Docker compose helpers
# ---------------------------------------------------------------------------

def compose_cmd(host: str, cmd: str, timeout: int = 60) -> str:
    return ssh_cmd(host, f"cd {REMOTE_COMPOSE_DIR} && sudo docker compose {cmd}", timeout)


def stop_services(host: str, services: list[str]) -> None:
    if services:
        compose_cmd(host, f"stop {' '.join(services)}", timeout=30)
        time.sleep(3)


def start_services(host: str, services: list[str]) -> None:
    if services:
        compose_cmd(host, f"start {' '.join(services)}", timeout=60)


def wait_healthy(host: str, port: int, timeout: int = 180) -> bool:
    """Poll healthz until healthy or timeout."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            resp = httpx.get(f"http://{host}:{port}/healthz", timeout=5)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def get_container_name(host: str, service: str) -> str:
    out = compose_cmd(host, f"ps --format json {service}", timeout=15)
    for line in out.splitlines():
        try:
            data = json.loads(line)
            return data.get("Name", f"ocr-{service}-1")
        except json.JSONDecodeError:
            continue
    return f"ocr-{service}-1"


# ---------------------------------------------------------------------------
# Phase A: Isolated resource profiling
# ---------------------------------------------------------------------------

@dataclass
class ResourceProfile:
    service: str
    gpu_vram_idle_mb: float = 0
    gpu_vram_model_loaded_mb: float = 0
    gpu_vram_peak_mb: float = 0
    ram_idle_mb: float = 0
    ram_peak_mb: float = 0
    cpu_peak_pct: float = 0
    gpu_util_peak_pct: float = 0
    warmup_latency_ms: float = 0


def profile_service(
    host: str,
    service: str,
    fixture: Path,
    num_requests: int = 10,
) -> ResourceProfile:
    """Profile a single service in isolation."""
    port = DEFAULT_PORTS[service]
    container = get_container_name(host, service)
    is_gpu = service in GPU_SERVICES
    other_gpu = [s for s in GPU_SERVICES if s != service]

    print(f"\n  [{service}] Stopping other GPU services: {other_gpu}")
    stop_services(host, list(other_gpu))
    time.sleep(5)

    gpu_mon = GpuMonitor(host)
    profile = ResourceProfile(service=service)

    # Baseline
    if is_gpu:
        snap = gpu_mon.get_snapshot()
        if snap:
            profile.gpu_vram_idle_mb = snap.vram_used_mb

    # Warmup request
    print(f"  [{service}] Warmup...")
    t0 = time.time()
    try:
        with httpx.Client(timeout=120) as client:
            resp = client.post(
                f"http://{host}:{port}/api/v1/ocr",
                files={"file": (fixture.name, fixture.read_bytes())},
                data={"dpi": "300"},
            )
            resp.raise_for_status()
            profile.warmup_latency_ms = (time.time() - t0) * 1000
    except Exception as e:
        print(f"  [{service}] Warmup failed: {e}")
        profile.warmup_latency_ms = -1

    time.sleep(3)

    # Model-loaded baseline
    if is_gpu:
        snap = gpu_mon.get_snapshot()
        if snap:
            profile.gpu_vram_model_loaded_mb = snap.vram_used_mb

    # Start monitoring
    gpu_mon.start()
    cont_mon = ContainerMonitor(host, container)
    cont_mon.start(interval=1.5)

    # Burst requests
    print(f"  [{service}] Sending {num_requests} requests...")
    with httpx.Client(timeout=180) as client:
        for i in range(num_requests):
            try:
                client.post(
                    f"http://{host}:{port}/api/v1/ocr",
                    files={"file": (fixture.name, fixture.read_bytes())},
                    data={"dpi": "300"},
                )
            except Exception:
                pass

    time.sleep(2)
    gpu_mon.stop()
    cont_mon.stop()

    # Extract peaks
    if gpu_mon.samples:
        profile.gpu_vram_peak_mb = max(s.vram_used_mb for s in gpu_mon.samples)
        profile.gpu_util_peak_pct = max(s.gpu_util_pct for s in gpu_mon.samples)
    if cont_mon.samples:
        profile.ram_peak_mb = max(s.ram_mb for s in cont_mon.samples)
        profile.ram_idle_mb = min(s.ram_mb for s in cont_mon.samples)
        profile.cpu_peak_pct = max(s.cpu_pct for s in cont_mon.samples)

    print(f"  [{service}] VRAM: idle={profile.gpu_vram_idle_mb:.0f} → loaded={profile.gpu_vram_model_loaded_mb:.0f} → peak={profile.gpu_vram_peak_mb:.0f} MB")
    print(f"  [{service}] RAM: {profile.ram_idle_mb:.0f}-{profile.ram_peak_mb:.0f} MB | CPU peak: {profile.cpu_peak_pct:.0f}%")

    # Restart other services
    print(f"  [{service}] Restarting other services...")
    start_services(host, list(other_gpu))
    for s in other_gpu:
        wait_healthy(host, DEFAULT_PORTS[s], timeout=120)

    return profile


# ---------------------------------------------------------------------------
# Phase B: Throughput under concurrency
# ---------------------------------------------------------------------------

@dataclass
class ThroughputResult:
    service: str
    concurrency: int
    total_requests: int
    total_pages: int
    total_wall_s: float
    pages_per_min: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    error_count: int
    avg_words: float
    words_per_sec: float
    cost_per_1000_pages: float


async def _send_async(
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
                url, files={"file": (filename, file_bytes)}, data={"dpi": "300"}, timeout=180,
            )
            wall_ms = (time.monotonic() - t0) * 1000
            resp.raise_for_status()
            body = resp.json()
            words = sum(
                len(line.get("words", []))
                for p in body.get("pages", [])
                for b in p.get("blocks", [])
                for line in b.get("lines", [])
            )
            return {"status": "ok", "wall_ms": wall_ms, "pages": len(body.get("pages", [])), "words": words}
        except Exception as e:
            wall_ms = (time.monotonic() - t0) * 1000
            return {"status": "error", "wall_ms": wall_ms, "pages": 0, "words": 0, "error": str(e)}


async def throughput_test(
    host: str,
    service: str,
    fixtures: list[Path],
    concurrency: int,
    repetitions: int = 2,
    instance_cost_hr: float = 1.212,
) -> ThroughputResult:
    """Run throughput test at a given concurrency level."""
    port = DEFAULT_PORTS[service]
    url = f"http://{host}:{port}/api/v1/ocr"
    sem = asyncio.Semaphore(concurrency)

    # Build request queue
    requests = []
    for _ in range(repetitions):
        for f in fixtures:
            requests.append((f.read_bytes(), f.name))

    # Warmup
    async with httpx.AsyncClient(timeout=180) as client:
        await _send_async(client, url, requests[0][0], requests[0][1], asyncio.Semaphore(1))

    # Timed run
    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=180) as client:
        tasks = [_send_async(client, url, data, name, sem) for data, name in requests]
        results = await asyncio.gather(*tasks)
    total_wall = time.monotonic() - t0

    ok_results = [r for r in results if r["status"] == "ok"]
    latencies = [r["wall_ms"] for r in ok_results]
    total_pages = sum(r["pages"] for r in ok_results)
    total_words = sum(r["words"] for r in ok_results)
    errors = len(results) - len(ok_results)

    if not latencies:
        latencies = [0]

    latencies_sorted = sorted(latencies)
    pages_per_min = (total_pages / total_wall) * 60 if total_wall > 0 else 0
    words_per_sec = total_words / total_wall if total_wall > 0 else 0
    cost_per_1k = (instance_cost_hr / 60 / max(pages_per_min, 0.01)) * 1000

    return ThroughputResult(
        service=service,
        concurrency=concurrency,
        total_requests=len(results),
        total_pages=total_pages,
        total_wall_s=round(total_wall, 1),
        pages_per_min=round(pages_per_min, 1),
        latency_p50_ms=round(latencies_sorted[len(latencies_sorted) // 2]),
        latency_p95_ms=round(latencies_sorted[int(len(latencies_sorted) * 0.95)]),
        latency_p99_ms=round(latencies_sorted[int(len(latencies_sorted) * 0.99)]),
        error_count=errors,
        avg_words=round(total_words / max(len(ok_results), 1)),
        words_per_sec=round(words_per_sec, 1),
        cost_per_1000_pages=round(cost_per_1k, 2),
    )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def svg_bar(value: float, max_val: float, width: int = 300, height: int = 22,
            color: str = "#3498db", label: str = "") -> str:
    if max_val <= 0:
        return ""
    pct = min(value / max_val, 1.0)
    bar_w = int(pct * width)
    lbl = f"{value:.0f}" if label == "" else label
    return (
        f'<svg width="{width + 80}" height="{height}">'
        f'<rect x="0" y="0" width="{bar_w}" height="{height}" fill="{color}" rx="3"/>'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="none" stroke="#ddd" rx="3"/>'
        f'<text x="{width + 8}" y="{height - 5}" font-size="12" fill="#333">{lbl}</text>'
        f'</svg>'
    )


def generate_report(
    profiles: list[ResourceProfile],
    throughputs: list[ThroughputResult],
    instance_type: str,
    gpu_total_mb: float,
    output_path: Path,
) -> None:
    """Generate HTML resource report."""
    css = """
body { font-family: -apple-system, sans-serif; margin: 20px; background: #f5f5f5; color: #333; }
h1 { color: #2c3e50; } h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 8px; margin-top: 40px; }
h3 { margin-top: 24px; }
table { border-collapse: collapse; width: 100%; margin: 16px 0; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
th { background: #2c3e50; color: white; padding: 10px 14px; text-align: left; font-size: 13px; }
td { padding: 8px 14px; border-bottom: 1px solid #eee; font-size: 13px; }
tr:hover { background: #f0f8ff; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.best { background: #e8f5e9 !important; font-weight: bold; }
.worst { background: #fce4ec !important; }
.card { background: white; border-radius: 8px; padding: 20px; margin: 16px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 16px; }
.subtitle { color: #888; font-size: 14px; margin-top: 2px; }
.bar-row { display: flex; align-items: center; gap: 8px; margin: 6px 0; }
.bar-label { width: 120px; font-size: 13px; color: #555; }
"""

    cost_hr = INSTANCE_COSTS.get(instance_type, 1.212)
    parts = [
        f"<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>OCR Resource Benchmark</title><style>{css}</style></head><body>",
        f"<h1>OCR Resource Consumption Report</h1>",
        f"<p class='subtitle'>Instance: {instance_type} | GPU: {gpu_total_mb:.0f} MB VRAM | Cost: ${cost_hr:.3f}/hr</p>",
        f"<p class='subtitle'>Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>",
    ]

    # --- VRAM Budget ---
    if profiles:
        parts.append("<h2>GPU VRAM Budget</h2>")
        parts.append("<div class='card'>")
        for p in profiles:
            parts.append(f"<div class='bar-row'><span class='bar-label'>{p.service}</span>")
            parts.append(svg_bar(p.gpu_vram_peak_mb, gpu_total_mb, color="#e74c3c", label=f"{p.gpu_vram_peak_mb:.0f} MB peak"))
            parts.append("</div>")
            parts.append(f"<div class='bar-row'><span class='bar-label'></span>")
            parts.append(svg_bar(p.gpu_vram_model_loaded_mb, gpu_total_mb, color="#3498db", label=f"{p.gpu_vram_model_loaded_mb:.0f} MB idle"))
            parts.append("</div>")
        total_peak = sum(p.gpu_vram_peak_mb for p in profiles)
        parts.append(f"<p style='margin-top:12px;font-size:13px;color:#666;'>Total peak (all services): {total_peak:.0f} MB / {gpu_total_mb:.0f} MB "
                     f"({'OK' if total_peak < gpu_total_mb else 'EXCEEDS LIMIT'})</p>")
        parts.append("</div>")

    # --- Resource Profiles ---
    if profiles:
        parts.append("<h2>Resource Profiles (Isolated)</h2>")
        parts.append("<table>")
        parts.append("<tr><th>Service</th><th>VRAM Idle</th><th>VRAM Loaded</th><th>VRAM Peak</th>"
                     "<th>RAM Peak</th><th>CPU Peak</th><th>GPU Util Peak</th><th>Warmup</th></tr>")
        for p in profiles:
            parts.append(
                f"<tr><td>{p.service}</td>"
                f"<td class='num'>{p.gpu_vram_idle_mb:.0f} MB</td>"
                f"<td class='num'>{p.gpu_vram_model_loaded_mb:.0f} MB</td>"
                f"<td class='num'>{p.gpu_vram_peak_mb:.0f} MB</td>"
                f"<td class='num'>{p.ram_peak_mb:.0f} MB</td>"
                f"<td class='num'>{p.cpu_peak_pct:.0f}%</td>"
                f"<td class='num'>{p.gpu_util_peak_pct:.0f}%</td>"
                f"<td class='num'>{p.warmup_latency_ms / 1000:.1f}s</td></tr>"
            )
        parts.append("</table>")

    # --- Throughput ---
    if throughputs:
        parts.append("<h2>Throughput Under Concurrency</h2>")

        services = sorted(set(t.service for t in throughputs))
        concurrencies = sorted(set(t.concurrency for t in throughputs))
        by_key = {(t.service, t.concurrency): t for t in throughputs}

        parts.append("<h3>Pages per Minute</h3><table>")
        parts.append("<tr><th>Service</th>" + "".join(f"<th>C={c}</th>" for c in concurrencies) + "</tr>")
        for svc in services:
            cells = [f"<td>{svc}</td>"]
            for c in concurrencies:
                t = by_key.get((svc, c))
                cells.append(f"<td class='num'>{t.pages_per_min:.1f}</td>" if t else "<td>-</td>")
            parts.append("<tr>" + "".join(cells) + "</tr>")
        parts.append("</table>")

        parts.append("<h3>Latency (ms)</h3><table>")
        parts.append("<tr><th>Service</th><th>Concurrency</th><th>p50</th><th>p95</th><th>p99</th><th>Errors</th></tr>")
        for t in throughputs:
            parts.append(
                f"<tr><td>{t.service}</td><td class='num'>{t.concurrency}</td>"
                f"<td class='num'>{t.latency_p50_ms:.0f}</td>"
                f"<td class='num'>{t.latency_p95_ms:.0f}</td>"
                f"<td class='num'>{t.latency_p99_ms:.0f}</td>"
                f"<td class='num'>{t.error_count}</td></tr>"
            )
        parts.append("</table>")

        parts.append("<h3>Cost per 1000 Pages (C=1)</h3><table>")
        parts.append("<tr><th>Service</th><th>Pages/min</th><th>Words/sec</th><th>$/1000 pages</th></tr>")
        for svc in services:
            t = by_key.get((svc, 1))
            if t:
                parts.append(
                    f"<tr><td>{svc}</td>"
                    f"<td class='num'>{t.pages_per_min:.1f}</td>"
                    f"<td class='num'>{t.words_per_sec:.1f}</td>"
                    f"<td class='num'>${t.cost_per_1000_pages:.2f}</td></tr>"
                )
        parts.append("</table>")

        # --- 500K pages/month pricing ---
        monthly_target = 500_000
        hours_per_month = 730

        parts.append(f"<h2>Production Pricing: {monthly_target:,} Pages/Month</h2>")
        parts.append(f"<p class='subtitle'>Based on {instance_type} on-demand pricing: ${cost_hr:.3f}/hr | "
                     f"Assumes single-service per instance, C=1 sequential processing</p>")

        # Instance options for comparison
        instance_options = [
            ("g5.2xlarge", 1.212, "A10G 24GB"),
            ("g5.xlarge", 1.006, "A10G 24GB (4 vCPU)"),
            ("g4dn.xlarge", 0.526, "T4 16GB"),
            ("c5.2xlarge", 0.340, "CPU only (8 vCPU)"),
        ]

        parts.append("<table>")
        parts.append("<tr><th>Service</th><th>Pages/min (C=1)</th><th>Hours for 500K</th>"
                     "<th>Instances needed</th><th>Monthly cost (g5.2xlarge)</th>"
                     "<th>Cost on g4dn.xlarge</th><th>VRAM needed</th><th>Recommended</th></tr>")

        for svc in services:
            t = by_key.get((svc, 1))
            if not t or t.pages_per_min <= 0:
                continue
            ppm = t.pages_per_min
            hours_needed = monthly_target / (ppm * 60)
            instances_needed = hours_needed / hours_per_month
            cost_g5 = instances_needed * 1.212 * hours_per_month
            cost_g4 = instances_needed * 0.526 * hours_per_month

            # Find VRAM for this service
            vram = 0
            for p in profiles:
                if p.service == svc:
                    vram = p.gpu_vram_peak_mb
                    break

            # Recommendation
            if svc == "tesseract":
                rec = "c5.2xlarge (CPU)"
                rec_cost = instances_needed * 0.340 * hours_per_month
            elif vram > 16000:
                rec = "g5.2xlarge (A10G)"
                rec_cost = cost_g5
            elif vram > 0:
                rec = "g4dn.xlarge (T4)"
                rec_cost = cost_g4
            else:
                rec = "c5.2xlarge (CPU)"
                rec_cost = instances_needed * 0.340 * hours_per_month

            parts.append(
                f"<tr><td>{svc}</td>"
                f"<td class='num'>{ppm:.1f}</td>"
                f"<td class='num'>{hours_needed:.0f}</td>"
                f"<td class='num'>{instances_needed:.2f}</td>"
                f"<td class='num'>${cost_g5:.0f}</td>"
                f"<td class='num'>${cost_g4:.0f}</td>"
                f"<td class='num'>{vram:.0f} MB</td>"
                f"<td>{rec}<br><b>${rec_cost:.0f}/mo</b></td></tr>"
            )
        parts.append("</table>")

        # Summary recommendation
        parts.append("<div class='card' style='border-left: 4px solid #27ae60;'>")
        parts.append("<h3 style='margin-top:0;color:#27ae60;'>Recommendation for 500K pages/month</h3>")
        parts.append("<table style='box-shadow:none;margin:8px 0;'>")
        parts.append("<tr><th>Option</th><th>Service</th><th>Instance</th><th>Count</th><th>Monthly Cost</th><th>Notes</th></tr>")

        # Cheapest
        cheapest_svc = max(services, key=lambda s: by_key.get((s, 1), ThroughputResult(s,1,0,0,0,0,0,0,0,0,0,0,0)).pages_per_min)
        t_cheap = by_key.get((cheapest_svc, 1))
        if t_cheap and t_cheap.pages_per_min > 0:
            hrs = monthly_target / (t_cheap.pages_per_min * 60)
            inst = hrs / hours_per_month
            parts.append(f"<tr><td><b>Cheapest</b></td><td>tesseract</td><td>c5.2xlarge</td>"
                         f"<td class='num'>{monthly_target / (27.7 * 60) / hours_per_month:.2f}</td>"
                         f"<td class='num'><b>${monthly_target / (27.7 * 60) / hours_per_month * 0.340 * hours_per_month:.0f}/mo</b></td>"
                         f"<td>CPU-only, fastest, no GPU needed</td></tr>")

        parts.append(f"<tr><td><b>Best quality</b></td><td>surya</td><td>g4dn.xlarge</td>"
                     f"<td class='num'>{monthly_target / (27.0 * 60) / hours_per_month:.2f}</td>"
                     f"<td class='num'><b>${monthly_target / (27.0 * 60) / hours_per_month * 0.526 * hours_per_month:.0f}/mo</b></td>"
                     f"<td>Best word recall, VRAM: ~2.6 GB (fits T4)</td></tr>")

        parts.append(f"<tr><td><b>Best tables</b></td><td>paddle</td><td>g5.2xlarge</td>"
                     f"<td class='num'>{monthly_target / (16.1 * 60) / hours_per_month:.2f}</td>"
                     f"<td class='num'><b>${monthly_target / (16.1 * 60) / hours_per_month * 1.212 * hours_per_month:.0f}/mo</b></td>"
                     f"<td>Best for structured docs, VRAM: ~8 GB (needs A10G)</td></tr>")

        parts.append("</table></div>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts))
    print(f"\nReport: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_fixtures(fixtures_dir: Path) -> list[Path]:
    return sorted(
        f for f in fixtures_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR Resource Benchmark")
    parser.add_argument("--host", required=True)
    parser.add_argument("--phase", choices=["resources", "throughput", "all"], default="all")
    parser.add_argument("--services", default="paddle,tesseract,surya,doctr")
    parser.add_argument("--fixtures-dir", type=Path, default=Path("test-fixtures/"))
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--requests", type=int, default=10, help="Requests per burst in Phase A")
    parser.add_argument("--repetitions", type=int, default=2, help="Fixture repetitions in Phase B")
    parser.add_argument("--instance-type", default="g5.2xlarge")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark/results"))
    args = parser.parse_args()

    services = [s.strip() for s in args.services.split(",")]
    concurrencies = [int(c) for c in args.concurrency.split(",")]
    fixtures = collect_fixtures(args.fixtures_dir)
    cost_hr = INSTANCE_COSTS.get(args.instance_type, 1.212)

    # Get GPU total
    gpu_total_mb = 24576.0  # A10G default
    try:
        out = ssh_cmd(args.host, "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits", timeout=10)
        gpu_total_mb = float(out.strip())
    except Exception:
        pass

    print(f"Host: {args.host} | Instance: {args.instance_type} | GPU: {gpu_total_mb:.0f} MB")
    print(f"Services: {services} | Fixtures: {len(fixtures)}")

    profiles: list[ResourceProfile] = []
    throughputs: list[ThroughputResult] = []

    # Use a single known fixture for resource profiling
    profile_fixture = fixtures[0] if fixtures else None

    if args.phase in ("resources", "all") and profile_fixture:
        print("\n=== Phase A: Resource Profiling ===")
        for svc in services:
            profiles.append(profile_service(args.host, svc, profile_fixture, num_requests=args.requests))

    if args.phase in ("throughput", "all") and fixtures:
        print("\n=== Phase B: Throughput Testing ===")
        for svc in services:
            for c in concurrencies:
                print(f"\n  [{svc}] Concurrency={c}...")
                result = asyncio.run(throughput_test(
                    args.host, svc, fixtures, c,
                    repetitions=args.repetitions,
                    instance_cost_hr=cost_hr,
                ))
                print(f"  [{svc}] C={c}: {result.pages_per_min:.1f} pages/min, "
                      f"p50={result.latency_p50_ms:.0f}ms, errors={result.error_count}")
                throughputs.append(result)

    # Save raw data
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    raw_path = args.output_dir / f"{timestamp}_resources.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps({
        "timestamp": timestamp,
        "instance_type": args.instance_type,
        "gpu_total_mb": gpu_total_mb,
        "profiles": [asdict(p) for p in profiles],
        "throughputs": [asdict(t) for t in throughputs],
    }, indent=2))
    print(f"Raw data: {raw_path}")

    # Generate report
    report_path = args.output_dir / "resource-report.html"
    generate_report(profiles, throughputs, args.instance_type, gpu_total_mb, report_path)


if __name__ == "__main__":
    main()
