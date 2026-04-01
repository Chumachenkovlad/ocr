"""OCR Comparison Gateway — sends requests to OCR services sequentially."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, UploadFile, Form
from fastapi.staticfiles import StaticFiles

from src.config import Settings

settings = Settings()
http_client: httpx.AsyncClient | None = None

SERVICES: dict[str, str] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, SERVICES
    http_client = httpx.AsyncClient(timeout=settings.request_timeout)
    SERVICES = {
        "paddle": settings.paddle_url,
        "tesseract": settings.tesseract_url,
        "surya": settings.surya_url,
        "doctr": settings.doctr_url,
    }
    yield
    await http_client.aclose()


app = FastAPI(title="OCR Comparison Gateway", lifespan=lifespan)


async def _call_service(
    name: str,
    url: str,
    file_bytes: bytes,
    filename: str,
    dpi: int,
    language: str,
) -> dict[str, Any]:
    """Call a single OCR service and return its result or error."""
    assert http_client is not None
    t0 = time.monotonic()
    try:
        files = {"file": (filename, file_bytes)}
        data = {"dpi": str(dpi), "language": language}
        resp = await http_client.post(f"{url}/api/v1/ocr", files=files, data=data)
        resp.raise_for_status()
        elapsed = (time.monotonic() - t0) * 1000
        return {"status": "ok", "result": resp.json(), "gateway_latency_ms": round(elapsed, 1)}
    except Exception as exc:
        elapsed = (time.monotonic() - t0) * 1000
        return {"status": "error", "error": str(exc), "gateway_latency_ms": round(elapsed, 1)}


@app.post("/api/v1/compare")
async def compare_endpoint(
    file: UploadFile,
    dpi: int = Form(default=300),
    language: str = Form(default="en"),
    providers: str = Form(default="paddle,tesseract,surya,doctr"),
) -> dict[str, Any]:
    """Send to selected services sequentially to avoid GPU memory contention."""
    content = await file.read()
    filename = file.filename or "upload"
    selected = [p.strip() for p in providers.split(",") if p.strip()]

    t0 = time.monotonic()
    results: dict[str, Any] = {}

    for name in selected:
        url = SERVICES.get(name)
        if not url:
            results[name] = {"status": "error", "error": f"Unknown provider: {name}"}
            continue
        results[name] = await _call_service(name, url, content, filename, dpi, language)

    total_ms = (time.monotonic() - t0) * 1000

    return {
        "total_latency_ms": round(total_ms, 1),
        "results": results,
    }


@app.get("/api/v1/providers")
async def list_providers() -> dict[str, Any]:
    """List available providers."""
    return {"providers": list(SERVICES.keys())}


@app.get("/healthz")
async def health() -> dict[str, Any]:
    """Check all downstream services."""
    assert http_client is not None
    statuses: dict[str, Any] = {}
    for name, url in SERVICES.items():
        try:
            resp = await http_client.get(f"{url}/healthz", timeout=5)
            statuses[name] = resp.json()
        except Exception as exc:
            statuses[name] = {"status": "error", "error": str(exc)}

    all_ok = all(s.get("status") == "ok" for s in statuses.values())
    return {"status": "ok" if all_ok else "degraded", "services": statuses}


# Serve frontend — must be last so it doesn't shadow API routes
app.mount("/", StaticFiles(directory="public", html=True), name="frontend")
