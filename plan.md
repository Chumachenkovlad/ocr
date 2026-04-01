# OCR Comparison Platform — Implementation Plan

## Context

We need to evaluate 4 self-hosted OCR engines (Surya, docTR, PaddleOCR, Tesseract) side by side on identical infrastructure. Two services already exist externally and need to be moved/rewritten; two are new. All must share a unified output schema, be deployable via a single `docker-compose.yml`, and feed into a comparison frontend and benchmark runner.

Existing docs in `docs/` (DR-001, comparison matrix, output structures) provide the canonical data model and adapter notes. This plan builds on that foundation.

---

## Project Structure

```
/Users/cvla/work/ocr/
  docs/                           # existing (keep as-is)
  shared/
    ocr_schema/
      __init__.py
      models.py                   # Pydantic v2 unified output schema
      pyproject.toml
  services/
    paddle/                       # moved from ocr-paddle-service, adapted
    tesseract/                    # Python rewrite of ocr-service
    surya/                        # new
    doctr/                        # new
    gateway/                      # orchestrator + comparison frontend
  benchmark/
    run.py                        # benchmark runner
    analyze.py                    # results analysis
    results/                      # output CSV/JSON
  test-fixtures/                  # shared test images/PDFs
  docker-compose.yml
  Makefile
```

---

## Phase 1: Foundation

### 1.1 — Init repo + shared schema

- `git init`, `.gitignore` (Python, Docker, __pycache__, .env)
- Create `shared/ocr_schema/models.py` — Pydantic v2 models derived from DR-001 canonical model (lines 137-203 of `docs/DR-001-ocr-adapter.md`):
  - `BoundingBox(x1, y1, x2, y2)` — pixel coords, float
  - `OcrWord(text, confidence, bbox)`
  - `OcrLine(text, words, bbox)`
  - `OcrBlock(block_type, lines, bbox)`
  - `OcrPage(page_number, width, height, blocks)`
  - `OcrResult(provider, pages, raw, latency_ms)`
- `shared/ocr_schema/pyproject.toml` — installable local package

### 1.2 — Move PaddleOCR service

Source: `/Users/cvla/work/pdffiller-other/ocr-paddle-service`

- Copy `src/`, `Dockerfile`, `pyproject.toml`, `public/`, `.env.example` into `services/paddle/`
- Add path dependency on `../../shared` in pyproject.toml
- Modify `src/core/adapter.py`:
  - Replace `PageSchema`/`BlockSchema`/`LineSchema`/`WordSchema`/`BBoxSchema` imports with shared `OcrResult`/`OcrPage`/`OcrBlock`/`OcrLine`/`OcrWord`/`BoundingBox`
  - Convert `BBox(x, y, w, h)` → `BoundingBox(x1=x, y1=y, x2=x+w, y2=y+h)` in `split_line_to_words` and block assembly
- Modify `src/api/router.py`:
  - Return `OcrResult(provider="paddle", pages=[...], latency_ms=elapsed, raw=raw_output)` instead of `OCRResponse`
- Keep `engine.py`, `pdf_renderer.py`, `utils/` unchanged
- Keep existing Dockerfile multi-stage (cpu/cpu-local/gpu)
- Copy existing `docker-compose.yml` profiles as reference

### 1.3 — Rewrite Tesseract service in Python

Source: `/Users/cvla/work/pdffiller-other/ocr-service` (TypeScript/NestJS)

- Create `services/tesseract/` with FastAPI structure matching paddle
- Port hOCR parser from TS (`src/ocr/hocr-parser.service.ts`) to Python:
  - Use `lxml.html` for parsing (CSS-class-based element lookup)
  - Port `parseTitleAttribute()` — parses `"bbox X1 Y1 X2 Y2; x_wconf N"` strings
  - Port block/line/word hierarchy: `ocr_page` → `ocr_carea` → `ocr_par` → `ocr_line` → `ocrx_word`
  - Confidence: `x_wconf / 100` (0-100 → 0-1)
  - BBox: hOCR `[x1,y1,x2,y2]` maps directly to `BoundingBox`
- Tesseract invocation: `subprocess.run(["tesseract", input, "stdout", "-l", lang, "--oem", "3", "--psm", "3", "--dpi", str(dpi), "hocr"])`
- Add PDF support via `pdf2image` (not in TS original)
- Dockerfile: `python:3.12-slim` + `apt-get install tesseract-ocr tesseract-ocr-eng poppler-utils`

---

## Phase 2: New Services

### 2.1 — Surya OCR service

- Create `services/surya/` — FastAPI
- Engine: load `surya-ocr` detection + recognition models at startup
- Pin exact `surya-ocr` version (import paths change between versions per docs)
- Adapter:
  - `TextLine.bbox` is `[x1,y1,x2,y2]` pixel coords — maps directly to `BoundingBox` (no denorm)
  - No word-level output — split line text into words using same proportional-width logic as PaddleOCR's `split_line_to_words`
  - No native block grouping — group lines into blocks by Y-proximity
  - Optionally run `batch_layout_detection` for block types (text/table/figure)
  - Line-level confidence propagated to words
- Dockerfile: `python:3.12-slim` + `surya-ocr` + `torch` (CPU by default, GPU optional)

### 2.2 — docTR service

- Create `services/doctr/` — FastAPI
- Engine: `ocr_predictor(pretrained=True)` loaded at startup
- Input: `DocumentFile.from_images()` or `DocumentFile.from_pdf()`
- Use `.export()` for dict output (not `.render()`)
- Adapter:
  - Coords are normalized 0-1 — denormalize: `x_px = x_norm * width`
  - Geometry `[[x_min, y_min], [x_max, y_max]]` → `BoundingBox(x1, y1, x2, y2)` after denorm
  - `dimensions` field is `[height, width]` (verify empirically)
  - 4-level nesting (page→block→line→word) maps directly to shared schema
  - Per-word confidence available — cleanest adapter
- Dockerfile: `python:3.12-slim` + `python-doctr[torch]`

---

## Phase 3: Gateway + Comparison Frontend

### 3.1 — Gateway service

- Create `services/gateway/` — thin FastAPI app
- `POST /api/v1/compare` — accepts file upload, fans out to all 4 services via `httpx.AsyncClient`
- Returns: `{results: {paddle: OcrResult, tesseract: OcrResult, surya: OcrResult, doctr: OcrResult}}`
- 30s timeout per service, partial results on failure
- `GET /healthz` — aggregates health from all downstream services
- Serves comparison frontend from `public/`

### 3.2 — Comparison frontend

- Single `index.html` — no build step (same pattern as existing PaddleOCR viewer)
- Start from PaddleOCR's `public/index.html` (canvas bbox overlay, JSON viewer, confidence color coding)
- Extend to:
  - 2x2 grid of overlay panels (one per provider)
  - Latency comparison display
  - Tabbed raw JSON view per provider
  - Provider toggle on/off
  - Side-by-side text diff between providers

---

## Phase 4: Docker Compose + Deployment

### 4.1 — docker-compose.yml

```yaml
services:
  paddle:
    build: {context: services/paddle, target: cpu}
    ports: ["8081:8080"]
    volumes: [model-cache:/models]
    deploy: {resources: {limits: {memory: 12G}}}
    healthcheck: ...

  tesseract:
    build: services/tesseract
    ports: ["8082:8080"]
    deploy: {resources: {limits: {memory: 2G}}}
    healthcheck: ...

  surya:
    build: services/surya
    ports: ["8083:8080"]
    deploy: {resources: {limits: {memory: 8G}}}
    healthcheck: ...

  doctr:
    build: services/doctr
    ports: ["8084:8080"]
    deploy: {resources: {limits: {memory: 6G}}}
    healthcheck: ...

  gateway:
    build: services/gateway
    ports: ["8080:8080"]
    environment:
      PADDLE_URL: http://paddle:8080
      TESSERACT_URL: http://tesseract:8080
      SURYA_URL: http://surya:8080
      DOCTR_URL: http://doctr:8080
    depends_on:
      paddle: {condition: service_healthy}
      tesseract: {condition: service_healthy}
      surya: {condition: service_healthy}
      doctr: {condition: service_healthy}
```

All services expose: `GET /healthz` + `POST /api/v1/ocr`

### 4.2 — AWS deployment

- Target: single EC2 instance (e.g. `g4dn.xlarge` with T4 GPU, or CPU-only `m5.2xlarge`)
- Push images to ECR, `docker compose up` on instance
- Sequential inference if GPU memory is tight (gateway queues, doesn't fan out in parallel)

---

## Phase 5: Benchmarking

### 5.1 — Test fixtures

```
test-fixtures/
  simple-1page.png        # clean typed text, single column
  table-1page.png         # document with table
  multipage-3.pdf         # 3-page mixed document
  noisy-scan.png          # low quality scan
  complex-layout.png      # multi-column, headers, figures
```

### 5.2 — Benchmark runner (`benchmark/run.py`)

- Takes `--fixtures-dir` and `--services` args
- For each fixture, sends to each service via `httpx`
- Collects per-request: `latency_ms` (from response), wall-clock time, word count, avg confidence
- Writes `benchmark/results/YYYY-MM-DD.json`

### 5.3 — Analysis (`benchmark/analyze.py`)

- Reads results JSON
- Computes: p50/p95 latency per provider, word-count variance, confidence distribution
- Text similarity between providers (difflib SequenceMatcher)
- Outputs Markdown report to `benchmark/results/report.md`

---

## Unified API Contract

All 4 services expose the same interface:

```
POST /api/v1/ocr
Content-Type: multipart/form-data
  file: <image or PDF>
  dpi: int (optional, default 300)
  language: str (optional, default "en")

Response: 200 OK
{
  "provider": "paddle" | "tesseract" | "surya" | "doctr",
  "pages": [...],
  "raw": {...} | null,
  "latency_ms": 1234.5
}

GET /healthz
Response: {"status": "ok", "provider": "..."}
```

---

## Critical Files

| File | Role |
|------|------|
| `shared/ocr_schema/models.py` | Single source of truth for output schema |
| `services/paddle/src/core/adapter.py` | Adapt PaddleOCR raw → shared schema (modify existing) |
| `services/tesseract/src/hocr_parser.py` | Port from TS `hocr-parser.service.ts` |
| `services/surya/src/adapter.py` | TextLine → shared schema + word splitting |
| `services/doctr/src/adapter.py` | Denormalize coords + map 4-level nesting |
| `services/gateway/src/main.py` | Fan-out orchestrator |
| `services/gateway/public/index.html` | Comparison frontend |
| `docker-compose.yml` | Single deployment orchestration |

---

## Verification

1. `docker compose build` — all 5 images build successfully
2. `docker compose up` — all services pass health checks
3. Upload a test image via gateway frontend → see 4 results side by side
4. Each service individually: `curl -F file=@test.png http://localhost:808X/api/v1/ocr | python -m json.tool` validates against `OcrResult` schema
5. `python benchmark/run.py --fixtures-dir test-fixtures/` completes and writes results
6. Results show all 4 providers returning structurally valid, non-empty responses for each fixture

---

## Implementation Order

### Wave 1 (serial, quick)
1. Shared schema + git init

### Wave 2 (all parallel — 6 tasks)
2. PaddleOCR service (move + adapt)
3. Tesseract service (Python rewrite)
4. Surya service (new)
5. docTR service (new)
6. Gateway + frontend (needs only the API contract from wave 1, not running services)
7. Docker compose (needs only port/healthcheck convention from wave 1)

### Wave 3 (serial — needs running services)
8. Benchmark runner + test fixtures
