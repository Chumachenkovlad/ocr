# OCR-Powered Document Editing — Feasibility Review

> **Date**: April 2026
> **Status**: Research complete. Ready for implementation planning.
> **Scope**: Self-hosted OCR integration + scanned document editing pipeline for a web-based document editor.

---

## Executive Summary

This knowledge base evaluates the feasibility of adding OCR-powered editing to a web document editor. The use case: a user uploads a scanned PDF, the system recognizes all text, and the user can edit it as if it were a native document — changing words, fonts, sizes — with the background image seamlessly updated.

The pipeline has **five distinct engineering problems**, each with different maturity levels in the open-source ecosystem. OCR and text recognition are production-ready. Font detection and background inpainting are solvable but require integration work. Layout reconstruction and reading order remain the least mature area.

**Bottom line**: a production-quality editing pipeline is achievable with self-hosted open-source tooling at **~$150-250/month** for 500K pages. The primary risk is not any single component but the integration complexity of chaining five steps while maintaining spatial precision across all of them.

---

## The Pipeline

```
  SCANNED DOCUMENT (image / PDF)
           |
           v
  +------------------+
  |   1. OCR ENGINE  |  Detect text, extract words + bounding boxes
  +------------------+
           |
           v
  +------------------+
  |  2. LAYOUT       |  Reconstruct reading order, paragraphs,
  |     ANALYSIS     |  columns, tables, headers/footers
  +------------------+
           |
           v
  +------------------+
  |  3. FONT         |  Estimate font family, size, weight,
  |     DETECTION    |  style from the rendered text image
  +------------------+
           |
           v
  +-----------------------------------------+
  |          EDITOR  (user interaction)      |
  |  User sees editable text blocks over     |
  |  the scanned image. Modifies content.    |
  +-----------------------------------------+
           |
           v
  +------------------+
  |  4. BACKGROUND   |  Erase original text from the background
  |     INPAINTING   |  image, preserving paper texture/lines
  +------------------+
           |
           v
  +------------------+
  |  5. TEXT          |  Render the edited text with matched
  |     RE-RENDERING |  font over the cleaned background
  +------------------+
           |
           v
  EDITED DOCUMENT (image / PDF)
```

---

## Problem Breakdown

### Problem 1: OCR — Text Recognition

**Status**: Solved. Production-ready.

Detect all text in a scanned document and extract words with bounding boxes and confidence scores. Four self-hosted engines were benchmarked across 10 fixture types.

| Engine | Best for | Latency (GPU) | License |
|--------|----------|--------------|---------|
| **PaddleOCR** | Tables, forms, layout analysis | ~200ms/page | Apache-2.0 |
| **Tesseract** | CPU-only, simple docs, 100+ languages | ~2s/page (CPU) | Apache-2.0 |
| **Surya** | Maximum recall, 90+ languages | ~150ms/page | GPL-3.0 |
| **docTR** | Throughput (30K pages/hr on T4) | ~80ms/page | Apache-2.0 |

**Recommendation**: PaddleOCR (PP-StructureV3) as primary — only engine with native layout analysis, table HTML, and reading order via pointer network. Tesseract as CPU fallback.

> **Detailed docs**: [DR-001-ocr-adapter.md](DR-001-ocr-adapter.md) (adapter design), [comparison.md](comparison.md) (feature matrix), [output-structures.md](output-structures.md) (raw output shapes), [cost-performance.md](cost-performance.md) (throughput + pricing)

---

### Problem 2: Layout Analysis & Reading Order

**Status**: Partially solved. The hardest unsolved gap.

Multi-column documents, mixed text/table pages, and complex reading order remain challenging. Without correct layout reconstruction, edited text ends up in wrong positions or wrong sequence.

| Aspect | Challenge |
|--------|-----------|
| Multi-column detection | Text flows column-by-column, not left-to-right across page |
| Reading order | Headers, sidebars, footnotes break simple top-to-bottom ordering |
| Table structure | Cell boundaries, merged cells, nested tables |
| Mixed content | Figures, captions, equations interleaved with text |

**Best available**: PaddleOCR's PP-StructureV3 with a 6-layer transformer pointer network for reading order — the only engine with explicit reading order recovery. Surya and docTR have no native reading order. Tesseract's PSM modes handle basic column detection but fail on complex layouts.

**Evaluation metric**: Normalized Edit Distance on block sequences (OmniDocBench methodology).

> **Detailed docs**: [research-benchmark-gaps.md](research-benchmark-gaps.md) §1 (layout metrics, datasets, engine comparison)

---

### Problem 3: Font Detection

**Status**: Solvable with integration work. No single tool covers all aspects.

The editor needs to re-render edited text with a visually matching font. Scanned documents have no embedded font metadata. This requires estimating four properties:

| Property | Approach | Accuracy | Effort |
|----------|----------|----------|--------|
| **Font size** | `bbox_height_px / DPI * 72 / 1.2` | ±1-2pt at 300 DPI | Already implemented |
| **Font family** | Storia AI ONNX classifier (~3K Google Fonts) | ~70-80% top-5 | 2-3 days |
| **Serif vs sans** | ResNet-18 coarse classifier | ~95% | 1-2 days |
| **Bold / italic** | Stroke Width Transform + ink density | ~85-90% | 3-5 days |

**Key finding**: VLMs (GPT-4o, Claude) achieve only ~31% accuracy on font recognition. Dedicated classifiers significantly outperform them.

> **Detailed docs**: [research-font-estimation.md](research-font-estimation.md) (full analysis, model comparison, proposed architecture)

---

### Problem 4: Background Inpainting (Text Removal)

**Status**: Solvable. Multiple approaches with clear quality/latency trade-offs.

After OCR detects text and the user edits it, the original text must be erased from the background image before re-rendering. The challenge varies with background complexity:

```
  Background type           Approach                    Latency
  ─────────────────────────────────────────────────────────────
  White / solid color  -->  OpenCV Telea (classical)     <10ms
  Paper texture        -->  LaMa (Fourier convolutions)  ~50ms GPU
  Lined / graph paper  -->  LaMa or GaRNet               ~50ms GPU
  Complex / stamps     -->  TMIM + Uformer-B             ~200ms GPU
  Born-digital PDF     -->  PyMuPDF redaction API         instant
```

**Mask generation**: OCR bounding boxes → dilate 3-5px → use as inpainting mask. For higher precision, use PaddleOCR's DBNet for pixel-level text segmentation masks.

**Browser-side**: OpenCV.js Telea (5.3 MB WASM) for white backgrounds. LaMa ONNX (~208 MB, cacheable) for complex backgrounds via ONNX Runtime Web.

> **Detailed docs**: [research-inpainting.md](research-inpainting.md) (10 sections covering all STR models, pipeline architecture, commercial alternatives)

---

### Problem 5: Text Re-Rendering

**Status**: Straightforward given correct font detection.

Render the edited text over the inpainted background using detected font metadata. This is standard web rendering (CSS `font-family`, `font-size`, `font-weight`, `font-style`) with Google Fonts.

**Key challenges**:
- Spatial precision — re-rendered text must align exactly with the original bounding box
- Line wrapping — edited text may be longer/shorter than original
- Multi-script — mixed language text needs correct font fallback chains

This step depends on the quality of all previous steps. No dedicated research was needed — it uses standard browser typography.

---

## Recommended Architecture

```
                           ┌─────────────────────────────────────┐
                           │          CLIENT (Browser)           │
                           │                                     │
                           │  ┌──────────┐    ┌──────────────┐  │
                           │  │ OpenCV.js│    │ LaMa ONNX    │  │
                           │  │ (simple  │    │ (complex     │  │
                           │  │  inpaint)│    │  inpaint)    │  │
                           │  └──────────┘    └──────────────┘  │
                           │         \             /             │
                           │          v           v              │
                           │  ┌──────────────────────────┐      │
                           │  │   Canvas / PDF Renderer   │      │
                           │  │   (re-render edited text) │      │
                           │  └──────────────────────────┘      │
                           └────────────────┬────────────────────┘
                                            │ upload
                                            v
  ┌──────────────────────────────────────────────────────────────┐
  │                      SERVER (Docker)                         │
  │                                                              │
  │  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
  │  │  PaddleOCR   │  │ Font Detect  │  │  LaMa / GaRNet   │  │
  │  │  PP-Struct   │  │ (Storia +    │  │  (server-side     │  │
  │  │  V3          │  │  SWT + bbox) │  │   inpainting)     │  │
  │  │              │  │              │  │                   │  │
  │  │ OCR + Layout │  │ Font family, │  │  Complex bg text  │  │
  │  │ + Reading    │  │ size, weight │  │  removal          │  │
  │  │   order      │  │              │  │                   │  │
  │  └──────┬───────┘  └──────┬───────┘  └───────┬───────────┘  │
  │         └─────────────────┴──────────────────┘              │
  │                           │                                  │
  │                    OcrResult + FontMetadata                  │
  │                    + InpaintedBackground                     │
  └──────────────────────────────────────────────────────────────┘
```

**Hybrid strategy**: simple inpainting (white/solid backgrounds) runs client-side for instant feedback. Complex inpainting and all OCR/font detection runs server-side.

---

## Technology Stack

| Step | Primary tool | Fallback | License | Runs on |
|------|-------------|----------|---------|---------|
| OCR + Layout | PaddleOCR PP-StructureV3 | Tesseract (CPU) | Apache-2.0 | Server (GPU) |
| Font size | bbox formula (implemented) | — | — | Server |
| Font family | Storia AI ONNX | Coarse serif/sans/mono | Apache-2.0 | Server (CPU) |
| Font weight/style | SWT + ink density | — | BSD | Server (CPU) |
| Simple inpainting | OpenCV Telea | — | BSD | Client (WASM) |
| Complex inpainting | LaMa | GaRNet | Apache-2.0 | Server (GPU) or Client (ONNX) |
| PDF text removal | PyMuPDF redaction | — | AGPL-3.0 | Server |
| Text rendering | CSS + Google Fonts | — | — | Client |

---

## Cost Estimate: 500,000 Pages/Month

### OCR + Layout (the dominant cost)

| Configuration | Monthly cost | Notes |
|---------------|-------------|-------|
| PaddleOCR on CPU (c6i.large) | **$14** | ~3K pages/hr, need ~170 hrs |
| PaddleOCR on GPU (g4dn.xlarge) | **$158** | ~20K pages/hr, need ~25 hrs. On-demand. |
| PaddleOCR on GPU (g4dn.xlarge spot) | **~$50-80** | ~60-70% discount, interruption risk |
| docTR on GPU (g4dn.xlarge) | **$105** | ~30K pages/hr, best throughput |
| AWS Textract (text only) | **$750** | Managed, no infra |
| AWS Textract (tables+forms) | **$7,500** | Managed, no infra |

### Font Detection (marginal cost)

| Component | Monthly cost |
|-----------|-------------|
| Storia ONNX on CPU | ~$0 (runs on same OCR instance, <5ms/block) |
| SWT + ink density | ~$0 (CPU, negligible) |
| **Subtotal** | **$0** (bundled with OCR instance) |

### Inpainting (per edited page, not per scanned page)

Inpainting only runs on pages the user actually edits. Assuming 5% of scanned pages are edited (25K pages/month):

| Configuration | Monthly cost |
|---------------|-------------|
| LaMa on GPU (g4dn.xlarge) | **$3-5** (~50ms/page, need ~0.35 hrs) |
| OpenCV Telea only | **$0** (client-side or negligible CPU) |
| **Subtotal** | **$3-5** |

### Total Estimated Monthly Cost

```
  ┌──────────────────────────────────────────┐
  │  500K pages/month — self-hosted          │
  │                                          │
  │  OCR (GPU, spot instances)    $50-80     │
  │  Font detection (CPU)         $0         │
  │  Inpainting (GPU)             $3-5       │
  │  ECR storage + misc           $5-10      │
  │  ────────────────────────────────        │
  │  TOTAL (spot)                 $58-95     │
  │  TOTAL (on-demand)            $166-173   │
  │                                          │
  │  Compare: Textract text-only  $750       │
  │  Compare: Textract tables     $7,500     │
  └──────────────────────────────────────────┘
```

**Cost is dominated by OCR**. Font detection and inpainting are negligible. Using GPU spot instances brings the total under $100/month for 500K pages — **7-75x cheaper than Textract** depending on feature set.

---

## Implementation Phases

### Phase 1: OCR + Basic Editing (4-6 weeks)

Delivers: user uploads a scan, sees editable text blocks, can modify text content.

| Task | Effort | Details |
|------|--------|---------|
| Deploy PaddleOCR PP-StructureV3 | 1 week | Docker, GPU, FastAPI wrapper |
| Adapter layer (OcrResult schema) | 1 week | Already prototyped in benchmark |
| Editor integration (text overlay) | 2-3 weeks | Render text blocks over scanned image |
| Basic inpainting (OpenCV Telea) | 2-3 days | White background text erasure |

### Phase 2: Font Matching + Quality Inpainting (3-4 weeks)

Delivers: edited text visually matches the original font. Complex backgrounds handled.

| Task | Effort | Details |
|------|--------|---------|
| Font size from bbox | Done | Already implemented |
| Storia ONNX font family | 2-3 days | CPU-only, ONNX runtime |
| Bold/italic SWT detection | 3-5 days | Rule-based, no training |
| LaMa inpainting service | 1 week | Docker, GPU, ~50ms/page |
| Mask refinement (DBNet) | 3-5 days | Pixel-level text masks |

### Phase 3: Layout + Reading Order (3-4 weeks)

Delivers: correct paragraph grouping, multi-column support, table editing.

| Task | Effort | Details |
|------|--------|---------|
| PP-StructureV3 layout integration | 1-2 weeks | Reading order pointer network |
| Multi-column reconstruction | 1 week | Post-processing on block order |
| Table structure extraction | 1 week | HTML → structured cells |
| Accuracy validation (OmniDocBench) | 3-5 days | CER/WER + reading order NED |

### Phase 4: Client-Side + Polish (2-4 weeks)

Delivers: instant preview editing, browser-side inpainting for simple cases.

| Task | Effort | Details |
|------|--------|---------|
| OpenCV.js client inpainting | 3-5 days | 5.3 MB WASM, white backgrounds |
| LaMa ONNX browser fallback | 1 week | 208 MB cacheable, moderate backgrounds |
| PDF export (PyMuPDF) | 1 week | Re-render as searchable PDF |
| End-to-end quality gate | 1 week | Ghost-text detection, visual regression |

---

## Key Risks

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Surya GPL-3.0 license | Can't distribute | Use PaddleOCR/docTR (Apache-2.0) instead |
| Layout reconstruction errors | Wrong paragraph grouping | PP-StructureV3 pointer network; fall back to spatial heuristics |
| Inpainting ghost artifacts | Visible remnants of erased text | LaMa + verification loop; GaRNet fallback |
| Font mismatch | Re-rendered text looks wrong | Google Fonts fallback chain; user can manually select font |
| GPU memory contention | OOM when running OCR + inpainting simultaneously | Separate GPU queues or sequential processing |
| Scanned PDF text layer reuse | Embedded layers unreliable | Our 7 test PDFs had zero text layers; always OCR fresh |

---

## Documents Index

### Architecture & Design

| Document | Description |
|----------|-------------|
| [DR-001-ocr-adapter.md](DR-001-ocr-adapter.md) | Adapter layer design, canonical data model, provider selection rationale |
| [comparison.md](comparison.md) | Feature matrix: 5 engines x 30+ capabilities |
| [output-structures.md](output-structures.md) | Raw output examples from each engine with adapter notes |
| [cost-performance.md](cost-performance.md) | Throughput benchmarks and cost models by volume and hardware |
| [tech-overview.md](tech-overview.md) | Benchmark platform architecture and deployment notes |

### Research

| Document | Description |
|----------|-------------|
| [research-font-estimation.md](research-font-estimation.md) | Font size/family/weight detection: models, accuracy, pipeline |
| [research-inpainting.md](research-inpainting.md) | Text erasure: LaMa, GaRNet, STR models, mask generation, client-side |
| [research-benchmark-gaps.md](research-benchmark-gaps.md) | Layout evaluation, bbox IoU, browser OCR, ground-truth datasets |
