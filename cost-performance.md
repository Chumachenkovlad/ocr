# OCR cost and performance

Cost model and throughput estimates for the full editing pipeline: OCR, font detection, and inpainting.

Related: [DR-001-ocr-adapter.md](./DR-001-ocr-adapter.md) · [comparison.md](./comparison.md) · [research-font-estimation.md](./research-font-estimation.md) · [research-inpainting.md](./research-inpainting.md)

> **Updated 2026-04-02:** Added font detection and inpainting cost estimates. Added full pipeline total at 500K pages/month. PaddleOCR now primary per [DR-001](./DR-001-ocr-adapter.md#decision-log).

---

## Assumptions

| Parameter | Value |
|-----------|-------|
| CPU instance | AWS c6i.large — $0.085/hr |
| GPU instance | AWS g4dn.xlarge (T4) — $0.526/hr |
| Infra overhead | +20% (monitoring, storage I/O, headroom) |
| Textract text-only | $1.50 per 1,000 pages |
| Textract tables + forms | $15.00 per 1,000 pages |
| Billing model (self-hosted) | Batch — instance runs only while processing |

Costs do not include storage, egress, or engineering time. All prices as of Q1 2026.

---

## Throughput estimates

Benchmarked on standard single-page A4 documents at 300 DPI. Complex documents (multi-column, tables, dense images) will be slower.

### CPU (c6i.large)

| Provider | Sec / page | Pages / hour | Notes |
|----------|-----------|-------------|-------|
| Tesseract | ~2.0s | ~1,500 | No GPU path; consistent across hardware |
| PaddleOCR | ~0.9s | ~3,000 | CPU inference with ONNX backend |
| docTR ✦ | ~0.8s | ~2,000 | PyTorch CPU, single-thread |
| Surya ✦ | ~0.6s | ~2,500 | Faster than docTR on CPU due to lighter detection model |
| AWS Textract ✦ | ~1.8s | ~2,000 | Network + processing; async PDF adds 3–8s overhead |

### GPU (g4dn.xlarge — T4)

| Provider | Sec / page | Pages / hour | Speedup vs CPU |
|----------|-----------|-------------|---------------|
| Tesseract | ~2.0s | ~1,500 | 1× (no GPU path) |
| PaddleOCR | ~0.18s | ~20,000 | ~5× |
| docTR ✦ | ~0.08s | ~30,000 | ~10× |
| Surya ✦ | ~0.12s | ~20,000 | ~4× |
| AWS Textract ✦ | ~1.8s | ~2,000 | 1× (cloud-side; hardware not user-controlled) |

**docTR on GPU is the throughput leader** at ~30,000 pages/hour. Surya and PaddleOCR are comparable at ~20,000/hr. Tesseract and Textract are hardware-invariant.

---

## Cost by volume

### Self-hosted — CPU (c6i.large at $0.085/hr + 20% overhead)

| Pages / month | Tesseract | PaddleOCR | docTR ✦ | Surya ✦ |
|---------------|-----------|-----------|---------|---------|
| 10,000 | $0.57 | $0.28 | $0.43 | $0.34 |
| 50,000 | $2.83 | $1.42 | $2.13 | $1.70 |
| 100,000 | $5.67 | $2.83 | $4.25 | $3.40 |
| 500,000 | $28.33 | $14.17 | $21.25 | $17.00 |
| 1,000,000 | $56.67 | $28.33 | $42.50 | $34.00 |

### Self-hosted — GPU (g4dn.xlarge at $0.526/hr + 20% overhead)

GPU is not economical at low volumes. Break-even vs CPU is around 30,000–40,000 pages/month depending on provider.

| Pages / month | PaddleOCR | docTR ✦ | Surya ✦ |
|---------------|-----------|---------|---------|
| 10,000 | $3.16 | $2.11 | $3.16 |
| 50,000 | $15.78 | $10.52 | $15.78 |
| 100,000 | $31.55 | $21.04 | $31.55 |
| 500,000 | $157.75 | $105.17 | $157.75 |
| 1,000,000 | $315.50 | $210.34 | $315.50 |

### AWS Textract ✦ — cloud pricing

| Pages / month | Text only ($1.50/1k) | Tables + forms ($15.00/1k) |
|---------------|---------------------|---------------------------|
| 10,000 | $15.00 | $150.00 |
| 50,000 | $75.00 | $750.00 |
| 100,000 | $150.00 | $1,500.00 |
| 500,000 | $750.00 | $7,500.00 |
| 1,000,000 | $1,500.00 | $15,000.00 |

Note: Textract pricing tiers exist above 1M pages/month. First 1M pages billed at the rates above.

---

## Cost comparison at key volume thresholds

### 10,000 pages/month

At low volume, engineering time dominates — cloud is cheaper total cost of ownership.

| Provider | Monthly infra cost | Relative |
|----------|--------------------|---------|
| Textract (text) | $15.00 | Cloud convenience |
| Surya CPU | $0.34 | ~44× cheaper |
| docTR CPU | $0.43 | ~35× cheaper |
| Textract (tables) | $150.00 | 10× more than text |

### 50,000 pages/month

| Provider | Monthly infra cost |
|----------|--------------------|
| Surya CPU | $1.70 |
| PaddleOCR CPU | $1.42 |
| docTR CPU | $2.13 |
| Textract (text) | $75.00 |
| Textract (tables) | $750.00 |
| docTR GPU | $10.52 |

Self-hosted CPU is clearly cheaper at this volume. GPU only makes sense here if latency is a hard requirement.

### 500,000 pages/month

| Provider | Monthly infra cost |
|----------|--------------------|
| Surya CPU | $17.00 |
| docTR CPU | $21.25 |
| docTR GPU | $105.17 |
| Textract (text) | $750.00 |
| Textract (tables) | $7,500.00 |

At 500k+ pages/month, self-hosted is the only viable path for table-heavy workloads. Textract tables+forms at $7,500/month becomes a budget line item requiring explicit approval.

---

## Decision framework

```
Need structured tables or forms?
├── Yes → Start with Textract for accuracy baseline
│          Switch to self-hosted (docTR + layout model) once
│          volume exceeds ~20k pages/month
└── No  → Self-hosted from day one
           ├── GPU available in target infra? → docTR (best throughput)
           └── CPU only?                      → Surya (fastest per-page on CPU)

Need multilingual beyond 20 languages?
└── docTR may be insufficient → evaluate Surya (90+) or PaddleOCR (80+)

Need air-gap / strict data residency?
└── Textract is not viable → Surya or docTR
```

---

## Latency profiles

End-to-end latency for a single page (not batch throughput):

| Provider | p50 | p95 | Notes |
|----------|-----|-----|-------|
| Tesseract (CPU) | 2.0s | 3.5s | Depends heavily on image size |
| PaddleOCR (GPU) | 0.2s | 0.5s | |
| docTR (GPU) | 0.1s | 0.3s | |
| Surya (GPU) | 0.15s | 0.4s | |
| Textract (sync, single page) | 1.5s | 3.0s | Network + queue |
| Textract (async, PDF) | 5–15s | 20–30s | Polling overhead; varies with AWS load |

For latency-sensitive synchronous use cases (e.g. real-time document preview), self-hosted GPU is the only option that meets sub-second requirements.

---

## Infrastructure notes

### Self-hosted scaling

- All three open-source picks support Docker — straightforward to containerise and run behind a queue worker.
- GPU instances should be provisioned on-demand (spot or scheduled) for batch workloads — not always-on.
- Surya and docTR both support batching multiple images in a single inference call — significant throughput improvement over single-image loops.
- Model weights are downloaded on first run (1–4 GB depending on variant) — bake into the Docker image for production.

### Textract async flow

```
1. POST  start_document_analysis   →  JobId
2. POLL  get_document_analysis     →  until JobStatus == SUCCEEDED
3. GET   get_document_analysis     →  paginate with NextToken until no more pages
```

Poll interval: 2–5 seconds. Typical PDF completion: 5–15 seconds for < 10 pages.

---

## Full editing pipeline cost (beyond OCR)

The document editing pipeline adds two steps after OCR: font detection and background inpainting. Both are lightweight relative to OCR itself.

### Font detection

Runs on the same instance as OCR. Negligible marginal cost.

| Component | Method | Latency per block | Hardware | Notes |
|-----------|--------|-------------------|----------|-------|
| Font size | bbox height / DPI * 72 | <1ms | CPU | Already implemented |
| Font family | Storia AI ONNX classifier | ~5ms | CPU | 3K Google Fonts, 174 MB model |
| Serif/sans/mono | ResNet-18 coarse classifier | ~3ms | CPU | ~95% accuracy |
| Bold / italic | Stroke Width Transform | ~2ms | CPU | Rule-based, no model |

Total per page (~20 blocks): **~200ms CPU**. At 500K pages/month this is ~28 compute-hours — fits within the OCR instance idle time.

**Monthly cost: $0** (bundled with OCR compute)

See [research-font-estimation.md](./research-font-estimation.md) for model details and architecture.

### Inpainting (text removal)

Only runs on pages the user actually edits, not on every scanned page. Assuming 5% edit rate (25K pages/month of 500K scanned):

| Method | Use case | Latency per page | Hardware | Model size |
|--------|----------|-------------------|----------|------------|
| OpenCV Telea | White / solid backgrounds | <10ms | CPU | 0 (built-in) |
| LaMa | Paper texture, moderate bg | ~50ms | GPU | 174 MB |
| GaRNet | Complex backgrounds (fallback) | ~80ms | GPU | ~200 MB |

At 25K edited pages/month with LaMa on GPU: ~0.35 GPU-hours.

**Monthly cost: $3–5** (g4dn.xlarge on-demand) or **$0** (fits within OCR instance idle time)

See [research-inpainting.md](./research-inpainting.md) for model comparison and pipeline architecture.

---

## Full pipeline cost at 500K pages/month

| Component | Configuration | Monthly cost |
|-----------|--------------|-------------|
| **OCR + layout** | PaddleOCR GPU (g4dn.xlarge spot) | $50–80 |
| **OCR + layout** | PaddleOCR GPU (g4dn.xlarge on-demand) | $158 |
| **Font detection** | CPU (same instance) | $0 |
| **Inpainting** | LaMa GPU (5% edit rate) | $3–5 |
| **Infra overhead** | ECR, storage, monitoring | $5–10 |

```
  Total (spot instances):      $58–95 /month
  Total (on-demand):           $166–173 /month

  Compare:
    Textract text-only:        $750 /month    (5–13x more)
    Textract tables+forms:     $7,500 /month  (50–130x more)
```

**OCR dominates the cost.** Font detection and inpainting combined are <5% of the total. The choice between spot and on-demand instances is the biggest cost lever.
