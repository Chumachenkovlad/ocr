# OCR Resource Consumption Report

> **Instance**: g5.2xlarge (A10G GPU, 24 GB VRAM) | **Cost**: $1.212/hr
> **Generated**: 2026-03-31 20:11 UTC

---

## GPU VRAM Budget

| Service | VRAM peak | VRAM idle |
|---------|----------|----------|
| paddle | 8203 MB | 8203 MB |
| tesseract | 1 MB | 0 MB |
| surya | 2629 MB | 2629 MB |
| doctr | 1 MB | 1 MB |

**Total peak (all services)**: 10834 MB / 23028 MB (OK)

---

## Resource Profiles (Isolated)

| Service | VRAM Idle | VRAM Loaded | VRAM Peak | RAM Peak | CPU Peak | GPU Util Peak | Warmup |
|---------|----------|------------|----------|---------|---------|--------------|--------|
| paddle | 515 MB | 8203 MB | 8203 MB | 1377 MB | 106% | 100% | 2.2s |
| tesseract | 0 MB | 0 MB | 1 MB | 221 MB | 218% | 0% | 2.4s |
| surya | 1307 MB | 2629 MB | 2629 MB | 1560 MB | 118% | 62% | 2.7s |
| doctr | 1 MB | 1 MB | 1 MB | 1002 MB | 387% | 0% | 2.9s |

---

## Throughput Under Concurrency

### Pages per Minute

| Service | C=1 | C=2 | C=4 |
|---------|-----|-----|-----|
| doctr | 22.6 | 26.1 | 26.0 |
| paddle | 16.1 | 26.0 | 31.9 |
| surya | 27.0 | 31.8 | 31.7 |
| tesseract | 27.7 | 48.3 | 71.7 |

### Latency (ms)

| Service | Concurrency | p50 | p95 | p99 | Errors |
|---------|------------|-----|-----|-----|--------|
| paddle | 1 | 2759 | 10383 | 10383 | 0 |
| paddle | 2 | 2762 | 12229 | 12229 | 0 |
| paddle | 4 | 5387 | 13840 | 13840 | 1 |
| tesseract | 1 | 1414 | 8594 | 8594 | 0 |
| tesseract | 2 | 1459 | 8787 | 8787 | 0 |
| tesseract | 4 | 1892 | 8662 | 8662 | 0 |
| surya | 1 | 1972 | 4610 | 4610 | 0 |
| surya | 2 | 3128 | 10411 | 10411 | 0 |
| surya | 4 | 5652 | 24638 | 24638 | 0 |
| doctr | 1 | 2286 | 4426 | 4426 | 0 |
| doctr | 2 | 4172 | 11378 | 11378 | 0 |
| doctr | 4 | 7311 | 28409 | 28409 | 0 |

### Cost per 1000 Pages (C=1)

| Service | Pages/min | Words/sec | $/1000 pages |
|---------|----------|----------|-------------|
| doctr | 22.6 | 34.9 | $0.89 |
| paddle | 16.1 | 15.8 | $1.26 |
| surya | 27.0 | 73.9 | $0.75 |
| tesseract | 27.7 | 63.5 | $0.73 |

---

## Production Pricing: 500,000 Pages/Month

> Based on g5.2xlarge on-demand pricing: $1.212/hr. Assumes single-service per instance, C=1 sequential processing.

| Service | Pages/min (C=1) | Hours for 500K | Instances needed | Monthly cost (g5.2xlarge) | Cost on g4dn.xlarge | VRAM needed | Recommended |
|---------|----------------|---------------|-----------------|--------------------------|--------------------|-----------  |-------------|
| doctr | 22.6 | 369 | 0.51 | $447 | $194 | 1 MB | g4dn.xlarge (T4) — **$194/mo** |
| paddle | 16.1 | 518 | 0.71 | $627 | $272 | 8203 MB | g4dn.xlarge (T4) — **$272/mo** |
| surya | 27.0 | 309 | 0.42 | $374 | $162 | 2629 MB | g4dn.xlarge (T4) — **$162/mo** |
| tesseract | 27.7 | 301 | 0.41 | $365 | $158 | 1 MB | c5.2xlarge (CPU) — **$102/mo** |

### Recommendation for 500K pages/month

| Option | Service | Instance | Count | Monthly Cost | Notes |
|--------|---------|----------|-------|-------------|-------|
| **Cheapest** | tesseract | c5.2xlarge | 0.41 | **$102/mo** | CPU-only, fastest, no GPU needed |
| **Best quality** | surya | g4dn.xlarge | 0.42 | **$162/mo** | Best word recall, VRAM: ~2.6 GB (fits T4) |
| **Best tables** | paddle | g5.2xlarge | 0.71 | **$627/mo** | Best for structured docs, VRAM: ~8 GB (needs A10G) |
