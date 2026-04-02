# OCR Comparison Platform — Technical Overview

## Goal

Evaluate self-hosted OCR engines under real-world conditions to determine the best fit for a production document editing pipeline. The comparison focuses on accuracy, latency, resource consumption, layout analysis, and cost across different document types.

## Service Selection

After an initial research phase covering five candidates (Tesseract, PaddleOCR, docTR, Surya, AWS Textract), four self-hosted engines were benchmarked:

| Service | Framework | GPU Support | Key Strength | Role (post-benchmark) |
|---------|-----------|-------------|--------------|----------------------|
| **PaddleOCR** | PaddlePaddle | Yes | Layout analysis, table detection, reading order | **Primary** |
| **Tesseract** | C++ / hOCR | No | Mature, lightweight, fast on CPU | CPU fallback |
| **docTR** | PyTorch | Yes | Best GPU throughput (~30k pages/hr on T4) | Batch throughput |
| **Surya** | PyTorch | Yes | 90+ languages, good recall | Multilingual (GPL-3.0 risk) |

AWS Textract was excluded from the live benchmark — managed cloud service, cannot run on shared infrastructure. Cost estimates documented in [cost-performance.md](./cost-performance.md).

## Architecture

Python monorepo with a unified REST interface:

```
ocr/
├── services/
│   ├── paddle/       # PaddleOCR wrapper (FastAPI) — primary
│   ├── tesseract/    # Tesseract + hOCR parser (FastAPI) — CPU fallback
│   ├── surya/        # Surya wrapper (FastAPI) — multilingual
│   ├── doctr/        # docTR wrapper (FastAPI) — throughput
│   └── gateway/      # Orchestrator — fans out to all engines
├── shared/           # Pydantic v2 unified output schema (OcrResult)
├── benchmark/        # Benchmark runner, analysis, report generation
├── test-fixtures/    # Shared test images and PDFs (10 synthetic + 7 real PDFs)
├── infra/            # EC2 deploy/teardown scripts
└── docker-compose.yml
```

Each service exposes `POST /api/v1/ocr` and `GET /healthz`. The gateway provides `POST /api/v1/compare` which fans out to all engines in parallel and returns results in the shared `OcrResult` schema.

## Deployment

Benchmarks run on a single **g4dn.2xlarge** EC2 instance (1x T4 GPU, 8 vCPU, 32 GB RAM) in `us-east-1`. Infrastructure is ephemeral — spun up for benchmarking, torn down after. ECR images retained for reuse.

Total resource allocation: ~29 GB RAM across all containers (Paddle 12G, Surya 8G, docTR 6G, Tesseract 2G, Gateway 0.5G).

**Known issue:** Surya OOMs on multi-page PDFs when GPU is shared with PaddleOCR + docTR (T4 has 16 GB VRAM, all three GPU models compete). Workaround: run Surya benchmarks separately or use g5.2xlarge (24 GB).

## Benchmarking

Three benchmark types conducted against 10 test fixtures (forms, tables, noisy scans, multilingual text, multi-page PDFs):

1. **Accuracy & latency** (`benchmark/run.py`) — per-fixture latency, word/line/block counts, confidence, CER/WER via majority-vote pseudo-ground-truth, pairwise text similarity matrix
2. **Resource consumption** (`benchmark/resource_bench.py`) — CPU/memory/GPU utilization under load
3. **Saturation** (`benchmark/saturation_bench.py`) — throughput ceiling under concurrent requests (C=1 to C=16)

Additional analysis:
- **Text layer extraction** (`benchmark/text_layer.py`) — checks for embedded OCR text layers in PDF fixtures. Finding: all 7 real PDFs are pure image scans with zero text layers.
- **Visual accuracy** (`benchmark/visualize.py`) — bbox overlay images for visual inspection

## Results

Published as interactive HTML reports on GitHub Pages:

| Report | What it shows |
|--------|---------------|
| [Overview](https://chumachenkovlad.github.io/ocr/) | Summary cards, provider metrics table |
| [Full Report](https://chumachenkovlad.github.io/ocr/report.html) | Latency, word counts, confidence, **CER/WER accuracy**, text similarity matrix |
| [Resource Report](https://chumachenkovlad.github.io/ocr/resource-report.html) | CPU, memory, GPU utilization per service |
| [Saturation Report](https://chumachenkovlad.github.io/ocr/saturation-report.html) | Throughput and error rates under concurrency |
| [Visual Report](https://chumachenkovlad.github.io/ocr/visual-report.html) | Side-by-side bbox overlays on real scanned PDFs |

## Key Findings

- **Layout analysis**: PaddleOCR (PP-StructureV3) is the only engine with native reading order recovery and block type classification. Critical for the document editing use case.
- **Accuracy (CER/WER)**: Surya and Tesseract have the lowest word error rates on average. PaddleOCR is close behind. docTR underperforms on dense layouts.
- **Latency**: Tesseract and docTR are fastest. PaddleOCR is moderate. Surya is comparable on GPU but OOMs on complex fixtures.
- **Confidence**: PaddleOCR reports the highest confidence. Tesseract is most consistent across fixture types.
- **Resources**: Tesseract is lightest (2 GB, CPU-only). GPU engines need 6-12 GB RAM.
- **Text layers**: No embedded OCR layers found in any real-world PDF fixture — fresh OCR is always required.

## Related Research

Beyond OCR engine selection, three adjacent problems were researched for the editing pipeline:

| Document | Topic | Key finding |
|----------|-------|-------------|
| [research-font-estimation.md](./research-font-estimation.md) | Font family, size, weight detection | Storia AI ONNX model (3K Google Fonts, CPU, Apache-2.0) is the best available |
| [research-inpainting.md](./research-inpainting.md) | Text erasure / background reconstruction | LaMa (~50ms GPU, Apache-2.0) for general use; OpenCV Telea (<10ms) for white backgrounds |
| [research-benchmark-gaps.md](./research-benchmark-gaps.md) | Layout evaluation, bbox IoU, browser OCR, ground-truth datasets | OmniDocBench for reading order; synthetic fixtures for bbox ground truth |

## Full Pipeline

See [README.md](./README.md) for the complete editing pipeline architecture, cost estimate (500K pages/month), implementation phases, and risk matrix.
