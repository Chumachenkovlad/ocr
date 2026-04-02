# OCR Benchmark — Overview

> **Last run**: 2026-04-02 10:24 UTC | **DPI**: 300 | **Runs per fixture**: 3
> **Providers**: doctr, paddle, surya, tesseract | **Fixtures**: 10

---

## Winners

| Category | Winner | Value |
|----------|--------|-------|
| **Fastest** (mean latency) | tesseract | 2922 ms |
| **Most words extracted** | surya | 172 words avg |
| **Highest confidence** | paddle | 94.0% avg |

## Provider Summary

| Provider | Mean latency | Mean words | Avg confidence | Errors |
|----------|-------------|------------|----------------|--------|
| **tesseract** | 2922 ms | 159 | 86.8% | 0 |
| **surya** | 3063 ms | 172 | 85.1% | 6 |
| **doctr** | 3175 ms | 119 | 78.4% | 0 |
| **paddle** | 4743 ms | 155 | 94.0% | 0 |

## Reports

| Report | Description |
|--------|-------------|
| [Full Analysis Report](benchmark-report.md) | Latency, word counts, confidence distributions, CER/WER accuracy, text similarity matrix, per-doc-type breakdown, and conclusion |
| [Resource Consumption](resource-report.md) | CPU, memory, and GPU utilization per service at steady-state load. Instance sizing recommendations |
| [Saturation Benchmark](saturation-report.md) | RPS vs concurrency sweep (1-16). Throughput curves, GPU utilization under load, queue saturation points |
| [Visual Accuracy Report](visual-report.html) | Side-by-side OCR output overlaid on source documents (HTML — contains embedded images) |
