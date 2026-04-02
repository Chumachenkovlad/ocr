# OCR integration — documentation

This folder documents the OCR provider evaluation, adapter design, and provider selection for the project.

## Contents

| File | Description |
|------|-------------|
| [DR-001-ocr-adapter.md](./DR-001-ocr-adapter.md) | Design record — adapter architecture and provider selection |
| [comparison.md](./comparison.md) | Full feature matrix across all evaluated providers |
| [output-structures.md](./output-structures.md) | Actual output shapes from each provider with annotated pain points |
| [cost-performance.md](./cost-performance.md) | Cost model and throughput estimates by volume and hardware |

## Quick summary

Three providers selected for the initial experiment:

| Provider | Role | License |
|----------|------|---------|
| **Surya** | Open-source primary | GPL-3.0 |
| **docTR** | Open-source alternative | Apache 2.0 |
| **AWS Textract** | Cloud reference (tables + forms) | Commercial |

All three sit behind a single `OcrAdapter` interface — see [DR-001](./DR-001-ocr-adapter.md) for the canonical data model and adapter contract.

## Status

| Date | Status |
|------|--------|
| 2026-03-30 | Draft — awaiting legal sign-off on Surya GPL-3.0 |

## Open questions

1. Is GPL-3.0 acceptable for our distribution model? → Legal
2. Do we have GPU in self-hosted deployment targets? → Infra
3. What document types are the primary use case? → Product
4. Should the adapter interface be sync or async-first? → Arch

See [DR-001](./DR-001-ocr-adapter.md#open-questions) for full tracking.
