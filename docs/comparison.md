# OCR provider comparison

Full feature matrix across all evaluated providers.

Related: [DR-001-ocr-adapter.md](./DR-001-ocr-adapter.md) · [output-structures.md](./output-structures.md) · [cost-performance.md](./cost-performance.md)

> **Updated 2026-04-02:** Selections revised after benchmarking. PaddleOCR is now the primary pick for the editing pipeline (layout + reading order). See [DR-001 decision log](./DR-001-ocr-adapter.md#decision-log).

---

## Legend

| Symbol | Meaning |
|--------|---------|
| ✓ | Yes / fully supported |
| ~ | Partial / limited |
| ✗ | No |
| ✦ | Selected — primary |
| ★ | Selected — fallback / secondary |

---

## Input handling

| Aspect | Tesseract ★ | PaddleOCR ✦ | docTR ★ | Surya ★ | Textract |
|--------|-----------|-----------|---------|---------|------------|
| PDF input | ~ | ✓ | ✓ | ✓ | ✓ |
| Multi-page docs | ✓ | ✓ | ✓ | ✓ | ✓ |
| Low DPI / noisy scan | ~ | ✓ | ✓ | ✓ | ✓ |
| Handwriting | ✗ | ✓ | ~ | ~ | ✓ |
| Mixed print + handwriting | ✗ | ~ | ✗ | ✗ | ✓ |

---

## Language and script

| Aspect | Tesseract ★ | PaddleOCR ✦ | docTR ★ | Surya ★ | Textract |
|--------|-----------|-----------|---------|---------|------------|
| Language count | 100+ | 80+ | 20+ | 90+ | ~12 |
| RTL (Arabic, Hebrew) | ✓ | ~ | ✗ | ✓ | ✗ |
| CJK support | ✓ | ✓ | ✗ | ✓ | ✗ |
| Math / LaTeX | ✗ | ~ | ✗ | ~ | ✗ |
| Mixed-language documents | ~ | ✓ | ✗ | ✓ | ✗ |
| Historical / rare scripts | ✗ | ✗ | ✗ | ✗ | ✗ |

---

## Document understanding

| Aspect | Tesseract ★ | PaddleOCR ✦ | docTR ★ | Surya ★ | Textract |
|--------|-----------|-----------|---------|---------|------------|
| Layout detection | ~ | ✓ | ✓ | ✓ | ✓ |
| Reading order reconstruction | ~ | ✓ | ✓ | ✓ | ✓ |
| Table detection | ✗ | ✓ | ~ | ~ | ✓ |
| Form / key-value extraction | ✗ | ~ | ~ | ~ | ✓ |
| Checkbox detection | ✗ | ✗ | ✗ | ✗ | ✓ |
| Header / footer awareness | ✗ | ~ | ✓ | ✓ | ✓ |
| Figure region detection | ✗ | ~ | ✓ | ✓ | ✓ |
| Page segmentation | ~ | ✓ | ✓ | ✓ | ✓ |

---

## Output quality and format

| Aspect | Tesseract ★ | PaddleOCR ✦ | docTR ★ | Surya ★ | Textract |
|--------|-----------|-----------|---------|---------|------------|
| Confidence scores | ✓ | ✓ | ✓ | ✓ | ✓ |
| Bounding boxes | ✓ | ✓ | ✓ | ✓ | ✓ |
| Word-level bounding box | ✓ | ✗ | ✓ | ✗ | ✓ |
| Font / style detection | ✗ | ✗ | ✗ | ✗ | ✗ |
| Plain text | ✓ | ✓ | ✓ | ✓ | ✓ |
| JSON / structured output | ~ | ✓ | ✓ | ✓ | ✓ |
| hOCR / ALTO XML | ✓ | ✗ | ~ | ~ | ✗ |
| Searchable PDF generation | ✓ | ~ | ✓ | ✓ | ✗ |
| Table as structured cells | ✗ | ✗ | ✗ | ✗ | ✓ |
| Markdown output | ✗ | ✗ | ~ | ✓ | ✗ |

---

## Technical stack

| Aspect | Tesseract ★ | PaddleOCR ✦ | docTR ★ | Surya ★ | Textract |
|--------|-----------|-----------|---------|---------|------------|
| Open source | ✓ | ✓ | ✓ | ✓ | ✗ |
| Self-hosted | ✓ | ✓ | ✓ | ✓ | ✗ |
| GPU acceleration | ✗ | ✓ | ✓ | ✓ | ✗ (cloud) |
| Docker support | ✓ | ✓ | ✓ | ✓ | ✗ |
| REST API | ✗ | ~ | ~ | ~ | ✓ |
| Python SDK | ✓ | ✓ | ✓ | ✓ | ✓ |
| Fine-tuning support | ~ | ✓ | ✓ | ✓ | ✗ |
| Custom vocabulary / dictionary | ✓ | ~ | ✗ | ✗ | ✗ |
| Adapter complexity | Medium | Medium | Medium | Low | High |

Adapter complexity is assessed relative to our canonical model — see [DR-001](./DR-001-ocr-adapter.md) and [output-structures.md](./output-structures.md) for detail.

---

## Cost and privacy

| Aspect | Tesseract ★ | PaddleOCR ✦ | docTR ★ | Surya ★ | Textract |
|--------|-----------|-----------|---------|---------|------------|
| Offline / air-gap capable | ✓ | ✓ | ✓ | ✓ | ✗ |
| License | Apache 2.0 | Apache 2.0 | Apache 2.0 | GPL-3.0 | Commercial |
| Free tier | ✓ | ✓ | ✓ | ✓ | ~ (1k pages) |
| Data leaves perimeter | ✗ | ✗ | ✗ | ✗ | ✓ |
| GDPR compliant | ✓ | ✓ | ✓ | ✓ | ~ (config dependent) |
| Domain models (invoices, etc.) | ✗ | ~ | ✗ | ✗ | ✓ |
| Pricing model | Free | Free | Free | Free | Per-page |

---

## Output shape summary

Quick reference for adapter implementation. Full annotated examples in [output-structures.md](./output-structures.md).

| Provider | Output type | Nesting depth | Bbox format | Normalised coords | Table cells | Word confidence |
|----------|------------|---------------|-------------|-------------------|-------------|-----------------|
| Tesseract | DataFrame / string | Flat (level 1–5 filter) | x, y, w, h | No | No | Per word |
| PaddleOCR | List of lists | 2 levels | Quad (4 pts) | No | HTML string | Per line |
| docTR ★ | Nested dict | 4 levels | `[[x1,y1],[x2,y2]]` | Yes (0–1) | No | Per word |
| Surya ★ | Custom objects | 2 levels | Rect `[x1,y1,x2,y2]` | No | Separate step | Per line |
| Textract | Flat Block list + ID refs | Flat + graph | Normalised rect | Yes (0–1) | CELL blocks | Per word |
