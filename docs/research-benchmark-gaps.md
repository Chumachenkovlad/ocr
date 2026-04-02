# Research: OCR Benchmark Gaps

> **Date**: 2026-04-02
> **Context**: FakeEdit v2 document editing use case. Current benchmarks cover latency, resource usage, throughput, and word extraction. This document covers five identified gaps with concrete, actionable recommendations for each.
>
> **Pipeline step**: Cross-cutting — covers evaluation gaps that affect multiple pipeline steps in the [editing pipeline](./README.md). Layout/reading order (Problem 2), bbox accuracy (Problem 1), browser-side OCR (architectural alternative).
>
> **Related**: [DR-001-ocr-adapter.md](./DR-001-ocr-adapter.md) (engine selection these gaps inform) · [tech-overview.md](./tech-overview.md) (benchmark platform) · [research-font-estimation.md](./research-font-estimation.md) (font detection depends on bbox quality) · [research-inpainting.md](./research-inpainting.md) (mask generation depends on bbox quality)

---

## Table of Contents

1. [Layout Reconstruction and Reading Order Evaluation](#1-layout-reconstruction-and-reading-order-evaluation)
2. [Bounding Box Accuracy Measurement](#2-bounding-box-accuracy-measurement)
3. [Browser-Side OCR Feasibility](#3-browser-side-ocr-feasibility)
4. [Existing OCR Text Layer Quality](#4-existing-ocr-text-layer-quality)
5. [Ground-Truth Accuracy Datasets](#5-ground-truth-accuracy-datasets)

---

## 1. Layout Reconstruction and Reading Order Evaluation

### Problem

Multi-column documents, mixed text/table layouts, and documents with headers/footers require correct reading order reconstruction. Our current benchmark captures `block_count` and pairwise text similarity (Jaccard and SequenceMatcher) but does not evaluate whether blocks are ordered correctly.

### Metrics

Three complementary metrics are used in the literature for reading order evaluation:

| Metric | What it measures | Strengths | Weaknesses |
|--------|-----------------|-----------|------------|
| **Kendall's tau (tau-b)** | Fraction of pairwise block orderings that agree with ground truth | Captures inversions directly; well-understood statistic | Requires 1:1 block correspondence; sensitive to block segmentation differences |
| **Normalized Edit Distance (NED)** | Minimum edits to transform predicted block sequence into GT sequence | Handles insertions/deletions (block count mismatch); used by OmniDocBench | Penalizes segmentation differences, not just ordering |
| **BLEU / NLCS on linearized text** | N-gram or subsequence overlap on the full linearized text stream | End-to-end metric (captures both ordering and text accuracy); easy to compute | Conflates text recognition errors with ordering errors |

**Recommendation**: Use **Normalized Edit Distance on block sequences** as the primary reading order metric (matches OmniDocBench methodology), with **Kendall's tau** as a secondary pairwise metric. Compute both on the block ordering index arrays.

**Implementation sketch** (Python):

```python
from scipy.stats import kendalltau

def reading_order_ned(predicted_order: list[int], gt_order: list[int]) -> float:
    """Normalized edit distance between two block orderings."""
    import Levenshtein  # python-Levenshtein package
    pred_str = " ".join(str(i) for i in predicted_order)
    gt_str = " ".join(str(i) for i in gt_order)
    dist = Levenshtein.distance(pred_str, gt_str)
    max_len = max(len(pred_str), len(gt_str))
    return 1.0 - (dist / max_len) if max_len > 0 else 1.0

def reading_order_kendall(predicted_order: list[int], gt_order: list[int]) -> float:
    """Kendall's tau correlation between predicted and GT block orders."""
    tau, _ = kendalltau(predicted_order, gt_order)
    return tau  # -1 to 1; 1.0 = perfect agreement
```

### Benchmark Datasets for Reading Order

| Dataset | Size | Document types | Reading order annotations | Link |
|---------|------|---------------|--------------------------|------|
| **OmniDocBench** | 981 pages | Academic, financial, legal, patents, textbooks | Yes - block-level reading order with NED metric | [github.com/opendatalab/OmniDocBench](https://github.com/opendatalab/OmniDocBench) |
| **DocLayNet** | 80,863 pages | Finance, science, patents, tenders, law, manuals | Layout bounding boxes with 11 class labels; no explicit reading order but spatial positions enable derivation | [github.com/DS4SD/DocLayNet](https://github.com/DS4SD/DocLayNet) |
| **PubLayNet** | 1M+ pages | Scientific articles (PubMed) | Layout bounding boxes for 5 classes; reading order derivable from XML source | [github.com/ibm-aur-nlp/PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet) |
| **DocBank** | 500K pages | arXiv papers | Token-level layout labels (12 classes); limited layout diversity | [github.com/doc-analysis/DocBank](https://github.com/doc-analysis/DocBank) |
| **ReadingBank** | 500K pages | Word documents rendered to images | Explicit reading order ground truth | [github.com/microsoft/unilm/tree/master/layoutlm](https://github.com/microsoft/unilm/tree/master/layoutlm) |

**Recommendation**: Use **OmniDocBench** as the primary benchmark (directly supports NED reading order evaluation with a ready-made evaluation harness). Supplement with **DocLayNet** for layout diversity since PubLayNet and DocBank are heavily biased toward scientific articles.

### Engine Comparison: Native Layout Analysis

| Engine | Layout analysis method | Reading order capability | Multi-column support |
|--------|----------------------|------------------------|---------------------|
| **PaddleOCR (PP-StructureV3)** | PP-DocLayoutV2 detector (14 element types) + 6-layer transformer pointer network for reading order | **Best native support** - explicit reading order recovery via pointer network; outputs `parsing_res_list` with `index` field | Yes - the pointer network specifically handles multi-column reordering |
| **Tesseract (hOCR)** | Page segmentation modes (PSM 1-4); `ocr_carea` / `ocr_par` hierarchy in hOCR | Moderate - PSM 1 with OSD attempts column-aware ordering; PSM 3 (default) often reads left-to-right across columns incorrectly | Partial - requires PSM 1 and post-processing of hOCR bounding boxes |
| **Surya** | EfficientViT-based detector; line-level detection with Y-proximity block grouping | **Limited** - Surya authors explicitly state "use something like marker or other postprocessing if you need to order the text"; our adapter (`surya/adapter.py`) groups by Y-gap only | No native multi-column awareness |
| **docTR** | Two-stage detection (db_resnet50/fast_base) + recognition; block-level output | **Limited** - blocks are spatial groupings without reading order logic | No native reading order; blocks ordered by spatial position |

**Verdict**: PP-StructureV3 is the clear leader for reading order. To evaluate the others fairly, a post-processing reading order algorithm (e.g., XY-cut or column-detection heuristic) should be applied before measurement.

### Implementation Plan

1. **Phase 1**: Add OmniDocBench evaluation harness (pip install omnidocbench, configure YAML for our 4 engines)
2. **Phase 2**: Extend the benchmark runner to capture `reading_order` array from each engine's response
3. **Phase 3**: Implement Kendall's tau + NED metrics in the analysis pipeline
4. **Phase 4**: Test on `dense-twocol.png` fixture and a subset of OmniDocBench (academic + financial documents)

---

## 2. Bounding Box Accuracy Measurement

### Problem

The editor overlays transparent text on scanned page images. If bounding boxes are inaccurate, text highlights misalign, copy-paste selects wrong regions, and the editing experience degrades. We currently have no IoU measurement in the benchmark.

### IoU Computation Without Manual Annotation

The key challenge is obtaining ground truth bounding boxes without expensive manual annotation. Three approaches:

#### Approach A: Synthetic Document Generation (Recommended)

Render documents with known text at known pixel positions, then OCR and compare.

| Tool | What it generates | BBox ground truth? | Language support | Link |
|------|------------------|-------------------|-----------------|------|
| **Our existing fixture generator** | PNG/PDF fixtures using Pillow + ReportLab | **Not yet** - but positions are computable from ReportLab/Pillow draw calls | English | Custom script |
| **SynthDOG** | Wikipedia-based document images with bounding boxes | Yes - line-level and word-level bbox annotations in JSON | 6+ languages | [github.com/clovaai/donut](https://github.com/clovaai/donut) (under `synthdog/`) |
| **TextRecognitionDataGenerator (TRDG)** | Single-line text images | Word-level bbox (implicit: full image = bbox) | 20+ languages | [github.com/Belval/TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator) |
| **docTR synthetic** | Document-like page images | Normalized bbox in docTR format | Configurable | [github.com/mindee/doctr](https://github.com/mindee/doctr) (data generation utils) |
| **Genalog** | Degraded synthetic documents from text | Character and word-level annotations | English | [github.com/microsoft/genalog](https://github.com/microsoft/genalog) |

**Recommended approach**: Extend the existing fixture generator to record ground-truth bounding boxes during generation. Since we already use Pillow `ImageDraw.text()` and ReportLab `drawString()`, we can capture the exact `(x, y, width, height)` of every rendered word. This gives us a perfect ground-truth dataset matching our existing fixtures.

```python
# Sketch: extend generate.py to emit ground truth
import json
from PIL import ImageDraw, ImageFont

def draw_text_with_gt(
    draw: ImageDraw.Draw,
    font: ImageFont.FreeTypeFont,
    text: str,
    xy: tuple[int, int],
) -> dict:
    """Draw text and return ground-truth bounding box."""
    bbox = draw.textbbox(xy, text, font=font)  # (x1, y1, x2, y2)
    draw.text(xy, text, fill="black", font=font)
    return {
        "text": text,
        "bbox": {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]},
    }
# Save all word GTs to a sidecar JSON file per fixture
```

#### Approach B: Cross-Engine Consensus

Use agreement between engines as a proxy for ground truth. Where 3+ of 4 engines agree on a word's bounding box (IoU > 0.5), treat the median box as pseudo-ground-truth. Less reliable but works with real-world scanned documents.

#### Approach C: Commercial OCR as Reference

Use Google Cloud Vision or AWS Textract as a high-accuracy reference. Practical for a small validation set but adds cost and API dependency.

### IoU Computation

```python
def compute_iou(box_a: dict, box_b: dict) -> float:
    """Compute IoU between two (x1, y1, x2, y2) bounding boxes."""
    x1 = max(box_a["x1"], box_b["x1"])
    y1 = max(box_a["y1"], box_b["y1"])
    x2 = min(box_a["x2"], box_b["x2"])
    y2 = min(box_a["y2"], box_b["y2"])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a["x2"] - box_a["x1"]) * (box_a["y2"] - box_a["y1"])
    area_b = (box_b["x2"] - box_b["x1"]) * (box_b["y2"] - box_b["y1"])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0
```

### IoU Thresholds for Text Overlay in an Editor

Research on document analysis IoU thresholds:

| IoU Threshold | Use case | Notes |
|--------------|----------|-------|
| **>= 0.9** | High-precision text overlay; character-level alignment | Required for pixel-accurate text replacement in editing |
| **>= 0.75** | Good text overlay; acceptable for highlight/selection | Recommended minimum for FakeEdit v2 |
| **>= 0.5** | Standard object detection threshold (PASCAL VOC) | Too loose for text overlay - visible misalignment at edges |
| **>= 0.6** | Lower bound used in some document layout benchmarks | May cause noticeable highlight drift on small text |

**Recommendation for FakeEdit v2**: Target **IoU >= 0.75** as the minimum acceptable threshold for word-level bounding boxes. At this level, text highlights will closely match the original text regions. For the benchmark, report the distribution of IoU values (mean, p10, p50, p90) and the percentage of words above 0.75 and 0.9.

### Implementation Plan

1. Extend the fixture generator to emit `<fixture-name>.gt.json` sidecar files
2. Add a bbox evaluation script that loads GT + engine results, matches words by text, computes IoU
3. Report per-engine: mean IoU, IoU@0.75 recall, IoU@0.9 recall
4. Optionally generate SynthDOG fixtures for multi-language bbox evaluation

---

## 3. Browser-Side OCR Feasibility

### Problem

FakeEdit v2 may benefit from client-side OCR for instant feedback, offline capability, and reduced server costs. The question is what is practical in a browser today.

### tesseract.js (WASM)

| Aspect | Details |
|--------|---------|
| **Project** | [github.com/naptha/tesseract.js](https://github.com/naptha/tesseract.js) (v6.0.0) |
| **Alternative** | [github.com/robertknight/tesseract-wasm](https://github.com/robertknight/tesseract-wasm) (lighter, SIMD-optimized) |
| **Download size** | ~15 MB (tesseract.js with eng data); ~2.1 MB with Brotli (tesseract-wasm) |
| **Browser support** | All modern browsers; SIMD requires Chrome 91+, Firefox 90+, Safari 16.4+ |
| **Typical latency** | 2-8 seconds per page (single-column A4 at 300 DPI) on modern desktop; significantly slower on mobile |
| **Accuracy** | Same as native Tesseract 5.x (LSTM engine); no layout analysis in WASM builds |
| **Limitations** | No multi-column layout detection; no table recognition; single-threaded WASM execution; large memory footprint (200-400 MB at peak) |
| **Best for** | Quick single-page preview; simple single-column documents; privacy-sensitive use cases |

### ONNX Runtime Web (PaddleOCR / docTR in browser)

| Aspect | Details |
|--------|---------|
| **Runtime** | [onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web) with WebAssembly or WebGPU backends |
| **PaddleOCR.js** | [github.com/X3ZvaWQ/paddleocr.js](https://github.com/X3ZvaWQ/paddleocr.js) - PaddleOCR v5 ONNX models; minimal size; supports onnxruntime-web and onnxruntime-node |
| **@gutenye/ocr-browser** | [npm: @gutenye/ocr-browser](https://www.npmjs.com/package/@gutenye/ocr-browser) - PP-OCRv4 models in browser |
| **@paddle-js-models/ocr** | [npm: @paddle-js-models/ocr](https://www.npmjs.com/package/@paddle-js-models/ocr) - Official Paddle.js models (PP-OCRv3, compressed for JS) |
| **Pre-converted models** | [huggingface.co/monkt/paddleocr-onnx](https://huggingface.co/monkt/paddleocr-onnx), [huggingface.co/marsena/paddleocr-onnx-models](https://huggingface.co/marsena/paddleocr-onnx-models) |
| **Model size** | Detection model ~2-4 MB + recognition model ~5-10 MB (quantized PP-OCRv4) |
| **Typical latency** | ~200-500 ms per page (WASM, modern desktop); significantly faster with WebGPU where available |
| **Accuracy** | PP-OCRv3/v4 mobile models; slightly lower accuracy than full server PP-OCRv5/StructureV3 |
| **Limitations** | No layout analysis (detection + recognition only); model loading time on first use (1-3 seconds); WebGPU support still inconsistent across browsers |

### Transformers.js (Surya-like models)

| Aspect | Details |
|--------|---------|
| **Project** | [github.com/huggingface/transformers.js](https://github.com/huggingface/transformers.js) (v4 preview, Feb 2026) |
| **WebGPU support** | Yes - v4 rewrote WebGPU runtime in C++ with ONNX Runtime team; Chrome 113+, Firefox 141+, Safari 26+ |
| **OCR models available** | Donut (OCR-free document understanding); TrOCR; no official Surya ONNX export available |
| **Surya status** | Surya uses custom EfficientViT architecture; **no official ONNX export or Transformers.js support**. Would require custom model conversion and potentially architecture modifications |
| **Typical latency** | Donut: 3-10 seconds per page (WebGPU); TrOCR: 1-3 seconds per cropped text region |
| **Limitations** | Large model downloads (50-200 MB for vision-language models); Surya specifically not portable to browser; memory pressure on mobile devices |

### Latency Expectations Summary

| Solution | First load | Per-page latency (desktop) | Per-page latency (mobile) | Model download |
|----------|-----------|---------------------------|--------------------------|----------------|
| tesseract-wasm | ~2 sec | 2-5 sec | 8-20 sec | 2.1 MB |
| tesseract.js | ~3 sec | 3-8 sec | 10-30 sec | 15 MB |
| PaddleOCR.js (ONNX/WASM) | ~2 sec | 0.2-0.5 sec | 1-3 sec | 7-14 MB |
| PaddleOCR.js (ONNX/WebGPU) | ~2 sec | 0.1-0.3 sec | 0.5-2 sec | 7-14 MB |
| Transformers.js (Donut) | ~5 sec | 3-10 sec | N/A (too large) | 50-200 MB |

### Recommended Hybrid Architecture

```
                    +-------------------+
                    |   Client Browser  |
                    |                   |
     Upload PDF --> | 1. Render page    |
                    |    to canvas      |
                    |                   |
                    | 2. Quick OCR      |  <-- PaddleOCR.js (ONNX/WASM)
                    |    (text detect + |      ~200ms, det+rec only
                    |     recognize)    |      No layout analysis
                    |                   |
                    | 3. Instant text   |  <-- Show immediately
                    |    overlay        |      for basic editing
                    +--------+----------+
                             |
                             | async (if needed)
                             v
                    +-------------------+
                    |   Server (Docker) |
                    |                   |
                    | 4. Full OCR with  |  <-- PP-StructureV3
                    |    layout analysis|      Tables, formulas,
                    |    reading order  |      reading order,
                    |    table/formula  |      multi-column
                    |                   |
                    | 5. Return rich    |  --> Replace client-side
                    |    result         |      results with full
                    +-------------------+      structured output
```

**Split rationale**:
- **Client-side**: Text detection + recognition (fast, privacy-preserving). Good enough for basic text selection and copy/paste.
- **Server-side**: Layout analysis, reading order, table recognition, formula detection. These require larger models and are not time-critical (user sees text immediately from client-side pass).

**Recommended client-side library**: `paddleocr.js` (X3ZvaWQ) or `@gutenye/ocr-browser` - both use ONNX Runtime Web with quantized PaddleOCR models. Size is manageable (7-14 MB) and latency is competitive (200-500 ms).

### Implementation Plan

1. Prototype with `paddleocr.js` in a standalone HTML page; benchmark latency on test fixtures
2. Compare output against server-side PaddleOCR results (word count, text match, bbox IoU)
3. If acceptable, integrate as "instant preview" in FakeEdit v2 with async server refinement
4. Measure real-world latency distribution across Chrome/Firefox/Safari on desktop and mobile

---

## 4. Existing OCR Text Layer Quality

### Problem

Many scanned PDFs already contain an embedded OCR text layer (from the scanner software or tools like OCRmyPDF). Can we reuse these layers instead of running OCR again?

### How Common Are Embedded Text Layers?

Embedded OCR text layers are **very common** in enterprise and institutional settings:

- **Office multifunction scanners** (Canon ImageFORMULA, Ricoh, Xerox): Most modern models apply OCR during scanning by default, embedding a searchable text layer in the PDF.
- **Document scanning software** (Fujitsu ScanSnap Home, NAPS2, Adobe Acrobat): OCR is typically enabled as a post-scan step.
- **Document management systems**: Many enterprise DMS (SharePoint, M-Files) auto-OCR incoming scans.
- **OCRmyPDF**: Open-source tool specifically designed to add OCR text layers to scanned PDFs. Uses Tesseract under the hood.

**Estimate**: 60-80% of scanned PDFs in professional settings have some form of embedded text layer. However, quality varies enormously.

### Tools to Extract Embedded Text Layers

| Tool | Method | Preserves bbox? | Performance | Link |
|------|--------|-----------------|-------------|------|
| **PyMuPDF (fitz)** | `page.get_text("dict")` or `page.get_text("rawdict")` | Yes - returns spans with bbox `(x0, y0, x1, y1)` | Fastest (C library); handles complex PDFs well | [pymupdf.readthedocs.io](https://pymupdf.readthedocs.io/en/latest/) |
| **pdfplumber** | `page.extract_words()` | Yes - returns word dicts with bbox | Moderate; good table detection | [github.com/jsvine/pdfplumber](https://github.com/jsvine/pdfplumber) |
| **pdfminer.six** | `LAParams` + `LTTextBox` hierarchy | Yes - character-level bbox | Slow; detailed layout analysis | [github.com/pdfminer/pdfminer.six](https://github.com/pdfminer/pdfminer.six) |
| **PyMuPDF4LLM** | Hybrid: native text extraction + OCR fallback for image-only regions | Yes | Fast; applies OCR only where needed | [pymupdf.readthedocs.io/en/latest/pymupdf4llm/](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/) |

**Recommendation**: Use **PyMuPDF** (`page.get_text("rawdict")`) as the primary extraction tool. It is the fastest, preserves bounding boxes at the span level, and handles edge cases (rotated text, embedded fonts) reliably.

### Typical Accuracy of Embedded Layers

| Source | Typical accuracy (clean scan, 300 DPI, 12pt text) | Common problems |
|--------|---------------------------------------------------|----------------|
| **Scanner-embedded OCR** (Canon, Ricoh, Fujitsu) | 92-97% character accuracy | Misses headers/footers; poor on small text; no table structure; often uses older Tesseract (3.x) or proprietary engine |
| **OCRmyPDF** (Tesseract 5.x) | 95-99% character accuracy | Same as Tesseract limitations; no layout analysis |
| **Adobe Acrobat OCR** | 97-99% character accuracy | Best commercial quality; proprietary |
| **Low-quality scans** (< 200 DPI, skewed, noisy) | 70-85% character accuracy | All engines degrade significantly |

### Can the Editor Reuse Embedded Layers?

**Decision matrix**:

| Scenario | Reuse embedded layer? | Rationale |
|----------|----------------------|-----------|
| PDF has embedded text with high word count and plausible bbox coverage | **Yes, as initial layer** | Extract with PyMuPDF; display immediately; optionally validate with quick OCR |
| PDF is image-only (no text layer) | **No** | Must run full OCR |
| PDF has embedded text but bbox positions are clearly wrong (e.g., all at origin) | **No** | Some scanners embed searchable text without positional accuracy |
| PDF has mixed pages (some with text, some image-only) | **Hybrid** | Use PyMuPDF4LLM approach: extract where available, OCR the rest |

**Quality check heuristic** (to decide whether to trust an embedded layer):

```python
import fitz  # PyMuPDF

def assess_embedded_text_quality(page: fitz.Page) -> dict:
    """Assess whether an embedded text layer is usable."""
    text_dict = page.get_text("rawdict")
    blocks = text_dict.get("blocks", [])

    text_blocks = [b for b in blocks if b["type"] == 0]  # type 0 = text
    if not text_blocks:
        return {"usable": False, "reason": "no_text_blocks"}

    total_chars = sum(
        len(span["text"])
        for block in text_blocks
        for line in block["lines"]
        for span in line["spans"]
    )

    # Check bbox sanity: are bboxes spread across the page?
    bboxes = [span["bbox"] for block in text_blocks
              for line in block["lines"] for span in line["spans"]]
    x_spread = max(b[2] for b in bboxes) - min(b[0] for b in bboxes)
    y_spread = max(b[3] for b in bboxes) - min(b[1] for b in bboxes)

    page_w, page_h = page.rect.width, page.rect.height
    coverage_x = x_spread / page_w if page_w > 0 else 0
    coverage_y = y_spread / page_h if page_h > 0 else 0

    return {
        "usable": total_chars > 50 and coverage_x > 0.3 and coverage_y > 0.3,
        "char_count": total_chars,
        "coverage_x": round(coverage_x, 2),
        "coverage_y": round(coverage_y, 2),
    }
```

### Implementation Plan

1. Add a PyMuPDF-based text layer extractor to the gateway service
2. Before dispatching to an OCR engine, check if the PDF has a usable embedded layer
3. If usable, return the embedded layer result immediately (with `provider: "embedded"`)
4. Benchmark embedded layer accuracy against fresh OCR on the same documents
5. Let the editor decide: use embedded layer for instant display, optionally refine with server OCR

---

## 5. Ground-Truth Accuracy Datasets

### Problem

Without ground-truth text transcriptions, we cannot measure actual OCR accuracy (CER, WER). Our current benchmark uses cross-engine similarity as a proxy, which cannot detect errors all engines share.

### Public OCR Accuracy Datasets

#### Document OCR Datasets (most relevant to our use case)

| Dataset | Size | Document types | Annotation level | Relevance to us | Link |
|---------|------|---------------|-----------------|-----------------|------|
| **FUNSD** | 199 scanned forms | Forms with handwritten + printed text | Word-level bbox + text + entity labels | **High** - forms are a key FakeEdit use case | [guillaumejaume/FUNSD](https://github.com/guillaumejaume/FUNSD) |
| **SROIE** | 1,000 receipts | Scanned receipts | Word-level bbox + text | **Medium** - tests small text, varied fonts | [ICDAR 2019 SROIE](https://rrc.cvc.uab.es/?ch=13) |
| **CORD** | 11,000 receipts | Receipts | Word-level bbox + text + parsing labels | **Medium** - receipt-specific but good for bbox testing | [clovaai/cord](https://github.com/clovaai/cord) |
| **OmniDocBench** | 981 pages | Academic, financial, legal, patents, textbooks, newspapers | Block-level text + reading order + tables + formulas | **Very high** - diverse document types, matches our use case | [opendatalab/OmniDocBench](https://github.com/opendatalab/OmniDocBench) |
| **HierText** | 11,639 images | Natural scene + document | Character, word, line, paragraph hierarchy | **Medium** - very dense annotations (103 words/image avg) | [google-research-datasets/hiertext](https://github.com/google-research-datasets/hiertext) |
| **DocILE** | 6,680 pages | Business documents | Word-level bbox (using docTR OCR) + key-value labels | **High** - business documents match editing use case | [rossumai/docile](https://github.com/rossumai/docile) |

#### ICDAR Competitions (Which Are Relevant?)

| Competition | Year | Relevance | Why |
|------------|------|-----------|-----|
| **ICDAR 2019 SROIE** | 2019 | Medium | Receipt OCR; text localization + recognition tasks |
| **ICDAR 2023 HierText Challenge** | 2023 | High | Hierarchical text detection; unified OCR + layout evaluation |
| **ICDAR 2023 Robust Layout Segmentation** | 2023 | High | Layout segmentation in corporate documents |
| **ICDAR 2024 Multi-Font Group Recognition** | 2024 | Low | Specialized (early modern prints) |
| **ICDAR 2025 Indic Handwriting** | 2025 | Low | Indic scripts; not our primary use case |

#### Comprehensive Benchmarks

| Benchmark | Focus | Relevant metrics | Link |
|-----------|-------|-----------------|------|
| **OmniDocBench** (CVPR 2025) | End-to-end document parsing | Text NED, table TEDS, formula CDM, reading order NED | [github.com/opendatalab/OmniDocBench](https://github.com/opendatalab/OmniDocBench) |
| **CC-OCR** (ICCV 2025) | Comprehensive OCR (31 scenarios, 23 tasks) | Task-specific accuracy metrics | [CC-OCR paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Yang_CC-OCR_A_Comprehensive_and_Challenging_OCR_Benchmark_for_Evaluating_Large_ICCV_2025_paper.pdf) |
| **OCRBench v2** | Multimodal model OCR evaluation | Multiple OCR sub-task scores | [OCRBench](https://github.com/Yuliang-Liu/MultimodalOCR) |

### Building a Ground-Truth Dataset from Our Fixtures

Our test fixtures directory contains 10 synthetic fixtures + 1 multipage PDF, generated with known text. We can build ground truth in two ways:

#### Option A: Record Ground Truth During Generation (Recommended)

Since the fixture generator renders all text programmatically, we can record every word + position at generation time:

```python
# Output format: <fixture-name>.gt.json
{
    "fixture": "simple-typed.png",
    "width": 2480,
    "height": 3508,
    "words": [
        {
            "text": "quarterly",
            "bbox": {"x1": 150, "y1": 200, "x2": 380, "y2": 235},
            "line_index": 0,
            "block_index": 0
        },
        ...
    ],
    "lines": [...],
    "blocks": [...],
    "reading_order": [0, 1, 2, ...]
}
```

This gives us pixel-perfect ground truth for all synthetic fixtures. The existing `generate.py` already uses Pillow `ImageDraw` and ReportLab; both provide text measurement APIs (`draw.textbbox()`, `stringWidth()`).

#### Option B: Manual Transcription of Real Scanned Documents

For a small set of real-world documents (not synthetic), manual transcription provides the most reliable ground truth:

1. Select 10-20 representative real scanned documents (invoices, contracts, academic papers, government forms)
2. Use a transcription tool (Label Studio with OCR pre-annotation, or Prodigy)
3. Have 2 annotators transcribe independently; resolve disagreements
4. Expected effort: ~30 minutes per page for word-level annotation

**Tooling for annotation**:

| Tool | Type | OCR pre-annotation? | Export format | Link |
|------|------|---------------------|---------------|------|
| **Label Studio** | Open source, self-hosted | Yes (Tesseract integration built-in) | JSON, CSV, COCO | [labelstud.io](https://labelstud.io) |
| **CVAT** | Open source, self-hosted | Yes (custom model integration) | COCO, PASCAL VOC | [github.com/opencv/cvat](https://github.com/opencv/cvat) |
| **Prodigy** | Commercial, local | Yes (custom model) | JSONL | [prodi.gy](https://prodi.gy) |

#### Option C: Commercial OCR as Reference

Use a high-accuracy commercial OCR service as a pseudo-ground-truth reference:

1. Run Google Cloud Vision or AWS Textract on all fixtures
2. Manually review and correct any errors (faster than full transcription)
3. Use corrected output as reference for benchmarking our 4 open-source engines

**Estimated cost**: ~$0.01-0.03 per page (Google Vision); ~10 pages = $0.30 total.

### Accuracy Metrics to Implement

| Metric | Formula | What it captures |
|--------|---------|-----------------|
| **CER** (Character Error Rate) | `levenshtein(pred, gt) / len(gt)` | Character-level accuracy including substitutions, insertions, deletions |
| **WER** (Word Error Rate) | `levenshtein(pred_words, gt_words) / len(gt_words)` | Word-level accuracy |
| **BoW F1** (Bag of Words) | F1 on word multiset overlap | Order-independent word accuracy |
| **1 - NED** (Normalized Edit Distance) | `1 - levenshtein(pred, gt) / max(len(pred), len(gt))` | Normalized accuracy (0-1 scale) |

```python
# Add to the analysis pipeline
import Levenshtein

def cer(predicted: str, ground_truth: str) -> float:
    """Character Error Rate."""
    if not ground_truth:
        return 0.0 if not predicted else 1.0
    return Levenshtein.distance(predicted, ground_truth) / len(ground_truth)

def wer(predicted: str, ground_truth: str) -> float:
    """Word Error Rate."""
    pred_words = predicted.split()
    gt_words = ground_truth.split()
    if not gt_words:
        return 0.0 if not pred_words else 1.0
    return Levenshtein.distance(
        " ".join(pred_words), " ".join(gt_words)
    ) / len(gt_words)
```

### Implementation Plan

1. **Phase 1**: Extend the fixture generator to emit `.gt.json` files for all synthetic fixtures
2. **Phase 2**: Add an accuracy evaluation script with CER, WER, BoW F1 metrics
3. **Phase 3**: Run all 4 engines against fixtures with ground truth; add accuracy columns to the benchmark report
4. **Phase 4**: Download FUNSD + OmniDocBench subsets; integrate into benchmark pipeline
5. **Phase 5** (optional): Run Google Cloud Vision on fixtures for reference comparison

---

## Priority Matrix

| Gap | Impact on FakeEdit v2 | Implementation effort | Recommended priority |
|-----|----------------------|----------------------|---------------------|
| Bounding box accuracy (Section 2) | **Critical** - directly affects text overlay quality | Low (extend generate.py + add IoU metric) | **P0 - Do first** |
| Ground-truth datasets (Section 5) | **High** - enables all other accuracy measurements | Low-Medium (synthetic GT is easy; real-world GT takes time) | **P0 - Do first** (prerequisite for others) |
| Reading order (Section 1) | **High** - affects document editing flow | Medium (OmniDocBench integration + metric implementation) | **P1** |
| Existing text layers (Section 4) | **Medium** - optimization; reduces server load | Low (PyMuPDF extraction is straightforward) | **P2** |
| Browser-side OCR (Section 3) | **Medium** - UX improvement; not blocking | Medium-High (prototype + cross-browser testing) | **P2** |

---

## References

### Layout and Reading Order
- [OmniDocBench (CVPR 2025)](https://github.com/opendatalab/OmniDocBench) - comprehensive document parsing benchmark with reading order evaluation
- [DocLayNet](https://github.com/DS4SD/DocLayNet) - 80K+ pages, 11 layout classes, 6 document categories
- [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet) - 1M+ scientific article layout annotations
- [PP-StructureV3 Documentation](http://www.paddleocr.ai/main/en/version3.x/algorithm/PP-StructureV3/PP-StructureV3.html)
- [Surya OCR](https://github.com/datalab-to/surya) - OCR with layout detection, limited reading order
- [Lapata (2006), Automatic Evaluation of Information Ordering: Kendall's Tau](https://aclanthology.org/J06-4002.pdf)

### Bounding Box and Synthetic Data
- [SynthDOG (within Donut)](https://github.com/clovaai/donut) - synthetic document generator with bbox annotations
- [TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator) - synthetic text line images
- [Genalog (Microsoft)](https://github.com/microsoft/genalog) - degraded document synthesis
- [LightOnOCR-bbox-bench](https://www.emergentmind.com/topics/lightonocr-bbox-bench) - document image localization benchmark

### Browser OCR
- [tesseract.js](https://github.com/naptha/tesseract.js) - WASM Tesseract for browser
- [tesseract-wasm](https://github.com/robertknight/tesseract-wasm) - lighter alternative with SIMD
- [paddleocr.js](https://github.com/X3ZvaWQ/paddleocr.js/) - PaddleOCR v5 with ONNX Runtime
- [@gutenye/ocr-browser](https://www.npmjs.com/package/@gutenye/ocr-browser) - PP-OCRv4 in browser
- [Transformers.js v4](https://github.com/huggingface/transformers.js/) - WebGPU-accelerated ML inference
- [client-side-ocr](https://libraries.io/npm/client-side-ocr) - ONNX Runtime + PaddleOCR client-side

### PDF Text Layers
- [PyMuPDF documentation](https://pymupdf.readthedocs.io/en/latest/)
- [PyMuPDF4LLM (hybrid OCR)](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/)
- [pdfplumber](https://github.com/jsvine/pdfplumber)
- [OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF)

### Ground Truth and Accuracy
- [FUNSD](https://guillaumejaume.github.io/FUNSD/) - form understanding benchmark
- [SROIE (ICDAR 2019)](https://rrc.cvc.uab.es/?ch=13) - receipt OCR competition
- [CORD](https://github.com/clovaai/cord) - receipt parsing dataset
- [DocILE](https://github.com/rossumai/docile) - document information extraction benchmark
- [HierText](https://github.com/google-research-datasets/hiertext) - hierarchical text detection
- [CC-OCR (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Yang_CC-OCR_A_Comprehensive_and_Challenging_OCR_Benchmark_for_Evaluating_Large_ICCV_2025_paper.pdf)
- [Label Studio](https://labelstud.io/) - open-source annotation tool with OCR integration
