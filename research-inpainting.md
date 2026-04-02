# Text Removal and Image Inpainting for Scanned Document Editing

> Research document -- April 2026
> Context: FakeEdit v2 needs to cleanly erase original text from scanned document images before re-rendering edited text. This is Step 4 of the editing pipeline: OCR detect -> user edits -> **erase original text** -> render new text.
>
> **Pipeline step**: Problem 4 in the [editing pipeline](./README.md). Consumes text masks derived from OCR bounding boxes (Problem 1, [DR-001](./DR-001-ocr-adapter.md)). Runs before text re-rendering (Problem 5).
>
> **Related**: [research-font-estimation.md](./research-font-estimation.md) (font metadata for re-rendering after inpainting) · [research-benchmark-gaps.md](./research-benchmark-gaps.md) (bbox accuracy affects mask quality) · [cost-performance.md](./cost-performance.md) (inpainting cost: $3-5/month at 500K pages)

---

## Table of Contents

1. [Dedicated Text Erasure Models](#1-dedicated-text-erasure-models)
2. [General Image Inpainting Applied to Text Removal](#2-general-image-inpainting-applied-to-text-removal)
3. [Document-Specific Inpainting Considerations](#3-document-specific-inpainting-considerations)
4. [Mask Generation from OCR Output](#4-mask-generation-from-ocr-output)
5. [PDF-Native Text Removal](#5-pdf-native-text-removal)
6. [Browser-Side / Client-Side Inpainting](#6-browser-side--client-side-inpainting)
7. [Practical Pipeline Architecture](#7-practical-pipeline-architecture)
8. [Commercial / SaaS Alternatives](#8-commercial--saas-alternatives)
9. [Consolidated Benchmark Comparison](#9-consolidated-benchmark-comparison)
10. [Recommended Approach for FakeEdit v2](#10-recommended-approach-for-fakeedit-v2)

---

## 1. Dedicated Text Erasure Models

Scene text removal (STR) is a mature sub-field with dedicated models trained specifically to erase text while reconstructing the background. The primary benchmarks are **SCUT-EnsText** (3,562 real-world images) and **SCUT-Syn** (synthetic dataset). These models typically take an image + text mask as input and output the image with text regions replaced by plausible background.

### 1.1 Current SOTA: DiffSTR (Oct 2024)

**Paper:** "DiffSTR: Controlled Diffusion Models for Scene Text Removal" (arXiv:2410.21721)

DiffSTR is the first work applying conditional diffusion models to STR. It uses a ControlNet-conditioned latent diffusion model built on Paint-by-Example (PBE), with a novel TR-MAE pretraining pipeline and segmentation-based mask refinement.

| Dataset | PSNR | MSSIM | MSE | FPS |
|---------|------|-------|-----|-----|
| SCUT-EnsText | **37.25** | **0.9798** | **0.04** | 2 |
| SCUT-Syn | **43.75** | **0.9883** | **0.01** | 2 |

- **Hardware:** Single A100 GPU (80GB), 512x512 input
- **Inference speed:** 2 FPS -- significantly slower than non-diffusion methods
- **GitHub:** Not yet publicly released at time of writing
- **Verdict:** Best quality, but too slow for interactive editing. Potential for batch/offline processing.

### 1.2 TMIM + Uformer-B (Sep 2024)

**Paper:** "Leveraging Text Localization for Scene Text Removal via Text-aware Masked Image Modeling" (ECCV 2024)

TMIM is a pretraining framework (not a model itself) that enhances existing STR models using text-aware masked image modeling with only bounding-box-level annotations.

| Method | PSNR | MSSIM | MSE | Params |
|--------|------|-------|-----|--------|
| Uformer-B + TMIM | **37.42** | **97.70** | **0.0459** | 50.88M |
| PERT + TMIM | 34.75 | 97.18 | 0.0968 | -- |
| EraseNet + TMIM | 34.25 | 96.63 | 0.1231 | -- |

- **GitHub:** https://github.com/wzx99/TMIM (12 stars, active Jul 2024)
- **Weights:** Available via Google Drive for Uformer-B, PERT, and EraseNet variants
- **Key advantage:** Lighter than ViTEraser (50.88M vs 191.9M params) with competitive or better PSNR
- **Verdict:** Strong candidate -- good quality with reasonable model size. TMIM pretraining can enhance models we already have.

### 1.3 ViTEraser (AAAI 2024)

**Paper:** "ViTEraser: Harnessing the Power of Vision Transformers for Scene Text Removal with SegMIM Pretraining" (arXiv:2306.12106)

ViTEraser uses a Swin Transformer V2-based encoder-decoder with a novel SegMIM pretraining method. It was the previous SOTA before DiffSTR/TMIM.

| Variant | PSNR | MSSIM | MSE | FID | FPS |
|---------|------|-------|-----|-----|-----|
| ViTEraser-Tiny + SegMIM | 36.80 | 97.55 | 0.0491 | 10.79 | ~15 |
| ViTEraser-Small + SegMIM | 37.08 | 97.62 | 0.0447 | 10.16 | ~15 |
| ViTEraser-Base + SegMIM | 37.11 | 97.61 | 0.0474 | 10.15 | ~15 |

- **GitHub:** https://github.com/shannanyinxiang/ViTEraser (63 stars, last commit Jun 2023)
- **Weights:** Available (Tiny/Small/Base) via BaiduNetDisk and Google Drive
- **License:** MIT (non-commercial research only; commercial use requires contacting Prof. Lianwen Jin)
- **Input:** 512x512, requires text masks
- **Verdict:** Strong quality, reasonable speed (15 FPS). Non-commercial license is a concern.

### 1.4 DeepEraser (TMM 2024)

**Paper:** "DeepEraser: Deep Iterative Context Mining for Generic Text Eraser"

- **GitHub:** https://github.com/fh2019ustc/DeepEraser (51 stars, last commit Feb 2024)
- **Weights:** Included in repo (`deeperaser.pth`) + Google Drive
- **License:** MIT
- **Key feature:** Iterative context mining; has an online demo for interactive testing
- **Verdict:** Permissive license, decent quality, good for evaluation.

### 1.5 CTRNet (ECCV 2022)

**Paper:** "Don't Forget Me: Accurate Background Recovery for Text Removal via Modeling Local-Global Context"

Decouples text removal into four heads: Text Perception, Low-level/High-level Contextual Guidance (LCG/HCG), and Local-global Content Modeling (LGCM).

| Dataset | PSNR | MSSIM | FID |
|---------|------|-------|-----|
| SCUT-EnsText | 35.20 | 97.36 | 13.99 |

- **GitHub:** https://github.com/lcy0604/CTRNet (96 stars)
- **Weights:** Available (retrained Jul release)
- **License:** MIT
- **Input:** 512x512, requires RTV structure images
- **Also:** CTRNet++ available at https://github.com/lcy0604/CTRNet-plus
- **Verdict:** Solid quality. Extra RTV preprocessing adds complexity.

### 1.6 GaRNet (ECCV 2022)

**Paper:** "The Surprisingly Straightforward Scene Text Removal Method for Any Data"

A unified approach with gated attention and ROI generation. Repository also provides pre-trained weights for EnsNet, MTRNet, MTRNet++, and EraseNet.

| Dataset | PSNR | MSSIM | FID |
|---------|------|-------|-----|
| SCUT-EnsText | 35.45 | 97.14 | 15.50 |

- **GitHub:** https://github.com/naver/garnet (69 stars, last commit Jul 2022)
- **Weights:** Available on Hugging Face + Google Drive (includes 5 model variants)
- **License:** Apache-2.0
- **Input:** 512px, requires bounding box annotations
- **Verdict:** Apache license is attractive. Multiple model weights in one repo. Good for benchmarking.

### 1.7 PERT (TIP 2023)

**Paper:** "PERT: A Progressively Region-based Network for Scene Text Removal"

Progressive multi-stage erasure with a Region-based Modification Strategy (RegionMS). Claims SOTA speed.

| Dataset | PSNR | MSSIM | FPS |
|---------|------|-------|-----|
| SCUT-EnsText | 33.62 | 97.00 | **71** |

- **GitHub:** https://github.com/wangyuxin87/PERT (37 stars, last commit Jun 2021)
- **Weights:** Available via Google Drive
- **Key advantage:** 71 FPS -- fastest among dedicated STR models, 25%+ fewer parameters
- **Verdict:** Best speed-to-quality ratio. Good candidate for real-time/interactive use.

### 1.8 FETNet (Pattern Recognition 2023)

**Paper:** "FETNet: Feature Erasing and Transferring Network for Scene Text Removal"

Uses a Feature Erasing Module (FEM) + Feature Transferring Module (FTM) with attention guidance.

| Dataset | PSNR | MSSIM |
|---------|------|-------|
| SCUT-EnsText | 34.53 | 97.01 |

- **GitHub:** https://github.com/GuangtaoLyu/FETNet (35 stars, last commit Apr 2022)
- **Weights:** Available via Baidu/Google Drive
- **License:** Not explicitly stated (non-commercial dataset restriction)
- **Verdict:** Middle-of-pack quality. No clear advantage over alternatives.

### 1.9 Older Models (EnsNet, EraseNet, MTRNet++)

These are foundational works but significantly outperformed by newer methods:

| Method | PSNR | MSSIM | FID | GitHub |
|--------|------|-------|-----|--------|
| EnsNet (2019) | 29.54 | 92.74 | 32.71 | https://github.com/HCIILAB/Scene-Text-Removal (131 stars) |
| EraseNet (2020) | 32.30 | 95.42 | 19.27 | (within GaRNet repo) |
| MTRNet++ | 29.63 | 93.71 | 35.68 | (within GaRNet repo) |

**Verdict:** Useful as baselines. EnsNet/MTRNet++ weights available in the GaRNet repo.

---

## 2. General Image Inpainting Applied to Text Removal

These models are not text-specific but can be applied to text removal by providing a text mask. They tend to be more versatile but may produce artifacts specific to text (ghost outlines, inconsistent stroke reconstruction).

### 2.1 LaMa (WACV 2022) -- Recommended General Inpainter

**Paper:** "Resolution-robust Large Mask Inpainting with Fourier Convolutions" (arXiv:2109.07161)

LaMa uses Fast Fourier Convolutions (FFCs) for image-wide receptive fields. It is the most widely adopted open-source inpainting model.

| Metric | Value |
|--------|-------|
| Parameters | ~45.6M |
| Model size | ~174 MB |
| Training resolution | 256x256 |
| Inference resolution | Generalizes to ~2k |
| GPU inference | ~48ms per image |
| CPU inference | ~25s (36 cores) |
| ONNX available | Yes (see Section 6) |

- **GitHub:** https://github.com/advimman/lama (9.8k stars, last active Jan 2023)
- **License:** Apache-2.0
- **Pre-trained:** Places2, Places Challenge, CelebA-HQ
- **Text removal quality:** Good on uniform backgrounds; struggles with high-contrast text on complex backgrounds
- **Key strength:** Resolution robustness -- trained at 256x256 but works at 2k+
- **Key weakness:** Not trained for text specifically; may leave subtle ghost artifacts on textured backgrounds

**For text removal specifically:** LaMa is best suited for:
- White/off-white document backgrounds (excellent results)
- Uniform colored backgrounds (good results)
- Textured backgrounds (acceptable but may need post-processing)

### 2.2 IOPaint (formerly Lama-Cleaner)

IOPaint wraps LaMa and other models into a production-ready tool with a web UI, CLI, and Docker support.

- **GitHub:** https://github.com/Sanster/IOPaint (22.9k stars, last release Nov 2024, archived Aug 2025)
- **License:** Apache-2.0
- **Supported models:** LaMa, Stable Diffusion variants, PowerPaint, BrushNet, AnyText
- **Features:** Web UI, CLI batch processing, Docker, CPU/GPU/Apple Silicon
- **Install:** `pip install iopaint && iopaint start --model=lama --device=cpu --port=8080`
- **Verdict:** Best turnkey solution for evaluation and rapid prototyping. Archived but fully functional.

### 2.3 MAT (CVPR 2022 Best Paper Finalist)

**Paper:** "MAT: Mask-Aware Transformer for Large Hole Image Inpainting"

Unifies transformers and convolutions with a dynamic mask-aware attention mechanism.

| Metric | Places-512 Small Mask | Places-512 Large Mask |
|--------|-----------------------|-----------------------|
| FID | 0.78 | 1.96 |
| P-IDS | 31.72 | 23.42 |

- **GitHub:** https://github.com/fenglinglwb/MAT (970 stars, last commit Mar 2022)
- **License:** Research only
- **Resolution:** 512x512 (multiples of 512 only)
- **Verdict:** Higher quality than LaMa on large masks but slower, research-only license, and 512x512 constraint is limiting for documents.

### 2.4 Stable Diffusion Inpainting

Latent diffusion model fine-tuned for inpainting at 512x512.

- **Model:** `stable-diffusion-v1-5/stable-diffusion-inpainting` on Hugging Face
- **Training:** 595k regular steps + 440k inpainting steps on LAION Aesthetics
- **Architecture:** UNet with 5 extra input channels (4 encoded mask-image + 1 mask)

| Consideration | Assessment |
|---------------|------------|
| Quality | Excellent for natural images; overkill for document backgrounds |
| Latency | ~2-5s per image on GPU (50 diffusion steps) |
| Model size | ~4 GB (FP16) |
| Hallucination risk | May generate plausible but incorrect content (problematic for documents) |

- **Verdict:** Overkill and risky for document editing. The model may hallucinate patterns or textures that do not match the original document. Latency is 50-100x worse than LaMa. Use only as a fallback for extremely complex backgrounds.

### 2.5 OpenCV Classical Approaches (Telea / Navier-Stokes)

Two classical algorithms available in OpenCV:

**Telea (Fast Marching Method):**
- Based on FMM by Alexandru Telea (2004)
- Fills from boundary inward using normalized weighted sums of known pixels
- `cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)`

**Navier-Stokes:**
- Based on fluid dynamics PDEs
- `cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)`

| Consideration | Assessment |
|---------------|------------|
| Quality | Good for small masks on uniform backgrounds; artifacts on textured areas |
| Latency | <10ms (pure CPU, no GPU needed) |
| Model size | 0 (built into OpenCV) |
| Dependencies | `opencv-python` only |
| Browser support | Available in OpenCV.js (WASM, ~5.3 MB) |

- **Best for:** White-background documents with clean printed text. Descenders and serifs may leave faint residue if mask is too tight.
- **Worst for:** Textured backgrounds, large text blocks, handwritten text.
- **Verdict:** The simplest and fastest option. Should be the default for white/uniform-background documents where quality requirements are modest.

### 2.6 PaddlePaddle Ecosystem

PaddleOCR (which we already use) does **not** include a dedicated text removal or inpainting module. PP-Structure focuses on document layout analysis, table extraction, and key information extraction -- not text erasure.

The common PaddlePaddle-based pipeline for text removal is:
1. Use PaddleOCR for text detection (which we already do)
2. Use the detection masks as input to a separate inpainting model (typically LaMa)

There is no first-party PaddlePaddle inpainting model that competes with the options above.

---

## 3. Document-Specific Inpainting Considerations

### 3.1 Is Document Text Removal Easier Than Scene Text Removal?

**Yes, significantly.** Document images have several properties that simplify text removal:

| Property | Scene Text | Document Text |
|----------|-----------|---------------|
| Background complexity | Highly varied (buildings, nature, products) | Usually uniform (white, off-white, light colors) |
| Text orientation | Arbitrary angles, perspective distortion | Horizontal (or rotated 90/180/270) |
| Font variety | Unlimited (signs, graffiti, handwritten) | Limited set (standard print fonts) |
| Text-background contrast | Variable, sometimes low | Usually high (black on white) |
| Mask precision needed | High (irregular boundaries) | Moderate (rectangular bounding boxes often suffice) |
| Overlapping elements | Common (text over complex objects) | Rare in clean documents; possible with stamps/annotations |

**Implication:** A general inpainter like LaMa, which may struggle on complex scene text, performs well on typical document backgrounds. OpenCV classical methods (Telea/NS) may even suffice for the majority of cases.

### 3.2 Background Reconstruction Challenges

**Lined paper / Graph paper:**
- Regular periodic patterns are a strength of LaMa (FFCs handle periodic structures well)
- Classical methods will break the line continuity -- avoid for lined/graph paper
- STR models trained on scene text may struggle because they never see lined-paper backgrounds

**Watermarks:**
- Semi-transparent watermarks overlapping text are difficult
- Inpainting will remove both the text and the watermark fragment
- Possible mitigation: detect watermark pattern globally, reconstruct it in the inpainted region
- In practice, for FakeEdit this may be acceptable if the user is replacing text at the same location

**Stamps / Seals overlapping text:**
- The most challenging case: stamp ink and text ink occupy the same pixels
- No inpainting model reliably separates the two
- Best approach: accept partial stamp removal and document this as a known limitation
- Alternative: use color-channel separation (stamps are often red/blue, text is black)

**Borders and table lines:**
- Similar to lined paper -- periodic structures
- LaMa handles these well; classical methods may introduce discontinuities

### 3.3 Multi-Pass Approach

For highest quality, a multi-pass pipeline is recommended:

1. **Detect text** -- OCR bounding boxes (already have this)
2. **Generate mask** -- dilated/refined mask from bboxes (see Section 4)
3. **First pass inpaint** -- LaMa or STR model
4. **Verify** -- check for ghost artifacts, color discontinuities
5. **Optional second pass** -- micro-cleanup on problem areas with tighter masks
6. **Cache** -- store the clean background for subsequent edits on the same page

### 3.4 Ghost Artifact Detection

Ghost artifacts (faint remnants of erased text) are the most common inpainting failure mode for text removal. Detection approaches:

- **Pixel intensity analysis:** Compare the inpainted region's mean/variance with surrounding background. Ghost text creates subtle intensity shifts.
- **OCR re-run:** Run OCR on the inpainted region. If any text is detected, the erasure was incomplete.
- **Edge detection:** Apply Canny/Sobel to the inpainted region. Text ghosts produce weak but detectable edges.
- **SSIM comparison:** Compare the inpainted region with a synthetic "ideal" background (e.g., filled with median background color). Low SSIM suggests artifacts.

Recent research (COCO-Inpaint benchmark, 2025) provides standardized evaluation using mIoU and F1 scores for inpainting artifact localization.

---

## 4. Mask Generation from OCR Output

The quality of text removal depends critically on the mask. OCR bounding boxes are typically word-level or line-level rectangles, but the actual text occupies an irregular subset of pixels within those rectangles.

### 4.1 Simple Dilation Strategy (Recommended Starting Point)

```python
import cv2
import numpy as np

def bbox_to_mask(image_shape, bboxes, dilation_px=5):
    """Generate binary mask from OCR bounding boxes with dilation."""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        mask[y1:y2, x1:x2] = 255
    # Dilate to cover descenders, serifs, ink spread
    kernel = np.ones((dilation_px * 2 + 1, dilation_px * 2 + 1), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask
```

**Dilation guidance:**
- **3-5 px** for clean printed text at 300 DPI
- **7-10 px** for handwritten text or low-quality scans
- **2-3 px** for high-resolution (600 DPI) scans

The dilation must be large enough to cover:
- Character descenders (g, j, p, q, y) that may extend below the baseline bbox
- Serifs and stroke terminals
- Ink spread / anti-aliasing at character edges

### 4.2 Polygon Masks from Text Detection Models

For more precise masks, use a text detection model that outputs polygons rather than axis-aligned rectangles:

**DBNet (Differentiable Binarization):**
- Used internally by PaddleOCR for detection
- Outputs per-pixel probability maps, post-processed into polygons via adaptive thresholding
- Can produce tight-fitting polygonal masks around curved or rotated text
- Our PaddleOCR service already runs DBNet -- we can extract the raw segmentation maps

**CRAFT (Character Region Awareness for Text Detection):**
- GitHub: https://github.com/clovaai/CRAFT-pytorch
- Detects character-level regions and affinity between characters
- Outputs heatmaps that can be thresholded into pixel-precise masks
- Better for irregular (curved, artistic) text than DBNet

**Approach for FakeEdit:** Since PaddleOCR already runs DBNet internally, we should extract the raw segmentation probability map from the detection stage and use it as a pixel-level mask, rather than degrading to rectangular bounding boxes. This gives near-perfect text masks with zero additional inference cost.

### 4.3 Segmentation-Based Refinement

For maximum precision, crop each text region, binarize it (Otsu or adaptive thresholding), and use the binarized text as the mask:

```python
def refine_mask_from_crop(image, bbox, threshold_method='otsu'):
    """Refine mask using text binarization within a bbox crop."""
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    if threshold_method == 'otsu':
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
    
    # Dilate slightly to ensure full coverage
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    
    # Place back into full-size mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = binary
    return mask
```

**Advantage:** Pixel-perfect masks that follow exact character outlines.
**Disadvantage:** Fails when text and background have similar intensity (low contrast), or when background has strong texture.

---

## 5. PDF-Native Text Removal

For born-digital PDFs (not scanned images), text can be removed by editing the PDF content stream directly, avoiding image inpainting entirely.

### 5.1 PyMuPDF (fitz) Redaction API

PyMuPDF provides the most capable redaction workflow:

```python
import fitz  # PyMuPDF

doc = fitz.open("document.pdf")
page = doc[0]

# Search for text
text_instances = page.search_for("text to remove")

for inst in text_instances:
    # Add redaction annotation with white fill
    page.add_redact_annot(
        inst,
        text="replacement text",     # optional replacement
        fill=(1, 1, 1),              # white background fill
        text_color=(0, 0, 0),        # black text
        fontname="helv",             # Helvetica
        fontsize=11
    )

# Apply all redactions (permanently removes content)
page.apply_redactions(
    images=fitz.PDF_REDACT_IMAGE_NONE,  # don't touch images
    graphics=0                            # ignore vector graphics
)

doc.save("output.pdf")
```

**Capabilities:**
- Searches text in the PDF content stream (not OCR-based)
- Replaces text with specified fill color and optional new text
- `apply_redactions()` permanently removes the original content
- Control over whether images and graphics are affected

**Limitations:**
- Only works on born-digital PDFs with actual text objects
- Scanned PDFs contain only images -- redaction has no text to find
- `apply_redactions()` may remove more text than expected (reported issue #3433)
- Font matching is limited to built-in PDF fonts

### 5.2 pikepdf

pikepdf provides lower-level PDF content stream manipulation via its qpdf-based parser:

- **GitHub:** https://github.com/pikepdf/pikepdf
- **Approach:** Parse content stream operators, identify text-drawing operators (Tj, TJ, Tf), remove or modify them
- **Advantage:** Full control over the PDF content stream
- **Disadvantage:** Requires deep knowledge of PDF specification; no high-level text-removal API
- **Use case:** When PyMuPDF's redaction is too coarse and you need precise operator-level control

### 5.3 Applicability to FakeEdit v2

For our use case:
- **Scanned PDFs** (majority of documents): These are just images wrapped in PDF. We must use image-based inpainting (Sections 1-2). PyMuPDF can extract the page image, we inpaint it, and re-insert.
- **Born-digital PDFs** (minority): We can use PyMuPDF redaction for clean text removal without any image processing. This is faster and produces pixel-perfect results.
- **Detection:** Use PyMuPDF's `page.get_text("dict")` -- if it returns text blocks, the PDF is born-digital. If it returns nothing but the page has visible content, it is a scanned image.

---

## 6. Browser-Side / Client-Side Inpainting

Running inpainting client-side avoids server round-trips and reduces infrastructure cost. Feasibility depends on model size and browser runtime performance.

### 6.1 LaMa via ONNX Runtime Web

A community-maintained ONNX export of LaMa exists:

- **Model:** https://huggingface.co/Carve/LaMa-ONNX
- **Files:** `lama_fp32.onnx` (ONNX opset 17, recommended) and `lama.onnx` (opset 18)
- **Input shape:** Fixed 512x512
- **Model download size:** ~208 MB
- **Demo:** https://huggingface.co/spaces/Carve/LaMa-Demo-ONNX
- **Runtime options:** WebGL, WebGPU, WebNN (GPU) or WASM (CPU)
- **Export notebook:** https://colab.research.google.com/github/Carve-Photos/lama/blob/main/export_LaMa_to_onnx.ipynb

**Practical considerations:**
- 208 MB download is large but cacheable (browser IndexedDB / Cache API)
- First inference is slow (~5-10s) due to model compilation; subsequent runs faster
- 512x512 fixed input requires tiling for full-page documents
- Per-block cropping (crop text region, resize to 512x512, inpaint, resize back) is more practical
- WebGPU runtime (where available) provides near-native GPU performance

### 6.2 OpenCV.js Classical Inpainting

OpenCV.js provides `cv.inpaint()` with both Telea and Navier-Stokes algorithms:

- **Library size:** ~5.3 MB (WASM build)
- **Latency:** <50ms for typical text-block-sized crops
- **Availability:** The `photo` module (which contains `inpaint`) may not be included in default builds; requires a custom build or the full OpenCV.js distribution
- **Quality:** Adequate for white/uniform backgrounds; poor for textured backgrounds

**Implementation sketch:**
```javascript
// After loading OpenCV.js
let src = cv.imread(canvasElement);
let mask = /* binary mask from OCR bboxes */;
let dst = new cv.Mat();
cv.inpaint(src, mask, dst, 3, cv.INPAINT_TELEA);
cv.imshow(outputCanvas, dst);
```

### 6.3 Recommendation for FakeEdit v2

A hybrid approach:
- **White/uniform backgrounds (80%+ of documents):** Use OpenCV.js Telea in-browser. Near-instant, zero server cost.
- **Complex backgrounds:** Send to server-side LaMa or STR model. Fall back gracefully with a loading indicator.
- **Optional:** Ship the LaMa ONNX model as an opt-in "HD inpainting" mode for power users with capable hardware.

---

## 7. Practical Pipeline Architecture

### 7.1 End-to-End Pipeline

```
                    born-digital PDF?
                    /              \
                  YES               NO
                   |                 |
          PyMuPDF redact     Extract page image
                   |                 |
          Insert new text     Classify background
                   |              /       \
                 Done        Simple     Complex
                              |           |
                         OpenCV        LaMa / STR
                         Telea         model
                              \         /
                         Inpainted background
                                 |
                         Render new text
                                 |
                         Composite final image
                                 |
                         Re-insert into PDF (if PDF)
```

### 7.2 Per-Block vs Full-Page Inpainting

| Strategy | Pros | Cons |
|----------|------|------|
| **Per-block** (crop, inpaint, paste) | Faster (smaller inputs), can use fixed 512x512 models, parallel processing | Seam artifacts at crop boundaries, loses full-page context |
| **Full-page** (single inference) | No seams, full context awareness | Slow for high-res pages, may exceed model memory, text-agnostic models waste capacity on non-text regions |
| **Hybrid** (full-page mask, per-block execution) | Best of both worlds | More complex implementation |

**Recommendation:** Use **per-block** with generous padding (extend crop 50-100px beyond the text bbox on each side). This provides local context while keeping inference fast. For adjacent/overlapping text blocks, merge them into a single crop to avoid seam issues.

### 7.3 Caching Strategy

Inpainting is the most expensive step in the pipeline. Cache aggressively:

1. **Clean background cache:** After inpainting all text blocks on a page, store the resulting "clean" background image. Subsequent text edits by the user can skip inpainting entirely and render directly onto the cached clean background.
2. **Per-block cache:** Cache individual inpainted crops keyed by `(page_id, bbox_hash)`. If the user edits only one block, only that block needs re-processing.
3. **Invalidation:** Invalidate when the source image changes or when the user manually marks a region for re-processing.

### 7.4 Quality Verification Pipeline

```python
def verify_inpainting(original, inpainted, mask, ocr_engine):
    """Verify inpainting quality with multiple heuristics."""
    results = {}
    
    # 1. Ghost text detection: re-run OCR on inpainted region
    masked_region = inpainted * (mask > 0)
    detected_text = ocr_engine.detect(masked_region)
    results['ghost_text'] = len(detected_text) > 0
    
    # 2. Color consistency: compare inpainted region stats with surrounding area
    dilated_mask = cv2.dilate(mask, np.ones((20, 20)), iterations=1)
    surround_mask = dilated_mask - mask
    inpainted_mean = inpainted[mask > 0].mean()
    surround_mean = inpainted[surround_mask > 0].mean()
    results['color_shift'] = abs(inpainted_mean - surround_mean)
    
    # 3. Edge discontinuity at mask boundary
    boundary = cv2.dilate(mask, np.ones((3, 3))) - cv2.erode(mask, np.ones((3, 3)))
    boundary_gradient = cv2.Sobel(inpainted, cv2.CV_64F, 1, 0)[boundary > 0]
    results['boundary_discontinuity'] = boundary_gradient.std()
    
    return results
```

---

## 8. Commercial / SaaS Alternatives

### 8.1 Adobe Acrobat Pro

- **Approach:** When opening a scanned document for editing, Acrobat automatically runs OCR, creates an editable text layer, and generates custom fonts matching the original appearance.
- **Internally:** Appears to use OCR + font synthesis + image-layer manipulation. The "Editable Text & Images" mode suggests it does perform some form of background inpainting.
- **Cost:** $23/month (Acrobat Pro)
- **Limitation:** Desktop application, no API for programmatic use

### 8.2 ABBYY FineReader PDF

- **Approach:** Creates a temporary text layer for PDFs without one, making them editable. Uses AI-powered OCR with font style replication.
- **Text editing:** Direct text editing in PDF, including documents without a text layer
- **Cost:** $199/year or enterprise licensing
- **API:** Extensive API support for integration
- **Limitation:** Primarily a desktop tool; API is for document extraction, not pixel-level inpainting

### 8.3 Google Document AI

- **Capabilities:** OCR with 200+ language support, structure recognition (paragraphs, tables, forms), handwriting recognition, font-style detection
- **Text removal:** The API supports "removal and replacement operations" on document elements, but this appears limited to born-digital PDF manipulation, not image inpainting
- **Pricing:** $1.50 per 1,000 pages (OCR processor)
- **Limitation:** No documented image-level text erasure/inpainting capability

### 8.4 Other Notable Services

| Service | Capability | Pricing | Notes |
|---------|-----------|---------|-------|
| **cleanup.pictures** | LaMa-based web tool | Free tier + paid | Uses the same LaMa model we can self-host |
| **PixelPanda** | AI text remover | Free | Web-based, consumer-focused |
| **Replicate** | Stable Diffusion Inpainting API | $0.0023/run | Cloud GPU inference, pay-per-use |
| **remove.bg** | Background removal | Not for text removal | Different use case |

**Verdict:** No commercial service provides a dedicated "document text inpainting" API that surpasses what can be achieved with open-source tools. Adobe and ABBYY handle the full edit pipeline internally but are not embeddable. For FakeEdit v2, self-hosted open-source is both more capable and more cost-effective.

---

## 9. Consolidated Benchmark Comparison

### 9.1 Quality Rankings (SCUT-EnsText)

| Rank | Method | PSNR | MSSIM | FID | FPS | Params | License |
|------|--------|------|-------|-----|-----|--------|---------|
| 1 | Uformer-B + TMIM | 37.42 | 97.70 | -- | ~10 | 50.88M | -- |
| 2 | DiffSTR | 37.25 | 97.98 | -- | 2 | -- | -- |
| 3 | ViTEraser-Base + SegMIM | 37.11 | 97.61 | 10.15 | ~15 | 191.9M | MIT (non-comm) |
| 4 | ViTEraser-Small + SegMIM | 37.08 | 97.62 | 10.16 | ~15 | -- | MIT (non-comm) |
| 5 | ViTEraser-Tiny + SegMIM | 36.80 | 97.55 | 10.79 | ~15 | -- | MIT (non-comm) |
| 6 | GaRNet | 35.45 | 97.14 | 15.50 | -- | -- | Apache-2.0 |
| 7 | CTRNet | 35.20 | 97.36 | 13.99 | -- | -- | MIT |
| 8 | FETNet | 34.53 | 97.01 | -- | -- | -- | -- |
| 9 | PERT | 33.62 | 97.00 | -- | **71** | -- | -- |
| 10 | EraseNet | 32.30 | 95.42 | 19.27 | -- | -- | -- |
| 11 | EnsNet | 29.54 | 92.74 | 32.71 | -- | -- | -- |

### 9.2 Speed vs Quality Tradeoff

```
Quality (PSNR)
  38 |                                    * TMIM+Uformer
     |                              * DiffSTR
  37 |                        * ViTEraser
     |
  36 |
     |                  * GaRNet
  35 |                * CTRNet
     |           * FETNet
  34 |
     |      * PERT
  33 |
     |
  32 |  * EraseNet
     +--+-------+-------+-------+-------+-----> Speed (FPS)
        2       10      15      50      71
```

### 9.3 Practical Comparison for Document Use

| Approach | Doc quality | Complexity | Latency/block | Model size | License |
|----------|-----------|------------|---------------|------------|---------|
| OpenCV Telea | Good (white bg) | Trivial | <10ms | 0 | BSD |
| LaMa | Very good | Low | ~50ms (GPU) | 174 MB | Apache-2.0 |
| PERT | Good | Medium | ~14ms (GPU) | -- | -- |
| GaRNet | Very good | Medium | -- | -- | Apache-2.0 |
| ViTEraser-Tiny | Excellent | High | ~67ms (GPU) | -- | Non-comm |
| TMIM+Uformer-B | Excellent | High | ~100ms (GPU) | ~200 MB | -- |
| DiffSTR | Best | Very high | ~500ms (GPU) | ~4 GB | -- |
| SD Inpainting | Overkill | Very high | ~2-5s (GPU) | ~4 GB | CreativeML |

---

## 10. Recommended Approach for FakeEdit v2

### Tier 1: MVP (1-2 weeks implementation)

**Use LaMa for all inpainting, with OpenCV Telea as a fast fallback.**

Pipeline:
1. Receive OCR bounding boxes from existing PaddleOCR/Tesseract/Surya/docTR services
2. Generate dilated rectangular masks (5px dilation at 300 DPI)
3. For each text block: crop with 50px padding, resize to 512x512, run LaMa, resize back
4. Composite the inpainted crops back into the original image
5. Render new text over the clean background

Implementation:
- Use IOPaint's LaMa integration (`pip install iopaint`) for rapid setup
- Docker: `iopaint start --model=lama --device=cpu --port=8080`
- For white-background detection: check if the median pixel value in the mask's surrounding area is > 240 (near-white). If so, use OpenCV Telea instead of LaMa for speed.

**Effort:** 1-2 weeks. **Quality:** Good for 90%+ of document types.

### Tier 2: Enhanced Quality (2-4 weeks additional)

**Add pixel-level mask refinement and a dedicated STR model.**

Additions:
1. Extract DBNet probability maps from PaddleOCR's detection stage for pixel-precise masks
2. Implement per-block binarization refinement (`refine_mask_from_crop` from Section 4.3)
3. Add GaRNet as a second inpainting model (Apache-2.0 license, good quality, provides multiple pre-trained variants)
4. Implement background complexity classifier to route: simple backgrounds to OpenCV, moderate to LaMa, complex to GaRNet
5. Add ghost-text verification (re-run OCR on inpainted regions)
6. Implement clean background caching

**Effort:** 2-4 weeks. **Quality:** Very good for 95%+ of document types.

### Tier 3: Best Quality (4-8 weeks additional)

**Integrate SOTA STR models and client-side inpainting.**

Additions:
1. Integrate TMIM + Uformer-B for best-in-class text removal (requires training/fine-tuning on document-specific data)
2. Fine-tune on a document-specific dataset (create training pairs from our test fixtures)
3. Add client-side LaMa ONNX for white/simple backgrounds (eliminates server round-trip for 80% of cases)
4. Add OpenCV.js Telea for instant client-side fallback
5. Implement multi-pass inpainting with quality verification loop
6. Add PDF-native text removal path using PyMuPDF for born-digital PDFs

**Effort:** 4-8 weeks. **Quality:** Excellent across all document types.

### Architecture Summary

```
Client (Browser)
  |
  +--> Simple background? --> OpenCV.js Telea (instant, <50ms)
  |
  +--> Moderate background? --> LaMa ONNX (client, ~2-5s)
  |
  +--> Complex background? --> Server API
                                  |
                                  +--> Born-digital PDF? --> PyMuPDF redaction
                                  |
                                  +--> LaMa (default, ~50ms/block)
                                  |
                                  +--> GaRNet / TMIM+Uformer (fallback, ~100ms/block)
```

### Key Dependencies

| Component | Package | Version | License |
|-----------|---------|---------|---------|
| LaMa | `iopaint` or `advimman/lama` | 1.5.x / latest | Apache-2.0 |
| GaRNet | `naver/garnet` | latest | Apache-2.0 |
| OpenCV | `opencv-python` | 4.x | BSD |
| OpenCV.js | `@nicco.io/opencv.js` or custom build | 4.x | BSD |
| LaMa ONNX | `onnxruntime-web` + `Carve/LaMa-ONNX` | latest | Apache-2.0 / MIT |
| PDF handling | `PyMuPDF` (fitz) | 1.24+ | AGPL-3.0 |
| Mask refinement | `opencv-python` + PaddleOCR DBNet | 4.x | BSD / Apache-2.0 |

### Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| LaMa ghost artifacts on complex backgrounds | Fall back to GaRNet/STR model; add verification loop |
| Non-commercial licenses (ViTEraser) | Use Apache-2.0 alternatives (LaMa, GaRNet) |
| Client-side ONNX performance | Make it opt-in; default to server-side |
| PyMuPDF AGPL license | Only used server-side; evaluate pymupdf-commercial license if needed |
| Stamp/annotation overlap | Document as known limitation; suggest manual mask editing in UI |

---

## References

### Papers

1. Suvorov et al., "Resolution-robust Large Mask Inpainting with Fourier Convolutions," WACV 2022. arXiv:2109.07161
2. Li et al., "MAT: Mask-Aware Transformer for Large Hole Image Inpainting," CVPR 2022. arXiv:2203.15270
3. Liu et al., "Don't Forget Me: Accurate Background Recovery for Text Removal via Modeling Local-Global Context," ECCV 2022 (CTRNet)
4. Tursun et al., "The Surprisingly Straightforward Scene Text Removal Method for Any Data," ECCV 2022 (GaRNet)
5. Shannan et al., "ViTEraser: Harnessing the Power of Vision Transformers for Scene Text Removal with SegMIM Pretraining," AAAI 2024. arXiv:2306.12106
6. Wang et al., "PERT: A Progressively Region-based Network for Scene Text Removal," TIP 2023. arXiv:2106.13029
7. Lyu et al., "FETNet: Feature Erasing and Transferring Network for Scene Text Removal," Pattern Recognition 2023. arXiv:2306.09593
8. Du et al., "DeepEraser: Deep Iterative Context Mining for Generic Text Eraser," TMM 2024
9. "DiffSTR: Controlled Diffusion Models for Scene Text Removal," arXiv:2410.21721 (Oct 2024)
10. Wang et al., "Leveraging Text Localization for Scene Text Removal via Text-aware Masked Image Modeling," ECCV 2024 (TMIM)
11. Zhang et al., "EnsNet: Ensconce Text in the Wild," AAAI 2019
12. Liu et al., "EraseNet: End-to-End Text Removal in the Wild," IEEE TIP 2020

### GitHub Repositories

- LaMa: https://github.com/advimman/lama (9.8k stars, Apache-2.0)
- IOPaint: https://github.com/Sanster/IOPaint (22.9k stars, Apache-2.0)
- ViTEraser: https://github.com/shannanyinxiang/ViTEraser (63 stars, MIT non-comm)
- GaRNet: https://github.com/naver/garnet (69 stars, Apache-2.0)
- CTRNet: https://github.com/lcy0604/CTRNet (96 stars, MIT)
- PERT: https://github.com/wangyuxin87/PERT (37 stars)
- FETNet: https://github.com/GuangtaoLyu/FETNet (35 stars)
- DeepEraser: https://github.com/fh2019ustc/DeepEraser (51 stars, MIT)
- TMIM: https://github.com/wzx99/TMIM (12 stars)
- MAT: https://github.com/fenglinglwb/MAT (970 stars, research-only)
- EnsNet: https://github.com/HCIILAB/Scene-Text-Removal (131 stars)
- CRAFT: https://github.com/clovaai/CRAFT-pytorch
- LaMa-ONNX: https://huggingface.co/Carve/LaMa-ONNX
- Inpaint-Anything: https://github.com/geekyutao/Inpaint-Anything
- pikepdf: https://github.com/pikepdf/pikepdf
