# Font Recognition and Estimation for Scanned Documents

> Research document -- April 2026
> Context: FakeEdit v2 downstream consumer needs font metadata to re-render text over scanned images.
>
> **Pipeline step**: Problem 3 in the [editing pipeline](./README.md). Depends on OCR bounding boxes from Problem 1 ([DR-001](./DR-001-ocr-adapter.md)). Output feeds into text re-rendering (Problem 5).
>
> **Related**: [research-inpainting.md](./research-inpainting.md) (background cleanup before re-rendering) · [research-benchmark-gaps.md](./research-benchmark-gaps.md) (bbox accuracy affects font crop quality) · [cost-performance.md](./cost-performance.md) (font detection cost: $0 marginal)

---

## Table of Contents

1. [Font Size Estimation from Bounding Boxes](#1-font-size-estimation-from-bounding-boxes)
2. [Font Family Recognition from Images](#2-font-family-recognition-from-images)
3. [Font Weight and Style Detection](#3-font-weight-and-style-detection)
4. [OCR Engine Font Metadata](#4-ocr-engine-font-metadata)
5. [Font Matching for Rendering](#5-font-matching-for-rendering)
6. [Practical Recommendations](#6-practical-recommendations)

---

## 1. Font Size Estimation from Bounding Boxes

### The Formula

The standard formula for converting a text bounding box height in pixels to typographic points is:

```
font_size_pt = (bbox_height_px / dpi) * 72
```

Where:
- `bbox_height_px` is the height of the OCR bounding box in pixels
- `dpi` is the resolution used when rendering/scanning the page image
- `72` is the number of typographic points per inch

At 72 DPI the conversion is identity (1 px = 1 pt). At the standard scan resolution of 300 DPI, a 12 pt font produces a bounding box height of approximately 50 px: `(50 / 300) * 72 = 12.0`.

### Our Current Implementation

The PaddleOCR adapter already implements this in `services/paddle/src/utils/font_estimator.py`:

```python
def estimate_font_size_pt(bbox_height_px: int, dpi: int) -> float:
    return round((bbox_height_px / dpi) * 72, 1)

def estimate_block_font_size(line_heights: list[int], dpi: int) -> float:
    # Uses median line height across lines in a block
    ...
    return estimate_font_size_pt(int(median), dpi)
```

The approach is sound. Using the **median** line height across lines in a block is a good choice -- it resists outliers from unusually tall/short lines (subscripts, superscripts, or single-character lines).

The Tesseract hOCR parser (`services/tesseract/src/hocr_parser.py`) independently applies the same formula at the word level:

```python
font_size_estimate = (bbox.height * 72) / dpi if bbox.height > 0 else 0.0
```

### Accuracy and Limitations

**Typical accuracy: +/- 1-2 pt for standard body text at 300 DPI.**

Key factors that reduce accuracy:

| Factor | Impact | Mitigation |
|--------|--------|------------|
| Bbox includes ascenders + descenders | Overestimates by ~20% vs. the CSS "font-size" concept | Apply a correction factor of ~0.82-0.88 (varies by font) |
| Bbox includes line spacing | Severe overestimate if bbox is line-level not glyph-tight | Use word-level bboxes or tight line bboxes |
| Low DPI (<200) | Quantization error grows | Scan/render at 300 DPI minimum |
| Very small text (<8pt at 300 DPI) | Bboxes become unreliable | Flag low-confidence estimates |
| Skewed or rotated text | Bbox height inflated | Apply deskew before estimation |

**The ascender/descender problem.** A font's "point size" in CSS/typesetting refers to the em-square, not the full bounding box. The actual rendered glyph height (ascender to descender) is typically 110-130% of the em-square, depending on the font. For example, the Catamaran font uses 1100 + 540 ascender/descender units in a 1000 UPM em-square, so the rendered height is 164% of the nominal point size.

A practical correction:

```python
# Approximate: bbox typically covers ~1.2x the nominal point size
corrected_pt = raw_estimate_pt / 1.2
```

This correction factor varies per font family (1.15 for compact fonts like Arial, 1.25 for fonts with large descenders like Zapfino). Without knowing the font, 1.2 is a reasonable default. The current implementation does **not** apply this correction, which means our estimates are systematically ~20% high relative to the CSS font-size that would reproduce the same visual size.

### Recommendation for Our Codebase

The `estimate_block_font_size` approach is fundamentally correct. To improve accuracy:

1. Apply a configurable correction factor (default 1.2) to account for ascender/descender overshoot.
2. Use word-level bounding boxes where available (tighter fit than line-level).
3. Report the estimate alongside a confidence band (e.g., +/- 2 pt).
4. If the font family is identified (see section 2), use font-specific metrics for a precise correction.

---

## 2. Font Family Recognition from Images

### 2.1 DeepFont (Adobe, 2015)

The foundational paper: *"DeepFont: Identify Your Font from an Image"* (ACMMM 2015). Key ideas:

- Two-subnetwork CNN architecture: low-level feature extraction (shared between synthetic and real data via domain adaptation) and high-level classification.
- Trained on the AdobeVFR dataset (~2,400 font classes).
- Used a Stacked Convolutional Auto-Encoder (SCAE) for unsupervised cross-domain feature learning.

**Not open-sourced by Adobe.** Several community reimplementations exist:

| Repository | Framework | Notes |
|-----------|-----------|-------|
| [robinreni96/Font_Recognition-DeepFont](https://github.com/robinreni96/Font_Recognition-DeepFont) | Keras | Jupyter notebook implementation |
| [bobbyphtr/DeepFont-FontRecognition](https://github.com/bobbyphtr/DeepFont-FontRecognition) | Keras | Includes API wrapper |
| [artset/WhatTheFont](https://github.com/artset/WhatTheFont) | TensorFlow | Reimplementation with custom dataset |
| [Dexterp37/fontina](https://github.com/Dexterp37/fontina) | PyTorch | Most complete; includes synthetic data generation pipeline |
| [twelfth-star/universal-font-recognition](https://github.com/twelfth-star/universal-font-recognition) | PyTorch | Clean DeepFont reimplementation |

**fontina** is the most production-ready of these. It provides:
- Synthetic dataset generator following DeepFont recommendations (noise, blur, affine transforms).
- Two-stage training: unsupervised SCAE pre-training + supervised classification.
- Linux/macOS/Windows support.
- The AdobeVFR dataset is available for non-commercial research.

### 2.2 Modern CNN Font Classifiers

**Storia AI font-classify** -- the strongest open-source option available today.

- Repository: [Storia-AI/font-classify](https://github.com/Storia-AI/font-classify)
- Pre-trained model: [storia/font-classify-onnx](https://huggingface.co/storia/font-classify-onnx) on HuggingFace
- **Classifies ~3,000 Google Fonts**
- ONNX format for efficient inference (no PyTorch dependency at inference time)
- Includes `google_fonts_mapping.tsv` to map model output filenames to official Google Fonts API names
- Sample inference script: `infer_pretrained.py`

This is the most practical option for our use case because:
1. The font set is large (3,000 fonts vs. DeepFont's academic dataset).
2. All fonts are from Google Fonts, which are free and can be bundled for rendering.
3. ONNX format runs on CPU efficiently.

**gaborcselle/font-identifier** -- a simpler alternative.

- Repository: [gaborcselle/font-identifier](https://github.com/gaborcselle/font-identifier)
- HuggingFace: [gaborcselle/font-identifier](https://huggingface.co/gaborcselle/font-identifier)
- ResNet18 architecture, 48 standard fonts, **96.3% accuracy** on test set.
- Built in 1 day; useful as a reference implementation or for a serif/sans-serif coarse classifier.
- Dataset: [gaborcselle/font-examples](https://huggingface.co/datasets/gaborcselle/font-examples)

**ctensmeyer/font_classification** -- academic reference.

- Repository: [ctensmeyer/font_classification](https://github.com/ctensmeyer/font_classification)
- Paper: *"Convolutional Neural Networks for Font Classification"* (ICDAR 2017)
- ResNet-50 on 227x227 patches of document images.
- Demonstrated state-of-the-art on both Arabic and Latin font classification.

**OCR-D/ocrd_typegroups_classifier** -- historical document specialist.

- Repository: [OCR-D/ocrd_typegroups_classifier](https://github.com/OCR-D/ocrd_typegroups_classifier)
- DenseNet-121, classifies 12 font family groups for historical documents.
- Integrates with the OCR-D ecosystem.

### 2.3 Vision-Language Models for Font Recognition

A COLM 2025 paper systematically evaluated VLMs on font recognition:

**"Texture or Semantics? Vision-Language Models Get Lost in Font Recognition"** ([arXiv:2503.23768](https://arxiv.org/abs/2503.23768))

- Code: [Lizhecheng02/VLM4Font](https://github.com/Lizhecheng02/VLM4Font)
- Benchmark: Font Recognition Benchmark (FRB), 15 common fonts.

Results (zero-shot, easy version):

| Model | Accuracy |
|-------|----------|
| Claude 3.5 Sonnet | ~31% (best) |
| GPT-4o | ~20% |
| Gemini 2.0 Flash | ~15% |
| GPT-4o-mini | ~0% |
| Open-source VLMs | <15% |

**Conclusion: VLMs are not viable for font recognition.** Even the best model achieves only ~31% accuracy on 15 common fonts. Performance degrades further under Stroop-effect conditions. Few-shot prompting and chain-of-thought provide minimal improvement.

### 2.4 Google Fonts Visual Similarity Matching

An alternative to classification: use Vision Transformer embeddings to find the nearest Google Font.

Approach (described at [ideatum.ai](https://ideatum.ai/blog/better-font-list.html)):
1. Render each of the 1,600+ Google Fonts using a pangram ("The quick brown fox...").
2. Extract feature vectors from a Vision Transformer (e.g., ViT-B/16).
3. For an input image, extract the same feature vector.
4. Find the k-nearest fonts by cosine or Euclidean distance.

This approach is more flexible than classification because:
- It handles fonts outside the training set (finds the closest match).
- It can be updated by simply adding new font embeddings to the database.
- Distance provides a confidence/similarity score.

### 2.5 Commercial Font Identification Services

| Service | Font DB Size | Self-Hosted | API |
|---------|-------------|-------------|-----|
| [MyFonts WhatTheFont](https://www.myfonts.com/pages/whatthefont/) | 130,000+ | No | Yes (paid) |
| [WhatFontIs](https://www.whatfontis.com) | 900,000+ | No | Yes (paid) |
| [Font Squirrel Matcherator](https://www.fontsquirrel.com/matcherator) | Free fonts only | No | No |
| [Fontspring Matcherator](https://www.fontspring.com/matcherator) | Licensed fonts | No | No |

None of these are self-hostable. They are useful for validation/benchmarking but not for production integration.

---

## 3. Font Weight and Style Detection

### 3.1 Bold Detection

**Stroke Width Transform (SWT).** The classic approach from Epshtein, Ofek, Wexler (CVPR 2010):
- Computes per-pixel stroke width in the image.
- Bold text has consistently wider strokes than regular weight.
- Repository: [sunsided/stroke-width-transform](https://github.com/sunsided/stroke-width-transform) (Python)
- Fast and language-agnostic, but requires binarized text images.

**CNN-based approach.** Train a binary classifier (bold vs. regular) on rendered text patches:
- Use the same CNN architecture as font classification (ResNet-18 is sufficient).
- Training data is trivially generated: render text in bold and regular weights.
- Can be extended to a regression model for font-weight on the 100-900 scale.

**Heuristic approach (no ML).** Compare the ratio of black pixels to bounding box area:
- Bold text has higher ink density than regular text at the same font size.
- Threshold: typically >45% ink density suggests bold for Latin text.

### 3.2 Italic Detection

- **Shear/skew analysis.** Italic text has a consistent rightward lean of ~12 degrees. Measure the average angle of vertical strokes using projection profiles or Hough transforms.
- **CNN classifier.** Similar to bold detection; binary classification on rendered italic vs. upright patches.
- **Aspect ratio heuristic.** Italic text in the same font/size typically has a slightly wider bounding box than upright text.

### 3.3 Serif vs. Sans-Serif Classification

This is the most tractable sub-problem and can be solved reliably:

- **CNN approach.** A simple ResNet-18 trained on serif vs. sans-serif patches achieves >95% accuracy. Training data: render text in known serif and sans-serif fonts.
- **Stroke terminal analysis.** Detect the presence of serifs by analyzing stroke endpoints -- serifs create characteristic "feet" at the base of vertical strokes.
- **Pre-trained models.** The font classification models above (Storia, gaborcselle) implicitly solve this since their output includes the font family, from which serif/sans-serif can be derived.

### 3.4 Practical Approach

For FakeEdit v2, a two-stage pipeline:

1. **Coarse classification** (serif / sans-serif / monospace / script / decorative) using a lightweight CNN. This narrows the search space for font matching.
2. **Fine-grained properties** (bold, italic) using either:
   - SWT + ink density heuristics (no ML, fast), or
   - A small multi-label CNN trained on rendered text.

---

## 4. OCR Engine Font Metadata

### 4.1 Google Document AI (Enterprise OCR)

**The only commercial OCR engine with comprehensive font metadata.**

Enable with `compute_style_info: true` in the `OcrConfig.premium_features` of the process request.

Response fields in `Document.pages[].tokens[].styleInfo`:

| Field | Type | Description |
|-------|------|-------------|
| `fontSize` | float | Font size in points |
| `pixelFontSize` | float | Font size in pixels |
| `fontType` | string | `"SERIF"`, `"SANS_SERIF"`, `"MONOSPACE"`, etc. |
| `bold` | bool | True if font weight >= 700 |
| `italic` | bool | Italic style detected |
| `underlined` | bool | Underlined text |
| `strikeout` | bool | Strikethrough text |
| `handwritten` | bool | Handwritten text detected |
| `fontWeight` | int | 100 (thin) to 1000 (ultra-heavy); 400 = normal, 700 = bold |
| `textColor` | Color | RGB text color |
| `backgroundColor` | Color | RGB background color |
| `letterSpacing` | float | Inter-character spacing |

**Pricing note:** This is a premium feature. Enterprise Document OCR is priced higher than the standard tier. As of 2026, approximately $0.065 per page for the premium tier (vs. $0.0015 for standard OCR).

- Docs: [Enterprise Document OCR](https://docs.cloud.google.com/document-ai/docs/enterprise-document-ocr)
- Python SDK: `google.cloud.documentai_v1.types.Document.Page.Token.StyleInfo`

### 4.2 Azure Document Intelligence

Supports font style detection as an **add-on capability** (feature flag `STYLE_FONT`).

Response fields in the styles collection:

| Field | Type | Description |
|-------|------|-------------|
| `fontStyle` | string | `"italic"` or `"normal"` |
| `fontWeight` | string | `"bold"` or `"normal"` |
| `isHandwritten` | bool | Handwritten content flag |
| `similarFontFamily` | string | Closest matching font family name |
| `confidence` | float | Confidence score per style span |

**No font size, no font type (serif/sans-serif) in the response.** Font info is limited to style and weight attributes. The `similarFontFamily` field is useful but not well-documented in terms of which fonts it can match.

API version: `2024-11-30` (GA). Enable via `features: ["STYLE_FONT"]` in the analyze request.

- Docs: [Add-on capabilities](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/concept/add-on-capabilities?view=doc-intel-4.0.0)

### 4.3 AWS Textract

**Does not output any font metadata.** No font size, weight, style, or family information.

The API returns text content, bounding boxes, and confidence scores only. Workarounds require external ML models or heuristic analysis of bounding box dimensions.

- Docs: [Block API Reference](https://docs.aws.amazon.com/textract/latest/dg/API_Block.html)

### 4.4 ABBYY FineReader Engine

**Outputs font name, size, and style** through the `CharParams` XML export.

| Field | Source | Description |
|-------|--------|-------------|
| Font name | `CharParams` XML | Detected font family name |
| Font size | `xmlWriteCharFormatting` | Size in points |
| Bold | Style attributes | Detected from synthesis stage |
| Italic | Style attributes | Detected from synthesis stage |

**Requirements:**
- The corresponding fonts must be installed on the system for name detection.
- Use `CustomFontSet::setFolder()` to specify a font directory.
- Available in FineReader Engine SDK (commercial license).
- Can train custom patterns for non-standard fonts via the "Recognition with Training" feature.

**Limitation:** Not available as a cloud API in the same way as Google/Azure/AWS. Requires SDK integration (C++, .NET, Java, or Python wrapper).

- Docs: [Extract fonts in FineReader Engine](https://support.abbyy.com/hc/en-us/articles/360011836019-How-to-to-extract-fonts-in-FineReader-Engine)

### 4.5 Tesseract (Open Source)

**Limited and unreliable font detection.**

Tesseract can output font attributes via two mechanisms:

1. **hOCR output** with `--hocr_font_info 1`: adds `x_font <name>` to word-level title attributes.
2. **API**: `ResultIterator::WordFontAttributes()` returns font name, bold, italic, underlined, monospace, serif, smallcaps, pointsize, and font_id.

**Known issues:**
- Tesseract v5 with v5 tessdata often returns empty font names ([issue #3820](https://github.com/tesseract-ocr/tesseract/issues/3820)).
- `WordFontAttributes` returns null with v4+ tessdata; works only with v3.04 tessdata.
- Font name detection depends on training data -- LSTM models (default in v4/v5) have degraded font attribute support compared to the legacy engine.
- The font name, when returned, is the name of the training font that best matched, not necessarily the actual document font.

**Our hOCR parser** (`services/tesseract/src/hocr_parser.py`) does not currently parse `x_font` attributes. It could be extended, but the data would be unreliable.

### 4.6 PaddleOCR / Surya / docTR (Open Source)

**None of these engines output font metadata.** They provide text content, bounding boxes, and confidence scores only.

### Summary Matrix

| Engine | Font Size | Font Family | Bold/Italic | Serif/Sans | Self-Hosted | Cost |
|--------|-----------|-------------|-------------|------------|-------------|------|
| Google Document AI | Yes | Type only (not name) | Yes | Yes | No | ~$0.065/page |
| Azure Document Intelligence | No | similarFontFamily | Yes | No | No | ~$0.01/page |
| ABBYY FineReader | Yes | Yes (font name) | Yes | Implicit | Yes (SDK) | License fee |
| AWS Textract | No | No | No | No | No | ~$0.0015/page |
| Tesseract | Unreliable | Unreliable | Unreliable | Unreliable | Yes | Free |
| PaddleOCR | No | No | No | No | Yes | Free |
| Surya | No | No | No | No | Yes | Free |
| docTR | No | No | No | No | Yes | Free |

---

## 5. Font Matching for Rendering

Once a font family is identified (or a coarse category like serif/sans-serif), the next step is mapping to an available rendering font.

### 5.1 PANOSE Classification

[PANOSE](https://en.wikipedia.org/wiki/PANOSE) is a 10-digit classification system for typefaces based on visual characteristics.

For Latin Text fonts, the 10 digits encode:
1. Family Kind (e.g., Latin Text, Latin Script, Latin Decorative)
2. Serif Style (e.g., cove, square, thin, bone)
3. Weight (very light to extra black)
4. Proportion (old style to monospaced)
5. Contrast (none to very high)
6. Stroke Variation
7. Arm Style
8. Letterform
9. Midline
10. X-height

**Font distance** is computed as a weighted sum of per-digit differences:

```python
def panose_distance(a: list[int], b: list[int]) -> float:
    weights = [0, 2, 6, 4, 5, 5, 3, 6, 2, 4]  # Typographic importance
    return sum(w * abs(ai - bi) for w, ai, bi in zip(weights, a, b))
```

Most TrueType/OpenType fonts include a PANOSE number in the OS/2 table. This can be extracted with `fonttools`:

```python
from fontTools.ttLib import TTFont
font = TTFont("MyFont.ttf")
panose = font["OS/2"].panose  # 10-byte PANOSE classification
```

**Limitation:** Many fonts (especially free/Google Fonts) have PANOSE values set to all zeros, making this approach unreliable unless you curate a database of correct PANOSE values.

- Official spec: [monotype.github.io/panose](https://monotype.github.io/panose/pan2.htm)
- W3C reference: [PANOSE 2.0 White Paper](https://www.w3.org/Fonts/Panose/pan2.html)

### 5.2 Visual Embedding Similarity

A more robust modern approach using Vision Transformers (see section 2.4):

1. **Build a font embedding database.** For each available rendering font:
   - Render a set of canonical strings (alphabet, pangram, digits).
   - Extract feature vectors from a pre-trained ViT (e.g., CLIP, DINOv2).
   - Store the averaged embedding vector.

2. **At inference time:**
   - Crop the text region from the scanned document.
   - Extract the same feature vector.
   - Find the k-nearest fonts by cosine similarity.

This naturally handles the "closest available font" problem -- if the exact font is not in the rendering set, the most visually similar one is returned.

### 5.3 Google Fonts as the Rendering Font Set

For FakeEdit v2, Google Fonts is the ideal rendering font library:

- **1,600+ fonts**, all free and open-source.
- Available via CDN (`fonts.googleapis.com`) or self-hosted.
- The Storia AI model already maps to Google Fonts names.
- CSS `@font-face` integration is straightforward.
- Font files can be bundled with the application.

A fallback mapping for when font identification fails:

| Detected Category | Default Google Font | Rationale |
|-------------------|-------------------|-----------|
| Serif | Noto Serif | Neutral, good coverage |
| Sans-serif | Noto Sans | Matches system sans-serif |
| Monospace | Noto Sans Mono | Consistent with Noto family |
| Script/Cursive | Dancing Script | Common cursive stand-in |
| Decorative | Roboto | Safe fallback |
| Unknown | Noto Sans | Safe default |

### 5.4 CSS Font Matching Algorithm

The CSS font matching spec (CSS Fonts Module Level 4) defines a priority order:

1. `font-family` name (exact match)
2. `font-style` (italic vs. normal)
3. `font-weight` (numeric weight)
4. `font-size`

For re-rendering, we can construct a CSS rule from detected properties:

```css
.detected-text {
  font-family: "Detected Font", "Category Fallback", serif;
  font-size: 12pt;
  font-weight: 700;
  font-style: italic;
}
```

---

## 6. Practical Recommendations

### Ranked by Feasibility and Accuracy

#### Tier 1: Implement Now (High feasibility, good accuracy)

**1. Fix the font size estimation correction factor.**
- Effort: 1-2 hours.
- Apply a 1.2x correction divisor to `estimate_font_size_pt` to account for ascender/descender overshoot.
- Add a configuration parameter so it can be tuned per-font if the font family is known.

**2. Add serif/sans-serif/monospace coarse classification.**
- Effort: 1-2 days.
- Train a simple ResNet-18 on rendered text patches (3 classes: serif, sans-serif, monospace).
- Training data is free to generate by rendering Google Fonts text.
- Integrate as a lightweight classifier that runs on cropped text blocks.

**3. Integrate the Storia AI font-classify model.**
- Effort: 2-3 days.
- Model: [storia/font-classify-onnx](https://huggingface.co/storia/font-classify-onnx) (~3,000 Google Fonts)
- ONNX runtime runs on CPU; no GPU required.
- Use `google_fonts_mapping.tsv` to map output to Google Fonts API names.
- Run per-block: crop the block region from the page image, feed to the model.

#### Tier 2: Build Next (Medium feasibility, high accuracy)

**4. Bold/italic detection via SWT + ink density heuristics.**
- Effort: 3-5 days.
- Stroke Width Transform: [sunsided/stroke-width-transform](https://github.com/sunsided/stroke-width-transform)
- Ink density ratio as a secondary signal.
- No ML training required; rule-based with tunable thresholds.

**5. Font embedding similarity search.**
- Effort: 1-2 weeks.
- Pre-compute ViT embeddings for all Google Fonts.
- At inference, extract embedding from cropped text image, nearest-neighbor search.
- More robust than classification for fonts outside the training set.
- Can use CLIP or DINOv2 as the backbone.

#### Tier 3: Evaluate for High-Accuracy Use Cases

**6. Google Document AI for font metadata (commercial fallback).**
- Effort: 1-2 days to integrate.
- Best available font metadata but at $0.065/page.
- Useful as a validation oracle or for high-value documents.
- Returns fontSize, fontType, fontWeight, bold, italic, letterSpacing.

**7. Train a custom DeepFont-like model using fontina.**
- Effort: 2-4 weeks.
- Use [Dexterp37/fontina](https://github.com/Dexterp37/fontina) as the training framework.
- Generate synthetic training data from the exact font set used by FakeEdit.
- Higher accuracy than generic models for the specific rendering font set.

#### Not Recommended

- **VLMs (GPT-4o, Claude, Gemini) for font recognition.** Maximum ~31% accuracy on 15 fonts (COLM 2025). Not viable.
- **Tesseract font attributes.** Broken in v5 with current tessdata. Unreliable.
- **WhatTheFont / WhatFontIs APIs.** Not self-hosted, paid per query, latency overhead.

### Proposed Architecture

```
Scanned PDF
    |
    v
OCR Engine (PaddleOCR/Tesseract)
    |
    v
Per-block text + bounding boxes
    |
    +---> Font Size Estimator (bbox height / DPI * 72 / 1.2)
    |         => font_size_pt
    |
    +---> Coarse Classifier (ResNet-18, serif/sans/mono)
    |         => font_category
    |
    +---> Font Identifier (Storia ONNX, 3000 Google Fonts)
    |         => font_family_name, confidence
    |
    +---> Style Detector (SWT + ink density)
    |         => is_bold, is_italic
    |
    v
Font Metadata Bundle
    {
      "font_size_pt": 12.0,
      "font_family": "Merriweather",
      "font_category": "serif",
      "font_weight": 700,
      "font_style": "italic",
      "confidence": 0.87,
      "fallback_font": "Noto Serif"
    }
    |
    v
FakeEdit v2 Renderer
    => CSS: font-family: "Merriweather", "Noto Serif", serif;
           font-size: 12pt; font-weight: 700; font-style: italic;
```

### Key Repositories and Resources

| Resource | URL |
|----------|-----|
| Storia AI font-classify | https://github.com/Storia-AI/font-classify |
| Storia ONNX model | https://huggingface.co/storia/font-classify-onnx |
| fontina (PyTorch DeepFont) | https://github.com/Dexterp37/fontina |
| gaborcselle font-identifier | https://github.com/gaborcselle/font-identifier |
| Stroke Width Transform (Python) | https://github.com/sunsided/stroke-width-transform |
| VLM4Font benchmark | https://github.com/Lizhecheng02/VLM4Font |
| COLM 2025 VLM font paper | https://arxiv.org/abs/2503.23768 |
| Font Classification CNN (ICDAR 2017) | https://github.com/ctensmeyer/font_classification |
| OCR-D typegroups classifier | https://github.com/OCR-D/ocrd_typegroups_classifier |
| HuggingFace VFR Dataset | https://huggingface.co/datasets/Mcholo/VFRDataset |
| PANOSE spec | https://monotype.github.io/panose/pan2.htm |
| Google Document AI StyleInfo | https://cloud.google.com/python/docs/reference/documentai/latest/google.cloud.documentai_v1.types.Document.Page.Token.StyleInfo |
| Azure Font Style add-on | https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/concept/add-on-capabilities |
| ABBYY font extraction | https://support.abbyy.com/hc/en-us/articles/360011836019 |
| Google Fonts | https://fonts.google.com/ |
