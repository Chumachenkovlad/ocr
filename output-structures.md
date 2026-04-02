# OCR output structures

Real output shapes from each evaluated provider, annotated with adapter implications.

Related: [DR-001-ocr-adapter.md](./DR-001-ocr-adapter.md) · [comparison.md](./comparison.md)

---

## Difficulty legend

| Rating | Meaning |
|--------|---------|
| 🟢 Easy | Flat or predictable structure, minimal parsing |
| 🟡 Medium | Some nesting or custom types, manageable |
| 🔴 Hard | Deep nesting, custom objects, ID graphs, or inconsistent shape |

---

## 1. Tesseract — 🟡 Medium

### Default string output

```python
import pytesseract
from PIL import Image

text = pytesseract.image_to_string(Image.open("doc.png"))
# Plain string — simple but zero structure
```

### `image_to_data()` — TSV / DataFrame

```python
data = pytesseract.image_to_data(
    Image.open("doc.png"),
    output_type=pytesseract.Output.DATAFRAME
)
```

```
level  page_num  block_num  par_num  line_num  word_num  left  top  width  height  conf  text
    1         1          0        0         0         0     0    0   1280     960    -1
    2         1          1        0         0         0    36   36   1208     888    -1
    5         1          1        1         1         1    36   36     72      24    96  Hello
    5         1          1        1         1         2   120   36     48      24    91  World
```

### `image_to_boxes()` — character-level bounding boxes

```
H 36 924 48 936 0
e 48 924 57 936 0
```

### Adapter notes

- `level` integers 1–5 map to: page → block → paragraph → line → word. Filter by `level == 5` for words.
- `conf == -1` marks structural separator rows — drop before use.
- Coordinate system in `image_to_boxes()` is **bottom-left origin** (Y axis inverted) — inconsistent with all other methods.
- No table or layout data.

---

## 2. PaddleOCR — 🟡 Medium

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')
result = ocr.ocr("doc.png", cls=True)
```

```python
[                                           # list of pages
  [                                         # page 0 — list of lines
    [
      [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],   # quadrilateral bounding box (4 points)
      ("Hello World", 0.9823)               # (text, confidence)
    ],
    [
      [[36,36],[180,36],[180,60],[36,60]],
      ("Invoice #1042", 0.9741)
    ],
  ]
]
```

### Table output — PPStructure

```python
from paddleocr import PPStructure

engine = PPStructure(table=True, ocr=True, lang='en')
result = engine("doc.png")

# Each element:
{
  "type": "table",            # "text" | "title" | "figure" | "figure_caption"
  "bbox": [x1, y1, x2, y2],
  "res": {
    "html": "<table><tr><td>Cell A</td><td>Cell B</td></tr></table>",
    "cell_bbox": [[x1,y1,x2,y2], ...]
  }
}
```

### Adapter notes

- Bounding box is a **4-point quadrilateral** — compute `min/max` to get an axis-aligned rect.
- Table output is a **raw HTML string** — must parse HTML to get structured cells.
- `type` field only present in PPStructure, not plain `PaddleOCR`.
- Multi-page PDFs return one result per page; pages are not merged automatically.

---

## 3. EasyOCR — 🟢 Easy

```python
import easyocr

reader = easyocr.Reader(['en', 'uk'])
result = reader.readtext("doc.png")
```

```python
[
  ([[36,36],[180,36],[180,60],[36,60]], "Hello World",   0.9823),
  ([[36,80],[320,80],[320,104],[36,104]], "Invoice #1042", 0.9741),
]
# Each item: (bounding_box, text, confidence)
# bounding_box: 4 [x,y] corner points
```

### Flat text only

```python
result = reader.readtext("doc.png", detail=0)
# ["Hello World", "Invoice #1042"]
```

### Adapter notes

- Flat list — no paragraph, block, or line grouping.
- Bounding box is a 4-point quad (same as PaddleOCR).
- No table detection or layout hierarchy.
- Paragraph reconstruction must be done manually by grouping on Y proximity.

---

## 4. docTR ✦ — 🔴 Hard (structure) / 🟡 Medium (after export)

```python
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

model = ocr_predictor(pretrained=True)
doc = DocumentFile.from_pdf("doc.pdf")
result = model(doc)
output = result.export()    # convert to plain dict — recommended entry point
```

```json
{
  "pages": [
    {
      "page_idx": 0,
      "dimensions": [1080, 760],
      "blocks": [
        {
          "geometry": [[0.05, 0.04], [0.85, 0.12]],
          "lines": [
            {
              "geometry": [[0.05, 0.04], [0.75, 0.08]],
              "words": [
                {
                  "value": "Hello",
                  "confidence": 0.99,
                  "geometry": [[0.05, 0.04], [0.15, 0.08]]
                },
                {
                  "value": "World",
                  "confidence": 0.97,
                  "geometry": [[0.17, 0.04], [0.30, 0.08]]
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

### Adapter notes

- **Geometry is normalised (0–1)** — multiply by `dimensions` (width, height) to get pixels.
- Geometry format is `[[x_min, y_min], [x_max, y_max]]` — two corners, not four.
- 4-level nesting: page → block → line → word.
- `result.render()` gives plain text but discards all structure — use `.export()` instead.
- No table output from base model; requires separate layout model.

### Denormalisation helper

```python
def denorm(geo, width, height) -> BoundingBox:
    (x1, y1), (x2, y2) = geo
    return BoundingBox(x1=x1*width, y1=y1*height, x2=x2*width, y2=y2*height)
```

---

## 5. Surya ✦ — 🟡 Medium

```python
from surya.ocr import run_ocr
from surya.model.detection.model import load_model as load_det_model
from surya.model.recognition.model import load_model as load_rec_model

det_model, det_processor = load_det_model(), load_det_model()
rec_model, rec_processor = load_rec_model(), load_rec_model()

predictions = run_ocr(
    [image], [["en"]],
    det_model, det_processor,
    rec_model, rec_processor
)
```

```python
[
  OCRResult(
    text_lines=[
      TextLine(
        text="Hello World",
        confidence=0.97,
        bbox=[36, 36, 180, 60],              # [x1, y1, x2, y2] — standard pixel rect
        polygon=[[36,36],[180,36],[180,60],[36,60]]
      ),
    ],
    image_bbox=[0, 0, 1280, 960]
  )
]
```

### Layout detection (separate step)

```python
from surya.layout import batch_layout_detection

layout = batch_layout_detection([image], model, processor)

LayoutResult(
  bboxes=[
    LayoutBox(bbox=[x1,y1,x2,y2], label="Text",  position=0),
    LayoutBox(bbox=[x1,y1,x2,y2], label="Table", position=1),
  ],
  image_bbox=[0, 0, 1280, 960]
)
```

### Adapter notes

- Returns custom `OCRResult` / `TextLine` objects — access via attributes, not dict keys.
- Import paths differ between versions (`0.4.x` vs `0.6.x`) — pin the version.
- No native paragraph grouping — only flat line list.
- Layout and OCR are separate pipeline steps; combine manually in the adapter.
- `bbox` is a standard pixel rect — no normalisation needed. This is the lowest-cost adapter of the five.

---

## 6. AWS Textract ✦ — 🔴 Hard

```python
import boto3

client = boto3.client("textract", region_name="us-east-1")
response = client.analyze_document(
    Document={"Bytes": open("doc.png","rb").read()},
    FeatureTypes=["TABLES", "FORMS"]
)
```

### Block structure

The entire document is a **flat list of Blocks**. Hierarchy is expressed via ID references.

```json
{
  "Blocks": [
    {
      "BlockType": "PAGE",
      "Id": "abc-001",
      "Relationships": [{"Type": "CHILD", "Ids": ["abc-002", "abc-003"]}]
    },
    {
      "BlockType": "LINE",
      "Id": "abc-002",
      "Text": "Hello World",
      "Confidence": 98.23,
      "Geometry": {
        "BoundingBox": {"Left": 0.028, "Top": 0.037, "Width": 0.112, "Height": 0.025},
        "Polygon": [{"X":0.028,"Y":0.037},{"X":0.140,"Y":0.037},{"X":0.140,"Y":0.062},{"X":0.028,"Y":0.062}]
      },
      "Relationships": [{"Type": "CHILD", "Ids": ["abc-010"]}]
    },
    {
      "BlockType": "WORD",
      "Id": "abc-010",
      "Text": "Hello",
      "Confidence": 99.1,
      "Geometry": {"BoundingBox": {"Left": 0.028, "Top": 0.037, "Width": 0.05, "Height": 0.025}}
    },
    {
      "BlockType": "TABLE",
      "Id": "abc-050",
      "Relationships": [{"Type": "CHILD", "Ids": ["abc-060", "abc-061"]}]
    },
    {
      "BlockType": "CELL",
      "Id": "abc-060",
      "RowIndex": 1,
      "ColumnIndex": 1,
      "RowSpan": 1,
      "ColumnSpan": 1,
      "Relationships": [{"Type": "CHILD", "Ids": ["abc-070"]}]
    },
    {
      "BlockType": "KEY_VALUE_SET",
      "EntityTypes": ["KEY"],
      "Id": "abc-100",
      "Relationships": [
        {"Type": "VALUE",  "Ids": ["abc-101"]},
        {"Type": "CHILD",  "Ids": ["abc-102"]}
      ]
    }
  ]
}
```

### Tree reconstruction pattern

```python
block_map = {b["Id"]: b for b in response["Blocks"]}

def get_children(block):
    for rel in block.get("Relationships", []):
        if rel["Type"] == "CHILD":
            for child_id in rel["Ids"]:
                yield block_map[child_id]
```

### Adapter notes

- **No tree in the response** — must build it yourself by following `Relationships[].Ids`.
- `BoundingBox` is **normalised (0–1)** — multiply by page dimensions.
- Table reconstruction: TABLE → CHILD → CELL → CHILD → WORD → concatenate text.
- `KEY_VALUE_SET` links KEY and VALUE via separate relationship types — two lookups per pair.
- Multi-page PDFs require the async API: `start_document_analysis` → poll `get_document_analysis` → paginate with `NextToken`.
- Confidence is 0–100 (not 0–1) — normalise when mapping to canonical model.

### Denormalisation helper

```python
def denorm_textract(bb: dict, page_w: float, page_h: float) -> BoundingBox:
    return BoundingBox(
        x1=bb["Left"] * page_w,
        y1=bb["Top"] * page_h,
        x2=(bb["Left"] + bb["Width"]) * page_w,
        y2=(bb["Top"] + bb["Height"]) * page_h,
    )
```

---

## Quick reference

| Provider | Output type | Nesting | Bbox format | Normalised | Table cells | Confidence range |
|----------|------------|---------|-------------|------------|-------------|-----------------|
| Tesseract | DataFrame | Flat (level 1–5) | x, y, w, h | No | No | 0–100 |
| PaddleOCR | List of lists | 2 levels | Quad (4 pts) | No | HTML string | 0–1 |
| EasyOCR | List of tuples | Flat | Quad (4 pts) | No | No | 0–1 |
| docTR ✦ | Dict (via export) | 4 levels | `[[x1,y1],[x2,y2]]` | Yes | No | 0–1 |
| Surya ✦ | Custom objects | 2 levels | `[x1,y1,x2,y2]` | No | Separate step | 0–1 |
| Textract ✦ | Flat Block list | Flat + ID graph | Normalised rect | Yes | CELL blocks | 0–100 |
