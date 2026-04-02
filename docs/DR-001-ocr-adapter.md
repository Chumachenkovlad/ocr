# DR-001 — OCR provider adapter layer

| Field | Value |
|-------|-------|
| Status | Draft |
| Date | 2026-03-30 |
| Deciders | TBD |
| Related docs | [comparison.md](./comparison.md) · [output-structures.md](./output-structures.md) · [cost-performance.md](./cost-performance.md) |

---

## Context

We need OCR as a capability in the product. The OCR market is evolving fast — open-source models improve monthly, cloud pricing changes, and accuracy benchmarks shift between document types. Locking into a single provider at integration time creates migration cost later.

The goal is to integrate OCR behind an adapter so the provider is a configuration detail, not a structural dependency.

---

## Problem statement

Without an abstraction layer:

- Switching providers requires changes across multiple call sites
- Each provider returns a different output shape (see [output-structures.md](./output-structures.md))
- Testing requires a live provider or per-provider mocks
- Benchmarking providers against each other requires parallel integrations

---

## Goals

- Define a single internal OCR interface the rest of the codebase calls
- Normalize all provider outputs into one canonical model
- Make provider selection a config/env concern, not a code change
- Support running multiple providers in parallel for evaluation
- Keep adapter code isolated and independently testable

## Non-goals

- Training or fine-tuning models
- Building a UI for OCR result review (separate concern)
- Supporting every possible provider on day one

---

## Provider selection

> **Update (2026-04-02):** After benchmarking all four self-hosted engines (see [tech-overview.md](./tech-overview.md)), the initial selection has been revised. PaddleOCR replaces Surya as the primary self-hosted pick due to superior layout analysis and reading order reconstruction. See [Decision log](#decision-log) for details.

Four providers selected for the evaluation. Full evaluation matrix in [comparison.md](./comparison.md). Selection rationale below.

### Selection criteria

| Criterion | Weight | Rationale |
|-----------|--------|-----------|
| Output structure parsability | High | Directly affects adapter complexity |
| Layout + table detection | High | Core document use case |
| Self-hosted option | High | Data residency and cost control |
| Accuracy on mixed documents | Medium | Benchmarked against real samples |
| Community / maintenance health | Medium | Long-term reliability |
| License | Medium | Distribution compatibility |
| Latency | Low | Can be optimised later |

---

### Selected: Surya

**Role: open-source primary**

| Aspect | Detail |
|--------|--------|
| License | GPL-3.0 ⚠️ verify before starting |
| Deployment | Self-hosted, Docker, GPU optional |
| Adapter complexity | Low — standard rect bbox, no normalisation, 2-level flat output |
| Table support | Via separate layout step |
| Languages | 90+ |

Surya has the best balance of modern accuracy and clean output structure among open-source options. Bounding boxes are axis-aligned pixel rects — adapter normalisation is trivial. Layout detection and OCR are separate pipeline steps which maps naturally to an adapter that composes them.

**Risk:** GPL-3.0 must be verified against our distribution model before committing.

---

### Selected: docTR

**Role: open-source alternative / accuracy reference**

| Aspect | Detail |
|--------|--------|
| License | Apache 2.0 |
| Deployment | Self-hosted, Docker, CPU + GPU |
| Adapter complexity | Medium — normalised coords (0–1) require one denorm step |
| Table support | Requires separate layout model |
| Languages | 20+ |

Apache 2.0 eliminates the GPL concern. Outputs a proper hierarchy (block → line → word) with word-level confidence scores, which maps cleanly to the canonical model. If Surya's license is blocked, docTR becomes the primary open-source pick.

**Risk:** Smaller community than PaddleOCR. If multilingual documents beyond 20 languages are a hard requirement this may be replaced in phase 2.

---

### Selected: AWS Textract

**Role: cloud reference — tables and forms**

| Aspect | Detail |
|--------|--------|
| License | Commercial (AWS) |
| Deployment | Cloud-only |
| Adapter complexity | High — flat Block list linked by IDs, tree must be reconstructed |
| Table support | Native, structured CELL blocks with row/col index |
| Languages | ~12 (English-centric) |

Strongest available option for table and form extraction. Used as the cloud-side accuracy reference. The adapter is the most complex of the three, which makes it a good stress test for the abstraction — if the canonical model handles Textract, it handles everything.

**Risk:** Cloud-only. Any document sent leaves the perimeter. Must be gated behind a feature flag and used only with non-sensitive document types during the experiment phase. Multi-page PDFs require async polling.

---

### Added (2026-04-02): PaddleOCR

**Role: open-source primary — layout + reading order**

| Aspect | Detail |
|--------|--------|
| License | Apache 2.0 |
| Deployment | Self-hosted, Docker, CPU + GPU |
| Adapter complexity | Medium — quadrilateral bbox, structure mode has layout regions |
| Table support | Native HTML output via PP-StructureV3 |
| Languages | 80+ |
| Reading order | **Best available** — 6-layer transformer pointer network |

Benchmark results (2026-04-02) showed PaddleOCR with PP-StructureV3 is the only engine with native reading order recovery and layout block classification (14 element types). This is critical for the document editing use case where paragraph grouping and multi-column reconstruction are requirements.

**Trade-off vs Surya:** PaddleOCR is slower per-page (~200ms GPU vs ~150ms) but produces structured layout output that Surya lacks entirely. For the editing pipeline, layout quality outweighs raw OCR speed.

---

### Providers evaluated but not selected for primary

| Provider | Reason |
|----------|--------|
| Tesseract | No table support, poor layout detection. Retained as CPU-only fallback. |
| EasyOCR | No layout or table support. Suitable for line-level only use cases. |
| Google Vision | Strong accuracy but 5-level nesting + symbol-level output raises adapter cost without benefit over Textract. |
| ABBYY | Character-level XML, async polling, no Python SDK — highest adapter cost. Revisit in phase 2 if degraded-scan accuracy is critical. |
| GOT-OCR 2.0 | No bounding boxes, no confidence scores — cannot normalise to canonical model without major data loss. |
| TrOCR | Requires external line detector; not a full pipeline. |

---

## Design

### Canonical document model

```python
from dataclasses import dataclass, field

@dataclass
class BoundingBox:
    x1: float      # pixels, top-left origin
    y1: float
    x2: float      # pixels, bottom-right
    y2: float
    page: int = 0

@dataclass
class OcrWord:
    text: str
    confidence: float          # 0.0–1.0; -1.0 if unavailable
    bbox: BoundingBox

@dataclass
class OcrLine:
    text: str
    words: list[OcrWord]
    bbox: BoundingBox

@dataclass
class OcrCell:
    row: int
    col: int
    row_span: int
    col_span: int
    text: str
    bbox: BoundingBox

@dataclass
class OcrTable:
    rows: int
    cols: int
    cells: list[OcrCell]

@dataclass
class OcrBlock:
    block_type: str            # "text" | "table" | "figure" | "header" | "footer"
    lines: list[OcrLine]
    bbox: BoundingBox
    table: OcrTable | None = None

@dataclass
class OcrPage:
    page_number: int
    width: float               # pixels
    height: float
    blocks: list[OcrBlock]

    @property
    def full_text(self) -> str:
        return "\n".join(
            line.text
            for block in self.blocks
            for line in block.lines
        )

@dataclass
class OcrResult:
    provider: str              # "surya" | "textract" | "doctr"
    pages: list[OcrPage]
    raw: dict | None = None    # original provider output preserved for debugging
    latency_ms: float = 0.0
```

### Design decisions in the canonical model

- **Pixel coordinates throughout.** Normalised coords (Textract, docTR) are denormalised inside the adapter, not by consumers.
- **`raw` field preserved.** Allows debugging and future normalisation fixes without re-running inference.
- **`confidence = -1.0` convention.** Signals unavailable rather than zero (which would mean detected-but-uncertain).
- **`OcrTable` is optional on `OcrBlock`.** Not all providers return structured cells; the field is `None` rather than an empty list to distinguish "no table" from "table with no cells".

---

### Adapter interface

```python
from abc import ABC, abstractmethod
from pathlib import Path

class OcrAdapter(ABC):

    @abstractmethod
    def process(
        self,
        source: Path | bytes,
        languages: list[str] | None = None,
        extract_tables: bool = True,
    ) -> OcrResult:
        """Process a document and return a canonical OcrResult."""
        ...

    @abstractmethod
    def health_check(self) -> bool:
        """Verify the provider is reachable and operational."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @property
    @abstractmethod
    def supports_tables(self) -> bool: ...

    @property
    @abstractmethod
    def supports_offline(self) -> bool: ...
```

### Adapter registry

```python
from enum import StrEnum

class OcrProvider(StrEnum):
    SURYA    = "surya"
    TEXTRACT = "textract"
    DOCTR    = "doctr"

_registry: dict[OcrProvider, type[OcrAdapter]] = {}

def register(provider: OcrProvider):
    def decorator(cls: type[OcrAdapter]):
        _registry[provider] = cls
        return cls
    return decorator

def get_adapter(provider: OcrProvider, **config) -> OcrAdapter:
    cls = _registry.get(provider)
    if cls is None:
        raise ValueError(f"No adapter registered for: {provider}")
    return cls(**config)
```

### Consumer usage

```python
import os

OCR_PROVIDER = OcrProvider(os.getenv("OCR_PROVIDER", "surya"))

adapter = get_adapter(OCR_PROVIDER)
result: OcrResult = adapter.process(Path("invoice.pdf"), extract_tables=True)

for page in result.pages:
    print(page.full_text)
    for block in page.blocks:
        if block.block_type == "table" and block.table:
            process_table(block.table)
```

### Multi-provider evaluation mode

For the experiment phase, run providers in parallel and collect comparison data:

```python
class EvaluationAdapter(OcrAdapter):
    """Runs all registered providers and logs structured comparison."""

    def __init__(self, providers: list[OcrProvider]):
        self.adapters = [get_adapter(p) for p in providers]

    def process(self, source, **kwargs) -> OcrResult:
        import time
        import concurrent.futures

        results: dict[str, OcrResult] = {}

        with concurrent.futures.ThreadPoolExecutor() as pool:
            futures = {
                pool.submit(a.process, source, **kwargs): a
                for a in self.adapters
            }
            for future, adapter in futures.items():
                t0 = time.monotonic()
                results[adapter.provider_name] = future.result()
                results[adapter.provider_name].latency_ms = (
                    (time.monotonic() - t0) * 1000
                )

        self._log_comparison(results)
        return results[self.adapters[0].provider_name]

    def _log_comparison(self, results: dict[str, OcrResult]):
        # Emit structured metric event for analysis
        ...
```

---

## Directory structure

```
ocr/
├── __init__.py
├── models.py            # BoundingBox, OcrWord, OcrLine, OcrBlock, OcrPage, OcrResult
├── base.py              # OcrAdapter ABC + registry + get_adapter()
├── adapters/
│   ├── __init__.py
│   ├── surya.py
│   ├── textract.py
│   └── doctr.py
├── evaluation.py        # EvaluationAdapter
└── tests/
    ├── fixtures/        # sample documents per type (invoice, form, scan, mixed)
    ├── test_surya.py
    ├── test_textract.py
    ├── test_doctr.py
    └── test_canonical_model.py
```

---

## Experiment plan

> **Status (2026-04-02):** Phases 1-3 complete. Adapters implemented for all four self-hosted engines plus gateway. Benchmarks run against 10 fixture types. PaddleOCR selected as primary. See [tech-overview.md](./tech-overview.md) for results.

### Phase 1 — adapter scaffolding (week 1–2) DONE

- ~~Implement canonical model and base adapter~~ Done — `shared/ocr_schema/models.py`
- ~~Implement Surya adapter~~ Done — `services/surya/src/adapter.py`
- ~~Implement docTR adapter~~ Done — `services/doctr/src/adapter.py`
- ~~Implement Tesseract adapter (Python rewrite)~~ Done — `services/tesseract/src/adapter.py`
- ~~Implement PaddleOCR adapter~~ Done — `services/paddle/src/core/adapter.py`
- Validate Surya GPL-3.0 licence compatibility — **still open**

### Phase 2 — accuracy benchmarking (week 3–4) DONE

- ~~Deploy all 4 engines on g4dn.2xlarge~~ Done (2026-03-31, 2026-04-02)
- ~~Benchmark: latency, word count, confidence, CER/WER~~ Done — 10 fixtures, 3 runs each
- ~~Resource consumption profiling~~ Done — CPU/GPU/memory per service
- ~~Saturation testing~~ Done — concurrency sweep 1-16
- ~~Text similarity matrix~~ Done — pairwise SequenceMatcher
- ~~CER/WER via majority-vote pseudo-ground-truth~~ Done (2026-04-02)
- Results published at GitHub Pages

### Phase 3 — decision (week 5) DONE

- Primary: **PaddleOCR** (layout + reading order + table HTML)
- CPU fallback: **Tesseract** (lightest footprint, no GPU)
- Throughput: **docTR** (30K pages/hr on T4, best batch throughput)
- Multilingual: **Surya** (90+ languages, but GPL-3.0 + OOM risk)

### Phase 4 — editing pipeline integration (planned)

- See [README.md](./README.md) for the full pipeline plan and implementation phases

---

## Open questions

| # | Question | Owner | Status |
|---|----------|-------|--------|
| 1 | Is GPL-3.0 (Surya) acceptable for our distribution model? | Legal / Arch | **Open.** Mitigated — PaddleOCR (Apache-2.0) is now primary. Surya only needed for 90+ language edge case. |
| 2 | Do we send documents to AWS for any customer tier? | Security | **Open.** Textract deferred — self-hosted covers all current needs. |
| 3 | What document types are the primary use case? | Product | **Answered.** Scanned PDFs (all 7 test PDFs were pure image scans, zero embedded text layers). |
| 4 | Is GPU available in self-hosted deployment targets? | Infra | **Answered.** g4dn.xlarge (T4) benchmarked. GPU gives 5-10x speedup for PaddleOCR/docTR/Surya. |
| 5 | Should the adapter interface be sync or async-first? | Arch | **Answered.** Sync. Textract dropped from initial scope; all self-hosted engines are synchronous. |

---

## Decision log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-30 | Start with Surya + Textract + docTR | Best spread of self-hosted vs cloud, clean adapter complexity range |
| 2026-03-30 | Canonical model uses pixel coordinates | Lowest common denominator; normalisation handled per adapter |
| 2026-03-30 | `OcrResult.raw` preserved | Debugging and future normalisation without re-running inference |
| 2026-03-30 | Provider selection via env var | Zero code change to switch providers |
| 2026-04-02 | Add PaddleOCR as primary self-hosted pick | Benchmark showed PP-StructureV3 is the only engine with native reading order + layout classification. Critical for document editing pipeline. See [tech-overview.md](./tech-overview.md). |
| 2026-04-02 | Demote Surya to secondary | GPL-3.0 risk unresolved. Surya OOMs on multi-page PDFs under GPU contention. No native layout/reading order. |
| 2026-04-02 | Retain Tesseract as CPU fallback | Lightest resource footprint (2 GB, no GPU). Best for simple single-column docs. |
