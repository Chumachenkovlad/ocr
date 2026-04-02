# OCR Benchmark Report

- **Date**: 2026-04-02_102446
- **DPI**: 300
- **Runs per fixture**: 3
- **Providers**: doctr, paddle, surya, tesseract
- **Fixtures**: 10

## Latency (ms)

| Fixture | doctr | paddle | surya | tesseract |
| --- | --- | --- | --- | --- |
| dense-twocol.png | 2204 | 3707 | 2475 | 2445 |
| form-kvpairs.png | 2481 | 3399 | 3397 | 1820 |
| low-dpi.png | 4135 | 1466 | 1942 | 519 |
| mixed-content.png | 1746 | 3651 | 3168 | 1676 |
| multilingual.png | 2329 | 3172 | 2644 | 1806 |
| multipage.pdf | 7370 | 9206 | ERR | 5181 |
| noisy-scan.png | 3714 | 12062 | 4332 | 9425 |
| simple-typed.png | 3451 | 3470 | 2540 | 2263 |
| table-complex.png | 1358 | 3272 | ERR | 1558 |
| table-simple.png | 1993 | 3072 | 2756 | 1751 |

### Aggregate latency

| Provider | p50 | p95 | mean |
| --- | --- | --- | --- |
| doctr | 2667 | 7370 | 3175 |
| paddle | 3582 | 12062 | 4743 |
| surya | 2946 | 4332 | 3063 |
| tesseract | 2057 | 9425 | 2922 |

## Word / line / block counts

| Fixture | doctr (W/L/B) | paddle (W/L/B) | surya (W/L/B) | tesseract (W/L/B) |
| --- | --- | --- | --- | --- |
| dense-twocol.png | 66 / 14 / 1 | 300 / 17 / 1 | 303 / 17 / 2 | 322 / 17 / 3 |
| form-kvpairs.png | 78 / 60 / 1 | 128 / 66 / 1 | 134 / 66 / 7 | 131 / 28 / 5 |
| low-dpi.png | 243 / 19 / 1 | 178 / 13 / 1 | 222 / 12 / 1 | 25 / 9 / 4 |
| mixed-content.png | 39 / 9 / 1 | 75 / 36 / 1 | 115 / 36 / 10 | 76 / 18 / 16 |
| multilingual.png | 82 / 18 / 1 | 85 / 18 / 1 | 127 / 18 / 6 | 132 / 18 / 12 |
| multipage.pdf | 355 / 89 / 3 | 354 / 89 / 3 | ERR | 355 / 40 / 15 |
| noisy-scan.png | 90 / 14 / 1 | 101 / 28 / 1 | 190 / 34 / 18 | 132 / 14 / 12 |
| simple-typed.png | 164 / 10 / 1 | 144 / 10 / 1 | 206 / 10 / 7 | 254 / 10 / 6 |
| table-complex.png | 14 / 8 / 1 | 100 / 88 / 1 | ERR | 86 / 13 / 5 |
| table-simple.png | 58 / 42 / 1 | 83 / 43 / 1 | 83 / 43 / 11 | 80 / 11 / 4 |

## Confidence distribution

| Fixture | doctr | paddle | surya | tesseract |
| --- | --- | --- | --- | --- |
| dense-twocol.png | 0.588 ±0.249 [0.22–0.99] | 0.943 ±0.022 [0.93–0.96] | 0.958 ±0.066 [0.95–0.99] | 0.925 ±0.073 [0.88–0.96] |
| form-kvpairs.png | 0.807 ±0.190 [0.54–1.00] | 0.958 ±0.060 [0.92–1.00] | 0.858 ±0.115 [0.70–0.99] | 0.919 ±0.065 [0.87–0.96] |
| low-dpi.png | 0.871 ±0.170 [0.61–1.00] | 0.904 ±0.032 [0.86–0.94] | 0.925 ±0.066 [0.84–0.98] | 0.592 ±0.289 [0.19–0.92] |
| mixed-content.png | 0.802 ±0.184 [0.55–1.00] | 0.965 ±0.045 [0.94–0.99] | 0.803 ±0.095 [0.68–0.91] | 0.891 ±0.113 [0.71–0.96] |
| multilingual.png | 0.619 ±0.226 [0.33–0.97] | 0.854 ±0.085 [0.75–0.92] | 0.827 ±0.100 [0.74–0.91] | 0.715 ±0.327 [0.15–0.96] |
| multipage.pdf | 0.922 ±0.124 [0.71–1.00] | 0.980 ±0.014 [0.96–1.00] | ERR | 0.944 ±0.038 [0.91–0.96] |
| noisy-scan.png | 0.803 ±0.193 [0.54–1.00] | 0.876 ±0.192 [0.57–0.96] | 0.754 ±0.244 [0.33–0.96] | 0.934 ±0.095 [0.92–0.96] |
| simple-typed.png | 0.824 ±0.183 [0.54–0.99] | 0.956 ±0.054 [0.95–0.98] | 0.841 ±0.084 [0.72–0.93] | 0.940 ±0.041 [0.91–0.96] |
| table-complex.png | 0.782 ±0.225 [0.41–1.00] | 0.994 ±0.013 [0.98–1.00] | ERR | 0.907 ±0.083 [0.83–0.96] |
| table-simple.png | 0.825 ±0.178 [0.53–1.00] | 0.973 ±0.024 [0.94–1.00] | 0.841 ±0.146 [0.63–1.00] | 0.909 ±0.114 [0.84–0.96] |

*Format: mean ±std [p10–p90]*

### Aggregate confidence

| Provider | mean | p10 | p50 | p90 |
| --- | --- | --- | --- | --- |
| doctr | 0.784 | 0.619 | 0.807 | 0.922 |
| paddle | 0.940 | 0.876 | 0.958 | 0.994 |
| surya | 0.851 | 0.754 | 0.841 | 0.958 |
| tesseract | 0.868 | 0.715 | 0.919 | 0.944 |

## Text accuracy (vs. majority-vote reference)

*CER = Character Error Rate, WER = Word Error Rate. Lower is better.*
*Reference text is the provider output most similar to all others (centroid).*

### Character Error Rate (CER)

| Fixture | doctr | paddle | surya | tesseract |
| --- | --- | --- | --- | --- |
| dense-twocol.png | 0.604 | 0.129 | 0.000 | 0.523 |
| form-kvpairs.png | 0.247 | 0.000 | 0.252 | 0.091 |
| low-dpi.png | 0.226 | 0.268 | 0.000 | 0.901 |
| mixed-content.png | 0.719 | 0.326 | 0.000 | 0.674 |
| multilingual.png | 0.363 | 0.492 | 0.634 | 0.000 |
| multipage.pdf | 0.000 | 0.005 | — | 0.022 |
| noisy-scan.png | 0.131 | 0.107 | 0.308 | 0.000 |
| simple-typed.png | 0.125 | 0.369 | 0.315 | 0.000 |
| table-complex.png | 0.752 | 0.463 | — | 0.000 |
| table-simple.png | 0.102 | 0.000 | 0.222 | 0.107 |

### Word Error Rate (WER)

| Fixture | doctr | paddle | surya | tesseract |
| --- | --- | --- | --- | --- |
| dense-twocol.png | 0.970 | 0.228 | 0.000 | 0.630 |
| form-kvpairs.png | 0.727 | 0.000 | 0.359 | 0.211 |
| low-dpi.png | 0.531 | 0.540 | 0.000 | 0.951 |
| mixed-content.png | 0.939 | 0.548 | 0.000 | 0.791 |
| multilingual.png | 0.879 | 0.720 | 0.674 | 0.000 |
| multipage.pdf | 0.000 | 0.045 | — | 0.009 |
| noisy-scan.png | 0.667 | 0.538 | 0.470 | 0.000 |
| simple-typed.png | 0.787 | 0.457 | 0.512 | 0.000 |
| table-complex.png | 0.977 | 0.279 | — | 0.000 |
| table-simple.png | 0.554 | 0.000 | 0.253 | 0.133 |

### Aggregate accuracy

| Provider | Mean CER | Mean WER |
| --- | --- | --- |
| doctr | 0.327 | 0.703 |
| paddle | 0.216 | 0.335 |
| surya | 0.216 | 0.284 |
| tesseract | 0.232 | 0.272 |

## Text similarity (SequenceMatcher, 0.0–1.0)

| Fixture | doctr_vs_paddle | doctr_vs_surya | doctr_vs_tesseract | paddle_vs_surya | paddle_vs_tesseract | surya_vs_tesseract |
| --- | --- | --- | --- | --- | --- | --- |
| dense-twocol.png | 0.191 | 0.199 | 0.120 | 0.693 | 0.507 | 0.547 |
| form-kvpairs.png | 0.827 | 0.672 | 0.694 | 0.801 | 0.843 | 0.734 |
| low-dpi.png | 0.536 | 0.555 | 0.144 | 0.537 | 0.145 | 0.166 |
| mixed-content.png | 0.052 | 0.038 | 0.069 | 0.623 | 0.189 | 0.274 |
| multilingual.png | 0.308 | 0.227 | 0.487 | 0.313 | 0.370 | 0.293 |
| multipage.pdf | 0.997 | — | 0.975 | — | 0.969 | — |
| noisy-scan.png | 0.477 | 0.623 | 0.592 | 0.780 | 0.847 | 0.866 |
| simple-typed.png | 0.400 | 0.248 | 0.580 | 0.537 | 0.708 | 0.551 |
| table-complex.png | 0.366 | — | 0.169 | — | 0.341 | — |
| table-simple.png | 0.679 | 0.519 | 0.436 | 0.760 | 0.658 | 0.672 |

### Mean similarity

- **doctr_vs_paddle**: 0.483
- **doctr_vs_surya**: 0.385
- **doctr_vs_tesseract**: 0.427
- **paddle_vs_surya**: 0.630
- **paddle_vs_tesseract**: 0.558
- **surya_vs_tesseract**: 0.513

## Run-to-run variance (multi-run mode)

| Fixture | doctr | paddle | surya | tesseract |
| --- | --- | --- | --- | --- |
| dense-twocol.png | σ=304ms CV=13.12% | σ=391ms CV=10.11% | σ=288ms CV=11.25% | σ=211ms CV=8.36% |
| form-kvpairs.png | σ=187ms CV=7.46% | σ=318ms CV=9.15% | σ=313ms CV=8.91% | σ=267ms CV=14.02% |
| low-dpi.png | σ=399ms CV=9.26% | σ=258ms CV=17.50% | σ=239ms CV=11.84% | σ=192ms CV=34.16% |
| mixed-content.png | σ=261ms CV=14.33% | σ=166ms CV=4.67% | σ=281ms CV=8.62% | σ=253ms CV=14.36% |
| multilingual.png | σ=282ms CV=11.67% | σ=316ms CV=9.63% | σ=300ms CV=10.86% | σ=219ms CV=11.35% |
| multipage.pdf | σ=121ms CV=1.63% | σ=141ms CV=1.52% | ERR | σ=172ms CV=3.29% |
| noisy-scan.png | σ=546ms CV=14.28% | σ=1663ms CV=13.36% | σ=1182ms CV=24.26% | σ=144ms CV=1.51% |
| simple-typed.png | σ=288ms CV=8.04% | σ=309ms CV=8.79% | σ=255ms CV=9.62% | σ=234ms CV=9.84% |
| table-complex.png | σ=302ms CV=20.68% | σ=218ms CV=6.52% | ERR | σ=161ms CV=9.83% |
| table-simple.png | σ=269ms CV=12.90% | σ=298ms CV=9.40% | σ=272ms CV=9.50% | σ=254ms CV=14.22% |

## Per document-type analysis

| Type | Fixtures | Speed winner | Throughput winner | Confidence winner |
| --- | --- | --- | --- | --- |
| dense-layout | `dense-twocol.png` | doctr | tesseract | surya |
| form | `form-kvpairs.png` | tesseract | surya | paddle |
| low-dpi | `low-dpi.png` | tesseract | doctr | surya |
| mixed-content | `mixed-content.png` | tesseract | surya | paddle |
| multipage | `multilingual.png`, `multipage.pdf` | surya | tesseract | paddle |
| noisy-scan | `noisy-scan.png` | doctr | surya | tesseract |
| simple-typed | `simple-typed.png` | tesseract | tesseract | paddle |
| table | `table-complex.png`, `table-simple.png` | tesseract | paddle | paddle |

## Conclusion

### Observed rankings

| Metric | Ranking |
| --- | --- |
| Speed | tesseract (2922ms) > surya (3063ms) > doctr (3175ms) > paddle (4743ms) |
| Words extracted | surya (172) > tesseract (159) > paddle (155) > doctr (119) |
| Avg confidence | paddle (0.940) > tesseract (0.868) > surya (0.851) > doctr (0.784) |

### Text agreement between providers

- **Most agreeing pair**: paddle_vs_surya (0.630)
- **Least agreeing pair**: doctr_vs_surya (0.385)

### Use-case recommendations

| Use case | Recommended | Reason |
| --- | --- | --- |
| Real-time / latency-sensitive | **tesseract** | lowest mean latency (2922ms) |
| Maximum text extraction (recall) | **surya** | most words on average (172) |
| Highest confidence / precision | **paddle** | best avg confidence (0.940) |
| Table extraction | **paddle** | PaddleOCR has native table HTML output |
| Multi-language documents (90+ langs) | **surya** | Surya supports 90+ languages |
| Air-gap / CPU-only / no GPU infra | **tesseract** | pure CPU, Apache 2.0, zero model downloads at runtime |
| Word-level bounding boxes | **doctr** | docTR provides per-word bbox + confidence |

> Rankings derived from benchmark data. For document types not covered by fixtures, validate with representative samples before choosing a provider.

## Errors

- **surya** on `multipage.pdf`: Server error '500 Internal Server Error' for url 'http://98.80.151.147:8083/api/v1/ocr'
For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/500
- **surya** on `multipage.pdf`: Server error '500 Internal Server Error' for url 'http://98.80.151.147:8083/api/v1/ocr'
For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/500
- **surya** on `multipage.pdf`: Server error '500 Internal Server Error' for url 'http://98.80.151.147:8083/api/v1/ocr'
For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/500
- **surya** on `table-complex.png`: Server error '500 Internal Server Error' for url 'http://98.80.151.147:8083/api/v1/ocr'
For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/500
- **surya** on `table-complex.png`: Server error '500 Internal Server Error' for url 'http://98.80.151.147:8083/api/v1/ocr'
For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/500
- **surya** on `table-complex.png`: Server error '500 Internal Server Error' for url 'http://98.80.151.147:8083/api/v1/ocr'
For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/500
