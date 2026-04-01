# OCR Benchmark Report

- **Date**: 2026-03-31_071432
- **DPI**: 300
- **Runs per fixture**: 1
- **Providers**: doctr, paddle, surya, tesseract
- **Fixtures**: 10

## Latency (ms)

| Fixture | doctr | paddle | surya | tesseract |
| --- | --- | --- | --- | --- |
| dense-twocol.png | 2738 | 21611 | 24273 | 3176 |
| form-kvpairs.png | 4050 | 29700 | 65305 | 2364 |
| low-dpi.png | 5286 | 9046 | 16232 | 777 |
| mixed-content.png | 3162 | 25294 | 40286 | 2181 |
| multilingual.png | 2807 | 26849 | 25913 | 2319 |
| multipage.pdf | 57493 | 77207 | 104256 | 5735 |
| noisy-scan.png | 5595 | 35749 | 50114 | 11032 |
| simple-typed.png | 4173 | 23701 | 21385 | 12095 |
| table-complex.png | 2256 | 31404 | 89214 | 3619 |
| table-simple.png | 4339 | 29734 | 46483 | 2062 |

### Aggregate Latency

| Provider | p50 | p95 | mean |
| --- | --- | --- | --- |
| doctr | 4173 | 57493 | 9190 |
| paddle | 29700 | 77207 | 31029 |
| surya | 46483 | 104256 | 48346 |
| tesseract | 3176 | 12095 | 4536 |

## Word Count

| Fixture | doctr | paddle | surya | tesseract |
| --- | --- | --- | --- | --- |
| dense-twocol.png | 66 | 0 | 303 | 322 |
| form-kvpairs.png | 78 | 130 | 134 | 131 |
| low-dpi.png | 243 | 24 | 222 | 25 |
| mixed-content.png | 39 | 57 | 115 | 76 |
| multilingual.png | 82 | 79 | 127 | 132 |
| multipage.pdf | 355 | 238 | 353 | 355 |
| noisy-scan.png | 90 | 65 | 228 | 132 |
| simple-typed.png | 164 | 21 | 206 | 254 |
| table-complex.png | 14 | 95 | 99 | 86 |
| table-simple.png | 58 | 48 | 83 | 80 |

## Average Confidence

| Fixture | doctr | paddle | surya | tesseract |
| --- | --- | --- | --- | --- |
| dense-twocol.png | 0.588 | 0.000 | 0.958 | 0.925 |
| form-kvpairs.png | 0.807 | 0.971 | 0.929 | 0.919 |
| low-dpi.png | 0.871 | 0.840 | 0.925 | 0.592 |
| mixed-content.png | 0.802 | 0.973 | 0.830 | 0.891 |
| multilingual.png | 0.619 | 0.809 | 0.827 | 0.715 |
| multipage.pdf | 0.922 | 0.943 | 0.928 | 0.944 |
| noisy-scan.png | 0.803 | 0.868 | 0.711 | 0.934 |
| simple-typed.png | 0.824 | 0.824 | 0.833 | 0.940 |
| table-complex.png | 0.782 | 0.990 | 0.779 | 0.907 |
| table-simple.png | 0.825 | 0.981 | 0.909 | 0.909 |
