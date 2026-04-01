"""Shared test fixtures for the Tesseract OCR service."""

from __future__ import annotations

import pytest

from src.config import Settings


@pytest.fixture()
def settings() -> Settings:
    """Default test settings."""
    return Settings(
        tesseract_binary="tesseract",
        tesseract_lang="eng",
        tesseract_dpi=300,
        tesseract_timeout=30,
        max_file_size_mb=50,
        max_pages=100,
        log_level="DEBUG",
    )


SAMPLE_HOCR = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
    "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
<head>
    <title></title>
    <meta http-equiv="Content-Type" content="text/html;charset=utf-8" />
    <meta name='ocr-system' content='tesseract 5.3.0' />
    <meta name='ocr-capabilities' content='ocr_page ocr_carea ocr_par ocr_line ocrx_word'/>
</head>
<body>
    <div class='ocr_page' id='page_1' title='image "test.png"; bbox 0 0 2550 3300; ppageno 0'>
        <div class='ocr_carea' id='block_1_1' title="bbox 100 100 2450 500">
            <p class='ocr_par' id='par_1_1' lang='eng' title="bbox 100 100 2450 500">
                <span class='ocr_line' id='line_1_1' title="bbox 100 100 800 140; baseline 0 -5">
                    <span class='ocrx_word' id='word_1_1' title='bbox 100 100 250 140; x_wconf 95'>Hello</span>
                    <span class='ocrx_word' id='word_1_2' title='bbox 270 100 450 140; x_wconf 92'>World</span>
                </span>
                <span class='ocr_line' id='line_1_2' title="bbox 100 160 600 200; baseline 0 -3">
                    <span class='ocrx_word' id='word_1_3' title='bbox 100 160 300 200; x_wconf 88'>Second</span>
                    <span class='ocrx_word' id='word_1_4' title='bbox 320 160 450 200; x_wconf 90'>line</span>
                </span>
            </p>
        </div>
        <div class='ocr_carea' id='block_1_2' title="bbox 100 550 2450 800">
            <p class='ocr_par' id='par_1_2' lang='eng' title="bbox 100 550 2450 800">
                <span class='ocr_line' id='line_1_3' title="bbox 100 550 500 590; baseline 0 -4">
                    <span class='ocrx_word' id='word_1_5' title='bbox 100 550 350 590; x_wconf 97'>Another</span>
                    <span class='ocrx_word' id='word_1_6' title='bbox 370 550 500 590; x_wconf 85'>block</span>
                </span>
            </p>
        </div>
    </div>
</body>
</html>
"""

SAMPLE_HOCR_EMPTY_PAGE = """\
<?xml version="1.0" encoding="UTF-8"?>
<html>
<head><title></title></head>
<body>
    <div class='ocr_page' id='page_1' title='image "empty.png"; bbox 0 0 800 600; ppageno 0'>
    </div>
</body>
</html>
"""

SAMPLE_HOCR_NO_CAREA = """\
<?xml version="1.0" encoding="UTF-8"?>
<html>
<head><title></title></head>
<body>
    <div class='ocr_page' id='page_1' title='image "test.png"; bbox 0 0 1000 800; ppageno 0'>
        <p class='ocr_par' id='par_1_1' lang='eng' title="bbox 50 50 900 200">
            <span class='ocr_line' id='line_1_1' title="bbox 50 50 400 90; baseline 0 -3">
                <span class='ocrx_word' id='word_1_1' title='bbox 50 50 200 90; x_wconf 91'>Direct</span>
                <span class='ocrx_word' id='word_1_2' title='bbox 220 50 400 90; x_wconf 89'>paragraph</span>
            </span>
        </p>
    </div>
</body>
</html>
"""

SAMPLE_HOCR_MISSING_CONF = """\
<?xml version="1.0" encoding="UTF-8"?>
<html>
<head><title></title></head>
<body>
    <div class='ocr_page' id='page_1' title='image "test.png"; bbox 0 0 500 400; ppageno 0'>
        <div class='ocr_carea' id='block_1_1' title="bbox 10 10 490 390">
            <p class='ocr_par' id='par_1_1' lang='eng' title="bbox 10 10 490 390">
                <span class='ocr_line' id='line_1_1' title="bbox 10 10 200 50">
                    <span class='ocrx_word' id='word_1_1' title='bbox 10 10 100 50'>NoConf</span>
                </span>
            </p>
        </div>
    </div>
</body>
</html>
"""


@pytest.fixture()
def sample_hocr() -> str:
    return SAMPLE_HOCR


@pytest.fixture()
def sample_hocr_empty_page() -> str:
    return SAMPLE_HOCR_EMPTY_PAGE


@pytest.fixture()
def sample_hocr_no_carea() -> str:
    return SAMPLE_HOCR_NO_CAREA


@pytest.fixture()
def sample_hocr_missing_conf() -> str:
    return SAMPLE_HOCR_MISSING_CONF
