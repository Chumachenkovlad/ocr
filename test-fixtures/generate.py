"""Generate synthetic OCR test fixtures.

Run: python3 test-fixtures/generate.py
Produces PNG and PDF files in test-fixtures/ for benchmarking OCR services.
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.units import mm, inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Frame,
    PageTemplate,
    BaseDocTemplate,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY

OUTPUT_DIR = Path(__file__).parent

# Reproducible
random.seed(42)

LOREM = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
    "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris "
    "nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in "
    "reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla "
    "pariatur. Excepteur sint occaecat cupidatat non proident, sunt in "
    "culpa qui officia deserunt mollit anim id est laborum."
)

PARAGRAPHS = [
    "The quarterly revenue report shows a 15% increase compared to the previous period. "
    "This growth was primarily driven by expansion in the European market segment.",
    "Our engineering team completed the migration to the new cloud infrastructure. "
    "System uptime improved from 99.2% to 99.8% during Q3 2025.",
    "Customer satisfaction scores reached an all-time high of 4.7 out of 5.0. "
    "The support team resolved 94% of tickets within 24 hours.",
    "The new product line generated $2.3M in revenue during its first quarter. "
    "Pre-orders for the next release exceeded initial projections by 40%.",
    "Research and development spending increased to 18% of total revenue. "
    "Three patent applications were filed for novel machine learning techniques.",
]

INVOICE_ITEMS = [
    ("Web Development Services", "40", "$150.00", "$6,000.00"),
    ("UI/UX Design", "20", "$125.00", "$2,500.00"),
    ("Database Migration", "15", "$175.00", "$2,625.00"),
    ("API Integration", "10", "$160.00", "$1,600.00"),
    ("Quality Assurance", "25", "$100.00", "$2,500.00"),
    ("Project Management", "12", "$140.00", "$1,680.00"),
    ("Documentation", "8", "$90.00", "$720.00"),
]


def get_font(size: int = 16, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Get a system font that works cross-platform."""
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSText.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    if bold:
        font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ] + font_paths

    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except (OSError, Exception):
                continue
    return ImageFont.load_default()


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Wrap text to fit within max_width pixels."""
    words = text.split()
    lines: list[str] = []
    current_line: list[str] = []

    for word in words:
        test = " ".join(current_line + [word])
        bbox = font.getbbox(test)
        if bbox[2] - bbox[0] <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))
    return lines


# ---------------------------------------------------------------------------
# 1. simple-typed.png — clean single-column text
# ---------------------------------------------------------------------------
def generate_simple_typed():
    w, h = 2480, 3508  # A4 at 300 DPI
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    font_title = get_font(36, bold=True)
    font_body = get_font(20)

    y = 200
    draw.text((200, y), "Quarterly Business Report — Q3 2025", fill="black", font=font_title)
    y += 80

    for para in PARAGRAPHS:
        lines = wrap_text(para, font_body, w - 400)
        for line in lines:
            draw.text((200, y), line, fill="black", font=font_body)
            y += 30
        y += 20

    # Add some more text to fill the page
    lines = wrap_text(LOREM + " " + LOREM, font_body, w - 400)
    for line in lines:
        draw.text((200, y), line, fill="black", font=font_body)
        y += 30

    img.save(OUTPUT_DIR / "simple-typed.png", dpi=(300, 300))
    print("  simple-typed.png")


# ---------------------------------------------------------------------------
# 2. dense-twocol.png — two-column layout
# ---------------------------------------------------------------------------
def generate_dense_twocol():
    w, h = 2480, 3508
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    font_title = get_font(32, bold=True)
    font_body = get_font(16)

    # Title spanning both columns
    draw.text((200, 150), "Technical Bulletin — Infrastructure Update", fill="black", font=font_title)

    # Draw column separator
    col_gap = 60
    col_w = (w - 400 - col_gap) // 2
    mid_x = 200 + col_w + col_gap // 2
    draw.line([(mid_x, 230), (mid_x, h - 200)], fill="#CCCCCC", width=1)

    # Left column
    left_text = " ".join(PARAGRAPHS[:3]) + " " + LOREM
    lines_left = wrap_text(left_text, font_body, col_w)
    y = 250
    for line in lines_left:
        if y > h - 200:
            break
        draw.text((200, y), line, fill="black", font=font_body)
        y += 24

    # Right column
    right_text = " ".join(PARAGRAPHS[3:]) + " " + LOREM + " " + LOREM
    lines_right = wrap_text(right_text, font_body, col_w)
    y = 250
    for line in lines_right:
        if y > h - 200:
            break
        draw.text((mid_x + col_gap // 2, y), line, fill="black", font=font_body)
        y += 24

    img.save(OUTPUT_DIR / "dense-twocol.png", dpi=(300, 300))
    print("  dense-twocol.png")


# ---------------------------------------------------------------------------
# 3. table-simple.png — document with a bordered table
# ---------------------------------------------------------------------------
def generate_table_simple():
    w, h = 2480, 3508
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    font_title = get_font(28, bold=True)
    font_header = get_font(18, bold=True)
    font_body = get_font(16)

    draw.text((200, 150), "Monthly Sales Summary — March 2025", fill="black", font=font_title)

    # Intro paragraph
    intro = "The following table summarizes sales performance across all regions for the month of March 2025."
    for i, line in enumerate(wrap_text(intro, font_body, w - 400)):
        draw.text((200, 230 + i * 24), line, fill="black", font=font_body)

    # Table
    headers = ["Region", "Units Sold", "Revenue", "Growth %", "Target Met"]
    rows = [
        ["North America", "12,450", "$1,245,000", "+8.3%", "Yes"],
        ["Europe", "9,870", "$987,000", "+12.1%", "Yes"],
        ["Asia Pacific", "15,200", "$1,140,000", "+5.7%", "No"],
        ["Latin America", "4,320", "$345,600", "+18.4%", "Yes"],
        ["Middle East", "2,100", "$210,000", "-2.1%", "No"],
        ["Africa", "1,850", "$148,000", "+22.5%", "Yes"],
    ]

    table_x = 200
    table_y = 340
    col_widths = [400, 300, 350, 250, 250]
    row_height = 45

    # Header row
    x = table_x
    for ci, header in enumerate(headers):
        draw.rectangle([x, table_y, x + col_widths[ci], table_y + row_height], fill="#2C3E50")
        draw.text((x + 15, table_y + 12), header, fill="white", font=font_header)
        x += col_widths[ci]

    # Data rows
    for ri, row in enumerate(rows):
        y = table_y + (ri + 1) * row_height
        bg = "#F8F9FA" if ri % 2 == 0 else "white"
        x = table_x
        for ci, cell in enumerate(row):
            draw.rectangle([x, y, x + col_widths[ci], y + row_height], fill=bg, outline="#DEE2E6")
            draw.text((x + 15, y + 12), cell, fill="black", font=font_body)
            x += col_widths[ci]

    # Total row
    total_y = table_y + (len(rows) + 1) * row_height
    x = table_x
    totals = ["Total", "45,790", "$4,075,600", "+10.8%", "4/6"]
    for ci, cell in enumerate(totals):
        draw.rectangle([x, total_y, x + col_widths[ci], total_y + row_height], fill="#ECF0F1", outline="#DEE2E6")
        draw.text((x + 15, total_y + 12), cell, fill="black", font=font_header)
        x += col_widths[ci]

    # Outer border
    total_w = sum(col_widths)
    total_h = (len(rows) + 2) * row_height
    draw.rectangle([table_x, table_y, table_x + total_w, table_y + total_h], outline="#2C3E50", width=2)

    # Footer text
    footer_y = table_y + total_h + 40
    footer = "Note: Growth percentages are calculated relative to the same period in the previous fiscal year."
    draw.text((200, footer_y), footer, fill="#666666", font=font_body)

    img.save(OUTPUT_DIR / "table-simple.png", dpi=(300, 300))
    print("  table-simple.png")


# ---------------------------------------------------------------------------
# 4. table-complex.png — merged cells, borderless regions
# ---------------------------------------------------------------------------
def generate_table_complex():
    w, h = 2480, 3508
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    font_title = get_font(28, bold=True)
    font_header = get_font(16, bold=True)
    font_body = get_font(14)
    font_small = get_font(12)

    draw.text((200, 120), "Project Resource Allocation Matrix", fill="black", font=font_title)

    # Complex table with merged header cells
    table_x, table_y = 150, 200
    rh = 38

    # Draw a complex header with merged cells
    # Row 1: Department | Q1 (spans 3) | Q2 (spans 3)
    draw.rectangle([table_x, table_y, table_x + 350, table_y + rh * 2], fill="#34495E", outline="#2C3E50")
    draw.text((table_x + 15, table_y + 25), "Department", fill="white", font=font_header)

    q_col_w = 180
    for qi, qlabel in enumerate(["Q1 2025", "Q2 2025"]):
        qx = table_x + 350 + qi * q_col_w * 3
        draw.rectangle([qx, table_y, qx + q_col_w * 3, table_y + rh], fill="#2C3E50", outline="#34495E")
        draw.text((qx + q_col_w, table_y + 10), qlabel, fill="white", font=font_header)

        for mi, month in enumerate(["Jan", "Feb", "Mar"] if qi == 0 else ["Apr", "May", "Jun"]):
            mx = qx + mi * q_col_w
            draw.rectangle([mx, table_y + rh, mx + q_col_w, table_y + rh * 2], fill="#7F8C8D", outline="#95A5A6")
            draw.text((mx + 15, table_y + rh + 10), month, fill="white", font=font_small)

    # Data rows
    departments = ["Engineering", "Design", "Marketing", "Sales", "Operations", "Support", "Research", "Legal"]
    data_grid = [
        [12, 14, 13, 15, 16, 15],
        [5, 5, 6, 6, 7, 7],
        [8, 9, 10, 11, 10, 12],
        [15, 14, 16, 17, 18, 19],
        [6, 6, 5, 5, 6, 6],
        [10, 11, 12, 13, 12, 14],
        [4, 4, 5, 5, 6, 6],
        [3, 3, 3, 3, 3, 3],
    ]

    for ri, (dept, values) in enumerate(zip(departments, data_grid)):
        y = table_y + rh * 2 + ri * rh
        bg = "#F8F9FA" if ri % 2 == 0 else "white"
        draw.rectangle([table_x, y, table_x + 350, y + rh], fill=bg, outline="#DEE2E6")
        draw.text((table_x + 15, y + 10), dept, fill="black", font=font_body)

        for vi, val in enumerate(values):
            vx = table_x + 350 + vi * q_col_w
            draw.rectangle([vx, y, vx + q_col_w, y + rh], fill=bg, outline="#DEE2E6")
            draw.text((vx + 15, y + 10), str(val), fill="black", font=font_body)

    # Borderless summary below
    summary_y = table_y + rh * 2 + len(departments) * rh + 30
    draw.text((150, summary_y), "Summary Statistics (no borders):", fill="black", font=font_header)
    summary_y += 35

    stats = [
        ("Total Headcount", "63", "72", "70", "75", "78", "82"),
        ("Budget Utilization", "87%", "91%", "89%", "92%", "94%", "96%"),
        ("Avg Team Size", "7.9", "8.0", "8.8", "9.4", "9.8", "10.3"),
    ]
    for si, (label, *vals) in enumerate(stats):
        y = summary_y + si * 30
        draw.text((170, y), label, fill="#333333", font=font_body)
        for vi, val in enumerate(vals):
            draw.text((table_x + 350 + vi * q_col_w + 15, y), val, fill="#333333", font=font_body)

    img.save(OUTPUT_DIR / "table-complex.png", dpi=(300, 300))
    print("  table-complex.png")


# ---------------------------------------------------------------------------
# 5. form-kvpairs.png — invoice with key-value fields
# ---------------------------------------------------------------------------
def generate_form_kvpairs():
    w, h = 2480, 3508
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    font_company = get_font(36, bold=True)
    font_label = get_font(16, bold=True)
    font_value = get_font(16)
    font_small = get_font(14)
    font_header = get_font(15, bold=True)

    # Company header
    draw.text((200, 120), "ACME CORPORATION", fill="#2C3E50", font=font_company)
    draw.text((200, 175), "123 Business Avenue, Suite 400", fill="#666666", font=font_small)
    draw.text((200, 200), "San Francisco, CA 94105", fill="#666666", font=font_small)
    draw.text((200, 225), "Phone: (415) 555-0100  |  Email: billing@acme.com", fill="#666666", font=font_small)

    # INVOICE badge
    draw.rectangle([1800, 120, 2100, 175], fill="#E74C3C")
    draw.text((1850, 133), "INVOICE", fill="white", font=get_font(24, bold=True))

    # Key-value pairs
    kv_y = 300
    kvs = [
        ("Invoice Number:", "INV-2025-00847"),
        ("Invoice Date:", "March 15, 2025"),
        ("Due Date:", "April 14, 2025"),
        ("Payment Terms:", "Net 30"),
    ]
    for label, value in kvs:
        draw.text((1500, kv_y), label, fill="#666666", font=font_label)
        draw.text((1800, kv_y), value, fill="black", font=font_value)
        kv_y += 30

    # Bill To / Ship To
    draw.line([(200, 450), (2280, 450)], fill="#CCCCCC", width=1)

    draw.text((200, 480), "BILL TO:", fill="#666666", font=font_label)
    bill_to = ["Widget Industries Inc.", "John Smith, Procurement", "456 Oak Street", "Portland, OR 97201"]
    for i, line in enumerate(bill_to):
        draw.text((200, 510 + i * 25), line, fill="black", font=font_value)

    draw.text((1200, 480), "SHIP TO:", fill="#666666", font=font_label)
    ship_to = ["Widget Industries — Warehouse", "Attn: Receiving Dept", "789 Industrial Blvd", "Portland, OR 97203"]
    for i, line in enumerate(ship_to):
        draw.text((1200, 510 + i * 25), line, fill="black", font=font_value)

    # Items table
    table_y = 680
    headers = ["Description", "Hours", "Rate", "Amount"]
    col_widths = [900, 250, 300, 350]
    col_starts = [200]
    for cw in col_widths[:-1]:
        col_starts.append(col_starts[-1] + cw)

    # Header
    for ci, header in enumerate(headers):
        x = col_starts[ci]
        draw.rectangle([x, table_y, x + col_widths[ci], table_y + 40], fill="#2C3E50")
        draw.text((x + 15, table_y + 10), header, fill="white", font=font_header)

    # Items
    for ri, (desc, hrs, rate, amt) in enumerate(INVOICE_ITEMS):
        y = table_y + 40 + ri * 38
        bg = "#F8F9FA" if ri % 2 == 0 else "white"
        for ci in range(4):
            draw.rectangle([col_starts[ci], y, col_starts[ci] + col_widths[ci], y + 38],
                           fill=bg, outline="#DEE2E6")
        draw.text((col_starts[0] + 15, y + 10), desc, fill="black", font=font_value)
        draw.text((col_starts[1] + 15, y + 10), hrs, fill="black", font=font_value)
        draw.text((col_starts[2] + 15, y + 10), rate, fill="black", font=font_value)
        draw.text((col_starts[3] + 15, y + 10), amt, fill="black", font=font_value)

    # Totals
    totals_y = table_y + 40 + len(INVOICE_ITEMS) * 38 + 20
    totals = [
        ("Subtotal:", "$15,625.00"),
        ("Tax (8.5%):", "$1,328.13"),
        ("Shipping:", "$0.00"),
        ("Total Due:", "$16,953.13"),
    ]
    for label, value in totals:
        draw.text((1500, totals_y), label, fill="#666666", font=font_label)
        draw.text((1850, totals_y), value, fill="black", font=font_label if "Total" in label else font_value)
        totals_y += 30

    # Payment info
    pay_y = totals_y + 60
    draw.line([(200, pay_y), (2280, pay_y)], fill="#CCCCCC", width=1)
    draw.text((200, pay_y + 20), "Payment Instructions:", fill="#666666", font=font_label)
    draw.text((200, pay_y + 50), "Bank: First National Bank  |  Account: 1234-5678-9012  |  Routing: 021000089",
              fill="black", font=font_small)
    draw.text((200, pay_y + 80), "Please include invoice number INV-2025-00847 in the payment reference.",
              fill="black", font=font_small)

    img.save(OUTPUT_DIR / "form-kvpairs.png", dpi=(300, 300))
    print("  form-kvpairs.png")


# ---------------------------------------------------------------------------
# 6. noisy-scan.png — degraded photocopy with noise and skew
# ---------------------------------------------------------------------------
def generate_noisy_scan():
    # Start with a clean version
    w, h = 2480, 3508
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    font = get_font(20)
    font_title = get_font(28, bold=True)

    draw.text((200, 200), "MEMORANDUM", fill="black", font=font_title)
    draw.text((200, 260), "TO: All Department Heads", fill="black", font=font)
    draw.text((200, 290), "FROM: Chief Operating Officer", fill="black", font=font)
    draw.text((200, 320), "DATE: March 10, 2025", fill="black", font=font)
    draw.text((200, 350), "RE: Updated Expense Reimbursement Policy", fill="black", font=font)

    draw.line([(200, 390), (2280, 390)], fill="black", width=2)

    y = 420
    memo_text = [
        "Effective April 1, 2025, the following changes to the expense reimbursement policy will take effect:",
        "",
        "1. All expense reports must be submitted within 14 calendar days of the expense being incurred.",
        "2. Receipts are required for any individual expense exceeding $25.00.",
        "3. Meal expenses during business travel are capped at $75.00 per day.",
        "4. Mileage reimbursement rate has been updated to $0.67 per mile.",
        "5. Pre-approval is required for any single expense exceeding $500.00.",
        "",
        "Please ensure your teams are aware of these changes. The updated policy document is available",
        "on the company intranet under HR > Policies > Travel & Expenses.",
        "",
        "Questions should be directed to the Finance Department at ext. 4200.",
    ]
    for line in memo_text:
        if line:
            for wrapped in wrap_text(line, font, w - 400):
                draw.text((200, y), wrapped, fill="black", font=font)
                y += 30
        else:
            y += 15

    # Degrade the image
    # 1. Add noise
    import numpy as np
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 15, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # 2. Slight rotation (skew)
    img = img.rotate(1.5, expand=False, fillcolor=(240, 235, 228))

    # 3. Reduce contrast + yellowing (old photocopy look)
    arr = np.array(img).astype(np.float32)
    arr = arr * 0.8 + 40  # reduce contrast
    arr[:, :, 0] = np.clip(arr[:, :, 0] + 8, 0, 255)   # slight red
    arr[:, :, 1] = np.clip(arr[:, :, 1] + 5, 0, 255)   # slight green
    arr[:, :, 2] = np.clip(arr[:, :, 2] - 10, 0, 255)   # reduce blue
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # 4. Slight blur
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))

    # 5. Add some dark spots (photocopy artifacts)
    draw = ImageDraw.Draw(img)
    for _ in range(30):
        x = random.randint(0, w)
        y = random.randint(0, h)
        r = random.randint(2, 6)
        gray = random.randint(60, 120)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(gray, gray, gray))

    img.save(OUTPUT_DIR / "noisy-scan.png", dpi=(300, 300))
    print("  noisy-scan.png")


# ---------------------------------------------------------------------------
# 7. mixed-content.png — text + diagram placeholder + table + header/footer
# ---------------------------------------------------------------------------
def generate_mixed_content():
    w, h = 2480, 3508
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    font_header = get_font(14)
    font_title = get_font(30, bold=True)
    font_subtitle = get_font(20, bold=True)
    font_body = get_font(16)
    font_small = get_font(12)
    font_caption = get_font(13)

    # Page header
    draw.rectangle([0, 0, w, 60], fill="#2C3E50")
    draw.text((200, 18), "ACME Corp — System Architecture Document v2.1", fill="white", font=font_header)
    draw.text((1900, 18), "Page 1 of 1", fill="white", font=font_header)

    # Title
    draw.text((200, 100), "System Architecture Overview", fill="black", font=font_title)

    # Body paragraph
    y = 170
    body = (
        "This document outlines the high-level architecture of the ACME platform. "
        "The system consists of three primary layers: the presentation tier, the application tier, "
        "and the data tier. Each tier is designed for horizontal scalability and fault tolerance."
    )
    for line in wrap_text(body, font_body, w - 400):
        draw.text((200, y), line, fill="black", font=font_body)
        y += 24
    y += 20

    # Diagram placeholder (box with shapes inside)
    draw.text((200, y), "Figure 1: High-Level Architecture", fill="black", font=font_subtitle)
    y += 35
    diag_y = y
    diag_h = 400
    draw.rectangle([200, diag_y, w - 200, diag_y + diag_h], outline="#2C3E50", width=2)

    # Draw boxes inside
    boxes = [
        (300, diag_y + 50, 700, diag_y + 120, "#3498DB", "Load Balancer"),
        (300, diag_y + 160, 700, diag_y + 230, "#2ECC71", "API Gateway"),
        (250, diag_y + 280, 580, diag_y + 370, "#E74C3C", "Auth Service"),
        (650, diag_y + 280, 980, diag_y + 370, "#F39C12", "Core Service"),
        (1050, diag_y + 280, 1380, diag_y + 370, "#9B59B6", "Data Service"),
        (1450, diag_y + 160, 2100, diag_y + 230, "#1ABC9C", "Message Queue"),
        (1450, diag_y + 280, 2100, diag_y + 370, "#34495E", "Database Cluster"),
    ]
    for x1, y1, x2, y2, color, label in boxes:
        draw.rectangle([x1, y1, x2, y2], fill=color, outline="white", width=2)
        tw = font_small.getbbox(label)[2] - font_small.getbbox(label)[0]
        tx = x1 + (x2 - x1 - tw) // 2
        ty = y1 + (y2 - y1 - 16) // 2
        draw.text((tx, ty), label, fill="white", font=font_small)

    # Arrows
    draw.line([(500, diag_y + 120), (500, diag_y + 160)], fill="#333", width=2)
    draw.line([(500, diag_y + 230), (415, diag_y + 280)], fill="#333", width=2)
    draw.line([(500, diag_y + 230), (815, diag_y + 280)], fill="#333", width=2)
    draw.line([(500, diag_y + 230), (1215, diag_y + 280)], fill="#333", width=2)

    y = diag_y + diag_h + 10
    draw.text((800, y), "Figure 1: Three-tier architecture with message queue", fill="#666666", font=font_caption)
    y += 40

    # Small table
    draw.text((200, y), "Table 1: Service SLA Targets", fill="black", font=font_subtitle)
    y += 35
    cols = ["Service", "Uptime SLA", "Latency p99", "RPM Limit"]
    col_ws = [400, 300, 300, 300]
    data = [
        ["API Gateway", "99.99%", "< 50ms", "10,000"],
        ["Auth Service", "99.95%", "< 100ms", "5,000"],
        ["Core Service", "99.9%", "< 200ms", "8,000"],
        ["Data Service", "99.9%", "< 150ms", "6,000"],
    ]

    x = 200
    for ci, col in enumerate(cols):
        draw.rectangle([x, y, x + col_ws[ci], y + 35], fill="#2C3E50")
        draw.text((x + 10, y + 8), col, fill="white", font=font_small)
        x += col_ws[ci]

    for ri, row in enumerate(data):
        ry = y + 35 + ri * 30
        x = 200
        bg = "#F8F9FA" if ri % 2 == 0 else "white"
        for ci, cell in enumerate(row):
            draw.rectangle([x, ry, x + col_ws[ci], ry + 30], fill=bg, outline="#DEE2E6")
            draw.text((x + 10, ry + 6), cell, fill="black", font=font_small)
            x += col_ws[ci]

    # Footer
    draw.line([(200, h - 80), (w - 200, h - 80)], fill="#CCCCCC", width=1)
    draw.text((200, h - 65), "Confidential — ACME Corporation 2025", fill="#999999", font=font_small)
    draw.text((1800, h - 65), "Doc ID: ARCH-2025-001", fill="#999999", font=font_small)

    img.save(OUTPUT_DIR / "mixed-content.png", dpi=(300, 300))
    print("  mixed-content.png")


# ---------------------------------------------------------------------------
# 8. multipage.pdf — 3-page PDF mixing content types
# ---------------------------------------------------------------------------
def generate_multipage_pdf():
    path = str(OUTPUT_DIR / "multipage.pdf")
    doc = SimpleDocTemplate(path, pagesize=A4)
    styles = getSampleStyleSheet()

    story = []

    # Page 1: text
    story.append(Paragraph("Annual Performance Review — 2025", styles["Title"]))
    story.append(Spacer(1, 12))
    for para in PARAGRAPHS:
        story.append(Paragraph(para, styles["BodyText"]))
        story.append(Spacer(1, 8))
    story.append(Paragraph(LOREM, styles["BodyText"]))
    story.append(PageBreak())

    # Page 2: table
    story.append(Paragraph("Appendix A: Financial Summary", styles["Title"]))
    story.append(Spacer(1, 12))

    table_data = [["Description", "Hours", "Rate", "Amount"]]
    for item in INVOICE_ITEMS:
        table_data.append(list(item))
    table_data.append(["", "", "Total:", "$16,953.13"])

    t = Table(table_data, colWidths=[200, 80, 80, 100])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.HexColor("#F8F9FA"), colors.white]),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("LINEABOVE", (0, -1), (-1, -1), 1, colors.black),
    ]))
    story.append(t)
    story.append(PageBreak())

    # Page 3: mixed
    story.append(Paragraph("Appendix B: Key Metrics", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "The following metrics were collected during Q3 2025. "
        "All values represent monthly averages unless otherwise noted.", styles["BodyText"]
    ))
    story.append(Spacer(1, 12))

    metrics_data = [
        ["Metric", "Jan", "Feb", "Mar", "Q1 Avg"],
        ["Revenue ($K)", "245", "312", "289", "282"],
        ["Active Users", "12,450", "13,200", "14,100", "13,250"],
        ["Uptime (%)", "99.92", "99.97", "99.89", "99.93"],
        ["Avg Response (ms)", "142", "138", "145", "142"],
        ["Support Tickets", "234", "198", "267", "233"],
    ]
    t2 = Table(metrics_data, colWidths=[120, 80, 80, 80, 80])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495E")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F8F9FA"), colors.white]),
    ]))
    story.append(t2)
    story.append(Spacer(1, 20))
    story.append(Paragraph(LOREM, styles["BodyText"]))

    doc.build(story)
    print("  multipage.pdf")


# ---------------------------------------------------------------------------
# 9. multilingual.png — English + Ukrainian text
# ---------------------------------------------------------------------------
def generate_multilingual():
    w, h = 2480, 3508
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    font_title = get_font(28, bold=True)
    font_body = get_font(18)
    font_label = get_font(16, bold=True)

    draw.text((200, 150), "Bilingual Document / Двомовний документ", fill="black", font=font_title)

    y = 250
    draw.text((200, y), "ENGLISH:", fill="#2C3E50", font=font_label)
    y += 35

    en_text = (
        "This document serves as a test for multilingual OCR capabilities. "
        "The system should correctly recognize text in both English and Ukrainian, "
        "including special characters and diacritical marks. Accurate recognition "
        "of mixed-language documents is essential for international business operations."
    )
    for line in wrap_text(en_text, font_body, w - 400):
        draw.text((200, y), line, fill="black", font=font_body)
        y += 28
    y += 30

    draw.line([(200, y), (2280, y)], fill="#CCCCCC", width=1)
    y += 20

    draw.text((200, y), "UKRAINIAN / УКРАЇНСЬКА:", fill="#2C3E50", font=font_label)
    y += 35

    # Ukrainian text with proper characters
    uk_lines = [
        "Цей документ є тестом для багатомовних можливостей OCR.",
        "Система повинна правильно розпізнавати текст українською",
        "та англійською мовами, включаючи спеціальні символи.",
        "",
        "Точне розпізнавання змішаних документів є важливим",
        "для міжнародних бізнес-операцій та комунікацій.",
        "",
        "Додаткові символи: є, і, ї, ґ, Є, І, Ї, Ґ",
        "Числа: 1 234 567,89 грн",
        "Дата: 15 березня 2025 року",
    ]
    for line in uk_lines:
        draw.text((200, y), line, fill="black", font=font_body)
        y += 28
    y += 30

    draw.line([(200, y), (2280, y)], fill="#CCCCCC", width=1)
    y += 20

    # Mixed section
    draw.text((200, y), "MIXED / ЗМІШАНИЙ:", fill="#2C3E50", font=font_label)
    y += 35
    mixed_lines = [
        "Project Manager: John Smith / Менеджер проекту: Іван Петренко",
        "Budget: $45,000 / Бюджет: 1 845 000 грн",
        "Deadline: April 30, 2025 / Дедлайн: 30 квітня 2025",
        "Status: In Progress / Статус: В роботі",
    ]
    for line in mixed_lines:
        draw.text((200, y), line, fill="black", font=font_body)
        y += 28

    img.save(OUTPUT_DIR / "multilingual.png", dpi=(300, 300))
    print("  multilingual.png")


# ---------------------------------------------------------------------------
# 10. low-dpi.png — same content as simple-typed but at 100 DPI
# ---------------------------------------------------------------------------
def generate_low_dpi():
    # Generate at low resolution directly (simulating 100 DPI scan)
    scale = 100 / 300  # ratio to 300 DPI
    w, h = int(2480 * scale), int(3508 * scale)  # ~827 x 1169
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    font_title = get_font(14, bold=True)
    font_body = get_font(10)

    y = 67
    draw.text((67, y), "Quarterly Business Report — Q3 2025", fill="black", font=font_title)
    y += 27

    for para in PARAGRAPHS:
        lines = wrap_text(para, font_body, w - 134)
        for line in lines:
            draw.text((67, y), line, fill="black", font=font_body)
            y += 10
        y += 7

    lines = wrap_text(LOREM + " " + LOREM, font_body, w - 134)
    for line in lines:
        if y > h - 67:
            break
        draw.text((67, y), line, fill="black", font=font_body)
        y += 10

    img.save(OUTPUT_DIR / "low-dpi.png", dpi=(100, 100))
    print("  low-dpi.png")


# ---------------------------------------------------------------------------

def main():
    print("Generating test fixtures...")
    generate_simple_typed()
    generate_dense_twocol()
    generate_table_simple()
    generate_table_complex()
    generate_form_kvpairs()
    generate_noisy_scan()
    generate_mixed_content()
    generate_multipage_pdf()
    generate_multilingual()
    generate_low_dpi()
    print(f"\nDone. Files in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
