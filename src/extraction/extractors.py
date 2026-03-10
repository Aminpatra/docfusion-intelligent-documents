"""
extractors.py – unified field extraction for the DocFusion pipeline.

Two extraction paths:
  1. Box-based  (SROIE .txt annotation files) – highest accuracy
  2. Text-lines (plain list of strings from JSONL ocr_lines)  – used by harness

The pipeline calls extract_fields() with keyword args; this module routes to the
correct sub-extractor automatically.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from src.extraction.sroie_extractor import (
    extract_fields_from_text as _box_extract,
)

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

_AMOUNT_RE = re.compile(r"\b\d{1,6}[.,]\d{2}\b")

_DATE_PATTERNS = [
    re.compile(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b"),
    re.compile(r"\b\d{2}[-/]\d{2}[-/]\d{4}\b"),
    re.compile(r"\b\d{1,2}[-/ ]\w{3,9}[-/ ]\d{4}\b"),
    re.compile(r"\b\d{2}[-/]\d{2}[-/]\d{2}\b"),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
]

_BAD_VENDOR_KEYWORDS = {
    "date", "time", "cashier", "member", "tel", "phone", "fax",
    "tax", "gst", "qty", "amount", "total", "change", "cash",
    "invoice", "receipt", "bill", "ref", "no:", "thank you",
    "www.", "http", "subtotal", "sub-total", "discount",
    "address", "email", "@",
}

_BIZ_KEYWORDS = {
    "sdn", "bhd", "s/b", "enterprise", "enterprises", "trading",
    "restaurant", "restoran", "hardware", "mart", "supermarket",
    "store", "stores", "shop", "shops", "company", "co.", "ltd",
    "limited", "pte", "perniagaan", "kedai", "perusahaan",
    "pharmacy", "clinic", "medical", "hotel", "cafe", "bakery",
    "electrical", "electronics", "food", "kopitiam",
    "stationery", "books", "hypermarket",
}

_TOTAL_KWS = [
    "total sales (inclusive", "total sales inclusive", "inclusive of gst",
    "total (incl", "total incl", "total including gst", "total inc gst",
    "rounded total", "nett total", "net total", "amount due",
    "grand total", "total (rm", "total rm", "jumlah rm", "jumlah",
    "total sales", "total", "nett", "net",
]

_BAD_TOTAL_KWS = {
    "cash", "change", "gst", "tax", "sub total", "subtotal",
    "rounding", "adjust", "discount", "service charge", "tips",
    "qty", "quantity", "excluding",
}


# ---------------------------------------------------------------------------
# Extraction from plain text lines
# ---------------------------------------------------------------------------

def _score_vendor_line(text: str, rank: int) -> float:
    t = text.strip()
    tl = t.lower()
    score = max(0.0, 18.0 - rank * 3.0)

    for kw in _BIZ_KEYWORDS:
        if kw in tl:
            score += 12
            break

    for kw in _BAD_VENDOR_KEYWORDS:
        if kw in tl:
            score -= 8

    length = len(t)
    if 5 <= length <= 60:
        score += length * 0.12
    elif length > 60:
        score -= (length - 60) * 0.15

    digit_count = sum(c.isdigit() for c in t)
    if digit_count >= 5:
        score -= 10
    elif digit_count >= 3:
        score -= 4

    if re.search(r"\b\d{5}\b", t):
        score -= 15

    if re.fullmatch(r"[\d\s.\-/]+", t):
        score -= 30

    if len(t) < 3:
        score -= 100

    return score


def _extract_vendor_from_lines(lines):
    header = lines[:10]
    candidates = []
    for rank, line in enumerate(header):
        t = line.strip()
        if not t:
            continue
        sc = _score_vendor_line(t, rank)
        candidates.append((sc, t))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0][1]
    best = re.sub(r"\s*\([^)]*\d{4,}[^)]*\)\s*$", "", best).strip()
    return best or None


def _extract_date_from_lines(lines):
    text_full = "\n".join(lines)
    for pat in _DATE_PATTERNS:
        m = pat.search(text_full)
        if m:
            return m.group(0)
    return None


def _parse_amount(s: str) -> float:
    return float(s.replace(",", "."))


def _extract_total_from_lines(lines):
    candidates = []

    for kw in _TOTAL_KWS:
        for line in reversed(lines):
            ll = line.lower()
            if kw in ll:
                is_inclusive_kw = kw in (
                    "total sales (inclusive", "total sales inclusive",
                    "inclusive of gst", "total (incl", "total incl",
                    "total including gst",
                )
                if any(b in ll for b in _BAD_TOTAL_KWS) and not is_inclusive_kw:
                    continue

                amounts = _AMOUNT_RE.findall(line)
                if amounts:
                    try:
                        val = max(amounts, key=_parse_amount)
                        candidates.append((_parse_amount(val), val))
                    except Exception:
                        pass
                break

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_val, _ = candidates[0]
        return f"{best_val:.2f}"

    text_full = "\n".join(lines)
    all_amounts = _AMOUNT_RE.findall(text_full)
    if not all_amounts:
        return None

    try:
        best = max(all_amounts, key=_parse_amount)
        return f"{_parse_amount(best):.2f}"
    except Exception:
        return None


def extract_fields_from_lines(lines):
    """Full field extraction from a plain list of OCR text lines."""
    if lines is None:
        lines = []
    lines = [str(x).strip() for x in lines if str(x).strip()]

    return {
        "vendor": _extract_vendor_from_lines(lines),
        "date": _extract_date_from_lines(lines),
        "total": _extract_total_from_lines(lines),
    }


# ---------------------------------------------------------------------------
# Unified entry point used by the pipeline
# ---------------------------------------------------------------------------

def extract_fields(
    doc_id=None,
    lines=None,
    image_path=None,   # image file (not directly used for extraction)
    ocr_path=None,     # SROIE-style OCR box annotation file
):
    """
    Route to the best available extractor.

    Priority:
      1. SROIE box-annotation file (ocr_path)
      2. Plain text lines (lines)  <- used by the autograder harness
      3. Empty result
    """
    if ocr_path is not None and Path(str(ocr_path)).exists():
        try:
            return _box_extract(ocr_path)
        except Exception:
            pass

    if lines:
        return extract_fields_from_lines(lines)

    return {"vendor": None, "date": None, "total": None}