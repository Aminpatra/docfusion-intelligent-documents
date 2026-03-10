from pathlib import Path
import re

from src.extraction.sroie_extractor import (
    extract_fields_from_text as extract_fields_from_ocr_file,
)


def extract_fields_from_lines(lines):
    if lines is None:
        lines = []

    lines = [str(x).strip() for x in lines if str(x).strip()]
    text = "\n".join(lines)

    vendor = None
    date = None
    total = None

    # vendor: first non-trivial line
    for line in lines[:5]:
        if len(line) >= 3 and not re.fullmatch(r"[\d\W]+", line):
            vendor = line
            break

    # date patterns
    date_patterns = [
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{2}/\d{2}/\d{4}\b",
        r"\b\d{2}-\d{2}-\d{4}\b",
        r"\b\d{2}/\d{2}/\d{2}\b",
        r"\b\d{2}-\d{2}-\d{2}\b",
    ]

    for pat in date_patterns:
        m = re.search(pat, text)
        if m:
            date = m.group(0)
            break

    amount_pattern = r"\b\d+[.,]\d{2}\b"

    total_candidates = []
    for line in lines:
        if "total" in line.lower():
            total_candidates.extend(re.findall(amount_pattern, line))

    if not total_candidates:
        total_candidates = re.findall(amount_pattern, text)

    if total_candidates:
        def parse_amt(x):
            return float(x.replace(",", "."))
        try:
            total = max(total_candidates, key=parse_amt)
        except Exception:
            total = total_candidates[-1]

    return {
        "vendor": vendor,
        "date": date,
        "total": total,
    }


def extract_fields(doc_id=None, lines=None, ocr_path=None):
    """
    Unified interface for the pipeline.
    Prefer OCR-box extraction if available, otherwise use plain text lines.
    """
    if ocr_path is not None and Path(ocr_path).exists():
        return extract_fields_from_ocr_file(ocr_path)

    return extract_fields_from_lines(lines)