"""
DocFusion Web UI  –  Level 3B
Run with:  streamlit run ui/app.py
"""

from __future__ import annotations

import io
import re
import sys
import tempfile
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocFusion – Intelligent Document Analyser",
    page_icon="🔍",
    layout="wide",
)

# ── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_resource
def get_ocr_engine():
    """Try PaddleOCR, fall back to Tesseract."""
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        return ("paddle", ocr)
    except Exception:
        pass

    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return ("tesseract", None)
    except Exception:
        pass

    return ("none", None)


def run_ocr(image: Image.Image) -> tuple[list[str], list[dict]]:
    """
    Run OCR on a PIL image.
    Returns:
        lines  – list of text strings
        boxes  – list of {"text", "x", "y", "w", "h"} for highlighting
    """
    engine, handle = get_ocr_engine()
    lines = []
    boxes = []

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    if engine == "paddle":
        result = handle.ocr(tmp_path, cls=True)
        for block in (result or []):
            for item in (block or []):
                coords, (text, conf) = item
                xs = [p[0] for p in coords]
                ys = [p[1] for p in coords]
                boxes.append({
                    "text": text,
                    "x": int(min(xs)), "y": int(min(ys)),
                    "w": int(max(xs) - min(xs)),
                    "h": int(max(ys) - min(ys)),
                })
                lines.append(text)

    elif engine == "tesseract":
        import pytesseract
        data = pytesseract.image_to_data(
            image, output_type=pytesseract.Output.DICT
        )
        for i, text in enumerate(data["text"]):
            text = str(text).strip()
            if not text:
                continue
            boxes.append({
                "text": text,
                "x": data["left"][i], "y": data["top"][i],
                "w": data["width"][i], "h": data["height"][i],
            })
            lines.append(text)

    else:
        # Last resort: use any text embedded in the image filename or show a warning
        lines = []
        boxes = []

    return lines, boxes


def highlight_image(
    image: Image.Image,
    boxes: list[dict],
    suspicious_terms: list[str],
    fields: dict,
) -> Image.Image:
    """Draw bounding boxes on the image. Red = suspicious field, green = normal."""
    img = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    suspicious_lower = {t.lower() for t in suspicious_terms}

    for box in boxes:
        text = box["text"].lower()
        is_suspicious = any(s in text for s in suspicious_lower)
        color = (220, 50, 50, 80) if is_suspicious else (50, 180, 50, 40)
        outline = (220, 50, 50, 200) if is_suspicious else (50, 180, 50, 120)

        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        draw.rectangle([x, y, x + w, y + h], fill=color, outline=outline, width=2)

    result = Image.alpha_composite(img, overlay).convert("RGB")
    return result


def detect_anomaly_rules(fields: dict, text: str) -> tuple[int, list[str]]:
    """
    Simple rule-based anomaly detection for the UI demo
    (does not require a trained model).
    """
    reasons = []

    vendor = fields.get("vendor")
    date = fields.get("date")
    total = fields.get("total")

    if not vendor:
        reasons.append("❌ Vendor name missing")
    if not date:
        reasons.append("❌ Transaction date missing")
    if not total:
        reasons.append("❌ Total amount missing")

    if total:
        try:
            t = float(str(total).replace(",", "."))
            if t > 5000:
                reasons.append(f"⚠️ Unusually high total: {t:.2f}")
            if t <= 0:
                reasons.append(f"⚠️ Non-positive total: {t:.2f}")
        except ValueError:
            reasons.append("⚠️ Total is not a valid number")

    # Check for multiple different totals (inconsistency)
    amounts = re.findall(r"\b\d+[.,]\d{2}\b", text)
    if amounts and total:
        try:
            total_val = float(str(total).replace(",", "."))
            parsed = [float(a.replace(",", ".")) for a in amounts]
            large = [v for v in parsed if v > total_val * 1.05]
            if large:
                reasons.append(
                    f"⚠️ Values larger than stated total found: {large[:3]}"
                )
        except Exception:
            pass

    if vendor and any(kw in str(vendor).lower() for kw in ["unknown", "test", "xxx"]):
        reasons.append("⚠️ Vendor name looks synthetic")

    is_forged = 1 if len(reasons) >= 2 else 0
    return is_forged, reasons


# ── Main UI ──────────────────────────────────────────────────────────────────

st.title("🔍 DocFusion — Intelligent Document Analyser")
st.markdown(
    "Upload a receipt or invoice image. The system will extract key fields "
    "and flag potentially suspicious documents."
)

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📤 Upload Document")
    uploaded = st.file_uploader(
        "Supported formats: PNG, JPG, JPEG, TIFF",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
    )

    threshold = st.slider(
        "Anomaly sensitivity (higher = more suspicious flags)",
        min_value=1, max_value=5, value=2,
        help="Number of rule violations needed to mark a document as forged."
    )

if uploaded is not None:
    image = Image.open(io.BytesIO(uploaded.read()))

    with st.spinner("Running OCR and field extraction…"):
        lines, boxes = run_ocr(image)

        if not lines:
            st.warning(
                "⚠️ No OCR engine found. Install PaddleOCR or Tesseract. "
                "Showing raw image only."
            )

        # Extract fields from OCR lines
        from src.extraction.extractors import extract_fields_from_lines
        fields = extract_fields_from_lines(lines)

        # Anomaly detection
        text = "\n".join(lines)
        is_forged, reasons = detect_anomaly_rules(fields, text)
        # Override with threshold
        is_forged = 1 if len([r for r in reasons if r.startswith("❌") or r.startswith("⚠️")]) >= threshold else 0

    with col_left:
        # Determine which terms to highlight
        suspicious_terms = []
        if not fields.get("vendor"):
            suspicious_terms += [l for l in lines[:5]]
        if fields.get("total"):
            suspicious_terms.append(fields["total"])

        highlighted = highlight_image(image, boxes, suspicious_terms, fields)
        st.image(highlighted, caption="Document (green = extracted fields, red = suspicious)", use_container_width=True)

    with col_right:
        st.subheader("📋 Extracted Fields")

        status_color = "#e74c3c" if is_forged else "#27ae60"
        status_label = "🚨 SUSPICIOUS / FORGED" if is_forged else "✅ APPEARS GENUINE"
        st.markdown(
            f"<div style='background:{status_color};color:white;padding:12px;"
            f"border-radius:8px;font-size:1.1em;font-weight:bold;text-align:center'>"
            f"{status_label}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Field display
        field_data = {
            "🏪 Vendor": fields.get("vendor") or "—",
            "📅 Date": fields.get("date") or "—",
            "💰 Total": fields.get("total") or "—",
        }

        for label, value in field_data.items():
            is_missing = value == "—"
            bg = "#fff3f3" if is_missing else "#f0fff4"
            icon = "❌" if is_missing else "✅"
            st.markdown(
                f"<div style='background:{bg};padding:10px;border-radius:6px;"
                f"margin-bottom:8px'><b>{label}</b>: {icon} {value}</div>",
                unsafe_allow_html=True,
            )

        st.subheader("🕵️ Anomaly Report")

        if reasons:
            for r in reasons:
                is_error = r.startswith("❌")
                color = "#e74c3c" if is_error else "#e67e22"
                st.markdown(
                    f"<div style='color:{color};padding:4px 0'>{r}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.success("No anomalies detected.")

        st.markdown("---")
        st.subheader("📄 Raw OCR Output")
        if lines:
            st.text_area("OCR Lines", "\n".join(lines), height=200)
        else:
            st.info("No OCR text extracted.")

        # JSON export
        result_json = {
            "vendor": fields.get("vendor"),
            "date": fields.get("date"),
            "total": fields.get("total"),
            "is_forged": is_forged,
            "anomaly_reasons": reasons,
        }
        st.download_button(
            "⬇️ Download Result JSON",
            data=__import__("json").dumps(result_json, indent=2),
            file_name="docfusion_result.json",
            mime="application/json",
        )

else:
    with col_right:
        st.info("👈 Upload a receipt or invoice image to get started.")
        st.markdown("""
**What this tool does:**
- 🔤 **OCR** — reads text from scanned documents
- 📋 **Field Extraction** — automatically pulls vendor, date, and total
- 🕵️ **Anomaly Detection** — flags suspicious documents based on heuristic rules
- 🎨 **Visual Overlay** — highlights extracted regions on the original image

**Supported document types:**
Receipts, invoices, scanned forms
        """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#888'>DocFusion Challenge — "
    "Intelligent Document Processing Pipeline</p>",
    unsafe_allow_html=True,
)