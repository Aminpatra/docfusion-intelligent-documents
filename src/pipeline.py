"""
pipeline.py – orchestrates extraction + anomaly detection for the harness.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.extraction.extractors import extract_fields_from_lines
from src.anomaly.predict_detector import load_detector, predict_forgery

logger = logging.getLogger(__name__)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_image_path(images_dir: Path, doc_id: str) -> str:
    for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
        p = images_dir / f"{doc_id}{ext}"
        if p.exists():
            return str(p)
    return str(images_dir / f"{doc_id}.png")


def prepare_dataframe(records, images_dir):
    rows = []
    for rec in records:
        doc_id = rec["id"]

        pre_fields = rec.get("fields") or {}
        vendor = pre_fields.get("vendor") or rec.get("vendor")
        date   = pre_fields.get("date")   or rec.get("date")
        total  = pre_fields.get("total")  or rec.get("total")

        ocr_lines = rec.get("ocr_lines", [])
        if not isinstance(ocr_lines, list):
            ocr_lines = []

        if ocr_lines:
            text = "\n".join(str(x) for x in ocr_lines if x is not None)
        else:
            parts = []
            if vendor: parts.append(f"vendor {vendor}")
            if date:   parts.append(f"date {date}")
            if total:  parts.append(f"total {total}")
            text = " ".join(parts)

        rows.append({
            "id":         doc_id,
            "image_path": _resolve_image_path(Path(images_dir), doc_id),
            "lines":      ocr_lines,
            "text":       text,
            "_vendor":    vendor,
            "_date":      date,
            "_total":     total,
            "vendor":     vendor,   # also at top level for rule-based features
            "total":      total,
            "_has_pre":   bool(pre_fields),
        })

    return pd.DataFrame(rows)


def run_pipeline(model_dir, records, images_dir):
    df = prepare_dataframe(records, images_dir)

    # ── Anomaly detection ─────────────────────────────────────────────────────
    try:
        anomaly_model, tfidf, manual_cols, model_type, \
            constant_label, threshold, vendor_stats = \
            load_detector(Path(model_dir) / "anomaly")

        forged_prob = predict_forgery(
            df, anomaly_model, tfidf, manual_cols,
            model_type, constant_label, threshold, vendor_stats,
        )
        df["forged_prob"] = forged_prob
        df["is_forged"] = (df["forged_prob"] >= threshold).astype(int)

        logger.info(
            "Threshold=%.2f | predicted forged=%d/%d",
            threshold, int(df["is_forged"].sum()), len(df),
        )
    except Exception as exc:
        logger.warning("Anomaly model failed: %s — defaulting to 0", exc)
        df["is_forged"] = 0
        df["forged_prob"] = 0.0

    # ── Field extraction ──────────────────────────────────────────────────────
    predictions = []
    for _, row in df.iterrows():
        try:
            if row["_has_pre"]:
                fields = {
                    "vendor": row["_vendor"],
                    "date":   row["_date"],
                    "total":  row["_total"],
                }
            elif row["lines"]:
                fields = extract_fields_from_lines(row["lines"])
            else:
                fields = {"vendor": None, "date": None, "total": None}
        except Exception as exc:
            logger.warning("Extraction failed for %s: %s", row["id"], exc)
            fields = {"vendor": None, "date": None, "total": None}

        predictions.append({
            "id":        row["id"],
            "vendor":    fields.get("vendor"),
            "date":      fields.get("date"),
            "total":     fields.get("total"),
            "is_forged": int(row["is_forged"]),
        })

    return predictions