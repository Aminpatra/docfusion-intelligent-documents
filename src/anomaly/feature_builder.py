"""
feature_builder.py – shared feature engineering, dependency-free.

Handles both data schemas:
  Schema A (harness/real): has 'ocr_lines', no pre-extracted fields
  Schema B (dummy/structured): has 'fields' dict, no ocr_lines

In both cases the DataFrame passed here will have columns:
  text       – OCR text or synthetic text built from fields
  vendor     – extracted/provided vendor string (or NaN)
  date       – extracted/provided date string (or NaN)
  total      – extracted/provided total string (or NaN)
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

_amount_re = re.compile(r"\b\d+[.,]\d{2}\b")
_date_re = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
_digit_re = re.compile(r"\d")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    text = df["text"].fillna("") if "text" in df.columns else pd.Series("", index=df.index)

    # --- Text-based features ------------------------------------------------
    out["n_chars"]    = text.str.len()
    out["n_lines"]    = text.str.count("\n") + 1
    out["n_words"]    = text.str.split().apply(len)
    out["digit_count"]  = text.apply(lambda x: len(_digit_re.findall(x)))
    out["amount_count"] = text.apply(lambda x: len(_amount_re.findall(x)))
    out["date_count"]   = text.apply(lambda x: len(_date_re.findall(x)))
    out["has_total"]    = text.str.contains(r"\btotal\b",    case=False, regex=True).astype(int)
    out["has_tax"]      = text.str.contains(r"\btax\b|\bgst\b", case=False, regex=True).astype(int)
    out["has_date_kw"]  = text.str.contains(r"\bdate\b",    case=False, regex=True).astype(int)
    out["has_cash"]     = text.str.contains(r"\bcash\b",    case=False, regex=True).astype(int)
    out["digit_ratio"]  = out["digit_count"] / out["n_chars"].clip(lower=1)

    # Amount value features from text
    def _amounts(t):
        vals = []
        for m in _amount_re.findall(t):
            try:
                vals.append(float(m.replace(",", ".")))
            except Exception:
                pass
        return vals

    amounts = text.apply(_amounts)
    out["max_amount"]     = amounts.apply(lambda vs: max(vs) if vs else 0.0)
    out["mean_amount"]    = amounts.apply(lambda vs: float(np.mean(vs)) if vs else 0.0)
    out["log_max_amount"] = np.log1p(out["max_amount"])

    # --- Structured field features (work even with empty text) ---------------
    vendor_col = df.get("vendor", pd.Series(dtype=object))
    date_col   = df.get("date",   pd.Series(dtype=object))
    total_col  = df.get("total",  pd.Series(dtype=object))

    out["missing_vendor"] = vendor_col.isna().astype(int)
    out["missing_date"]   = date_col.isna().astype(int)
    out["missing_total"]  = total_col.isna().astype(int)
    out["missing_count"]  = out["missing_vendor"] + out["missing_date"] + out["missing_total"]

    # Numeric total
    total_num = pd.to_numeric(total_col, errors="coerce")
    out["total_num"]     = total_num.fillna(0.0)
    out["log_total"]     = np.log1p(total_num.fillna(0.0))
    out["total_is_null"] = total_num.isna().astype(int)

    # Vendor string features
    vendor_str = vendor_col.fillna("").astype(str)
    out["vendor_len"]       = vendor_str.str.len()
    out["vendor_has_digit"] = vendor_str.str.contains(r"\d", regex=True).astype(int)

    # Date validity
    dates = pd.to_datetime(date_col, errors="coerce")
    out["date_valid"] = dates.notna().astype(int)
    out["date_year"]  = dates.dt.year.fillna(-1).astype(float)
    out["date_month"] = dates.dt.month.fillna(-1).astype(float)

    return out.fillna(0.0)