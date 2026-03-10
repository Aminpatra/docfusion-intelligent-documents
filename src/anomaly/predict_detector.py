"""
predict_detector.py – loads and runs the trained hybrid anomaly detector.

Mirrors the structure of train_detector.py:
  1. Rule-based layer: vendor-level Z-score
  2. ML layer: GBM/LR on text + structured features
  3. Ensemble: weighted combination with configurable threshold
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from src.anomaly.feature_builder import build_features

logger = logging.getLogger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────

def load_detector(model_dir: str | Path):
    """
    Load all artefacts saved by train_detector.

    Returns
    -------
    (model, tfidf, manual_cols, model_type, constant_label, threshold, vendor_stats)
    """
    model_dir = Path(model_dir)

    model       = joblib.load(model_dir / "anomaly_model.joblib")
    tfidf       = joblib.load(model_dir / "tfidf.joblib")
    manual_cols = joblib.load(model_dir / "manual_cols.joblib")
    model_type  = joblib.load(model_dir / "model_type.joblib")
    threshold   = joblib.load(model_dir / "threshold.joblib")

    # constant_label only present when a constant model was saved
    const_path = model_dir / "constant_label.joblib"
    constant_label = joblib.load(const_path) if const_path.exists() else 0

    vendor_stats_path = model_dir / "vendor_stats.joblib"
    vendor_stats = (
        joblib.load(vendor_stats_path)
        if vendor_stats_path.exists()
        else pd.DataFrame(columns=["vendor", "count", "median", "mad"])
    )

    logger.info(
        "Loaded detector: type=%s  threshold=%.2f  vendor_stats=%d rows",
        model_type, threshold, len(vendor_stats),
    )
    return model, tfidf, manual_cols, model_type, constant_label, threshold, vendor_stats


def predict_forgery(
    df: pd.DataFrame,
    model,
    tfidf,
    manual_cols: list[str],
    model_type: str,
    constant_label: int,
    threshold: float,
    vendor_stats: pd.DataFrame,
) -> np.ndarray:
    """
    Run the hybrid anomaly detector on a DataFrame.

    Returns an array of floats in [0, 1] — the ensemble forgery probability
    for each row. The caller applies the threshold.
    """
    n = len(df)

    # ── Constant model ────────────────────────────────────────────────────────
    if model_type == "constant" or model is None:
        return np.full(n, float(constant_label))

    # ── ML probabilities ──────────────────────────────────────────────────────
    X_manual = build_features(df)

    # Align columns to those seen during training
    for col in manual_cols:
        if col not in X_manual.columns:
            X_manual[col] = 0.0
    X_manual = X_manual[manual_cols].fillna(0.0)

    text_series = df["text"].fillna("") if "text" in df.columns else pd.Series([""] * n)

    if model_type == "gbm_dense":
        if tfidf is not None:
            X_text = tfidf.transform(text_series).toarray()
        else:
            X_text = np.zeros((n, 1))
        X_dense = np.hstack([X_manual.values.astype(float), X_text])
        ml_prob = model.predict_proba(X_dense)[:, 1]

    elif model_type in ("lr_sparse", "lr_dense"):
        if tfidf is not None and model_type == "lr_sparse":
            X_text = tfidf.transform(text_series)
            X_sparse = hstack([csr_matrix(X_manual.values.astype(float)), X_text])
        else:
            X_sparse = X_manual.values.astype(float)
        ml_prob = model.predict_proba(X_sparse)[:, 1]

    else:
        logger.warning("Unknown model_type=%s — defaulting to rule-based only.", model_type)
        ml_prob = np.full(n, 0.1)

    # ── Rule-based Z-score probabilities ──────────────────────────────────────
    rule_prob = _rule_proba(df, vendor_stats)

    # ── Ensemble ──────────────────────────────────────────────────────────────
    ensemble_prob = 0.6 * ml_prob + 0.4 * rule_prob
    return ensemble_prob


# ── Internal helpers ─────────────────────────────────────────────────────────

def _rule_proba(df: pd.DataFrame, vendor_stats: pd.DataFrame) -> np.ndarray:
    """Convert rule-based Z-scores into [0,1] probabilities for ensemble use."""
    stats_map  = {str(r["vendor"]): r.to_dict() for _, r in vendor_stats.iterrows()}
    global_row = stats_map.get("__GLOBAL__", {})

    probs = []
    for _, row in df.iterrows():

        def _get(col):
            v = row.get(col)
            if v is None:
                return None
            try:
                if isinstance(v, float) and math.isnan(v):
                    return None
            except Exception:
                pass
            return v

        raw_total = _get("total") or _get("_total")
        total_num = pd.to_numeric(raw_total, errors="coerce")
        if pd.isna(total_num):
            probs.append(0.8)   # missing total → suspicious
            continue

        vendor = str(_get("vendor") or _get("_vendor") or "")
        vrow   = stats_map.get(vendor, {})

        if int(vrow.get("count", 0)) >= 5:
            med = float(vrow["median"])
            mad = max(float(vrow["mad"]), 0.01)
        elif global_row:
            med = float(global_row.get("median", total_num))
            mad = max(float(global_row.get("mad", 1.0)) * 1.5, 0.01)
        else:
            probs.append(0.1)
            continue

        z = abs(0.6745 * (float(total_num) - med) / mad)
        p = float(1 / (1 + np.exp(-0.5 * (z - 3.5))))
        probs.append(p)

    return np.array(probs)