"""
train_detector.py – trains a HYBRID anomaly detector.

Strategy:
  1. Rule-based layer: vendor-level Z-score (works even with tiny datasets)
  2. ML layer: GBM/LR on text + structured features
  3. Ensemble: combines both signals with configurable weights

CORD ratio cap: CORD is capped at 5× the competition genuine count so it
doesn't drown out the forgery signal the model needs to learn.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.anomaly.feature_builder import build_features

logger = logging.getLogger(__name__)

# Max ratio of CORD genuine : competition genuine
# e.g. 5 → if competition has 10 genuine, cap CORD at 50
_MAX_CORD_RATIO = 5


def _safe_tfidf(texts: pd.Series):
    for min_df in (2, 1):
        try:
            tfidf = TfidfVectorizer(
                lowercase=True, ngram_range=(1, 2),
                min_df=min_df, max_features=4000, sublinear_tf=True,
            )
            X = tfidf.fit_transform(texts)
            if X.shape[1] > 0:
                return tfidf, X
        except ValueError:
            continue
    return None, None


def _class_sample_weights(y: np.ndarray) -> np.ndarray:
    n_total  = len(y)
    n_forged = int(y.sum())
    n_genuine = n_total - n_forged
    w_genuine = n_total / (2 * max(n_genuine, 1))
    w_forged  = n_total / (2 * max(n_forged,  1))
    return np.where(y == 1, w_forged, w_genuine)


def _find_best_threshold(proba: np.ndarray, y: np.ndarray) -> float:
    best_t, best_f1 = 0.3, 0.0
    for t in np.arange(0.05, 0.95, 0.05):
        f = f1_score(y, (proba >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    logger.info("Best threshold: %.2f  (train F1=%.3f)", best_t, best_f1)
    return float(round(best_t, 2))


def _build_vendor_stats(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-vendor median total and MAD for rule-based Z-score detection.
    Adds a __GLOBAL__ row as fallback for unseen vendors.
    """
    df = train_df.copy()
    df["_total_num"] = pd.to_numeric(df["total"], errors="coerce")
    clean = df.dropna(subset=["vendor", "_total_num"])

    def mad(x):
        med = np.median(x)
        return np.median(np.abs(x - med))

    rows = []
    for vendor, grp in clean.groupby("vendor"):
        totals = grp["_total_num"].values
        med = float(np.median(totals))
        m = float(mad(totals))
        m = max(m, abs(med) * 0.05, 0.01)
        rows.append({"vendor": vendor, "count": len(totals),
                     "median": med, "mad": m})

    # Global fallback
    all_totals = clean["_total_num"].values
    if len(all_totals) > 0:
        gmed = float(np.median(all_totals))
        gmad = float(mad(all_totals))
        gmad = max(gmad, abs(gmed) * 0.05, 0.01)
        rows.append({"vendor": "__GLOBAL__", "count": len(all_totals),
                     "median": gmed, "mad": gmad})

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["vendor", "count", "median", "mad"]
    )


def _balance_cord(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cap CORD records so they don't overwhelm competition genuine records.
    CORD records are identified by id starting with 'cord_'.
    """
    is_cord = df["id"].str.startswith("cord_")
    competition_df = df[~is_cord]
    cord_df = df[is_cord]

    if len(cord_df) == 0:
        return df

    n_comp_genuine = int((competition_df["is_forged"] == 0).sum())
    max_cord = max(n_comp_genuine * _MAX_CORD_RATIO, 50)

    if len(cord_df) > max_cord:
        cord_df = cord_df.sample(n=max_cord, random_state=42)
        logger.info(
            "CORD capped from %d → %d records (ratio cap %d×)",
            is_cord.sum(), max_cord, _MAX_CORD_RATIO,
        )

    return pd.concat([competition_df, cord_df], ignore_index=True)


def train_detector(train_df: pd.DataFrame, model_dir: str | Path) -> Path:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Balance CORD records ──────────────────────────────────────────────────
    train_df = _balance_cord(train_df)

    y = train_df["is_forged"].astype(int).values
    n_forged  = int(y.sum())
    n_genuine = len(y) - n_forged
    n_total   = len(y)

    logger.info("Effective training set: %d genuine, %d forged (total %d)",
                n_genuine, n_forged, n_total)

    # ── Rule-based: vendor stats (always computed) ────────────────────────────
    vendor_stats = _build_vendor_stats(train_df)
    joblib.dump(vendor_stats, model_dir / "vendor_stats.joblib")
    logger.info("Saved vendor stats for %d vendors", len(vendor_stats) - 1)

    # ── Constant model if no class variation ─────────────────────────────────
    if n_forged == 0 or n_genuine == 0:
        logger.warning("Only one class — saving constant model.")
        constant = int(y[0]) if n_total > 0 else 0
        joblib.dump(None,      model_dir / "anomaly_model.joblib")
        joblib.dump(None,      model_dir / "tfidf.joblib")
        joblib.dump([],        model_dir / "manual_cols.joblib")
        joblib.dump("constant",model_dir / "model_type.joblib")
        joblib.dump(constant,  model_dir / "constant_label.joblib")
        joblib.dump(0.5,       model_dir / "threshold.joblib")
        return model_dir

    # ── ML model ──────────────────────────────────────────────────────────────
    X_manual = build_features(train_df)
    manual_cols = list(X_manual.columns)
    tfidf, X_text = _safe_tfidf(train_df["text"].fillna(""))
    has_text = tfidf is not None
    sample_weight = _class_sample_weights(y)

    if n_forged >= 5 and n_genuine >= 5:
        X_dense = np.hstack([
            X_manual.values.astype(float),
            X_text.toarray() if has_text else np.zeros((n_total, 1))
        ])
        model = GradientBoostingClassifier(
            n_estimators=300, max_depth=4,
            learning_rate=0.05, subsample=0.8,
            min_samples_leaf=1, random_state=42,
        )
        model.fit(X_dense, y, sample_weight=sample_weight)
        model_type = "gbm_dense"
        ml_prob = model.predict_proba(X_dense)[:, 1]
    else:
        X_sparse = hstack([csr_matrix(X_manual.values.astype(float)), X_text]) \
                   if has_text else X_manual.values.astype(float)
        model = LogisticRegression(max_iter=2000, class_weight="balanced")
        model.fit(X_sparse, y)
        model_type = "lr_sparse" if has_text else "lr_dense"
        ml_prob = model.predict_proba(X_sparse)[:, 1]

    # ── Rule-based Z-score probabilities ─────────────────────────────────────
    rule_prob = _rule_proba(train_df, vendor_stats)

    # ── Ensemble: weighted average ────────────────────────────────────────────
    # Give rules 40% weight, ML 60% — rules are reliable even with tiny data
    ensemble_prob = 0.6 * ml_prob + 0.4 * rule_prob

    threshold = _find_best_threshold(ensemble_prob, y)

    # ── Persist ───────────────────────────────────────────────────────────────
    joblib.dump(model,       model_dir / "anomaly_model.joblib")
    joblib.dump(tfidf,       model_dir / "tfidf.joblib")
    joblib.dump(manual_cols, model_dir / "manual_cols.joblib")
    joblib.dump(model_type,  model_dir / "model_type.joblib")
    joblib.dump(threshold,   model_dir / "threshold.joblib")

    logger.info("Saved model (type=%s, threshold=%.2f)", model_type, threshold)
    return model_dir


def _rule_proba(df: pd.DataFrame, vendor_stats: pd.DataFrame) -> np.ndarray:
    """Convert rule-based Z-scores into [0,1] probabilities for ensemble use."""
    stats_map  = {str(r["vendor"]): r.to_dict() for _, r in vendor_stats.iterrows()}
    global_row = stats_map.get("__GLOBAL__", {})

    probs = []
    for _, row in df.iterrows():
        # Safe scalar extraction — avoids "truth value of Series is ambiguous"
        def _get(col):
            v = row.get(col)
            if v is None:
                return None
            try:
                import math
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
        vrow = stats_map.get(vendor, {})

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