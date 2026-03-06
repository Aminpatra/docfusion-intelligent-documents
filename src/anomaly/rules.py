import numpy as np
import pandas as pd


# Minimum number of training receipts before we trust a vendor's own stats.
# Below this threshold we fall back to the global distribution.
_MIN_VENDOR_COUNT = 5


def robust_z(total, median, mad):
  return 0.6745 * (total - median) / mad


def predict_rules(df, vstats, z_thresh=3.5, rare_vendor_count=2, rare_vendor_high_total=300):
  preds = []
  scores = []
  reasons = []

  # Pull out global stats once — used as fallback for unknown / low-count vendors
  global_row = vstats[vstats["fields.vendor"] == "__GLOBAL__"]
  if not global_row.empty:
    global_median = float(global_row["vendor_median"].iloc[0])
    global_mad    = float(global_row["vendor_mad"].iloc[0])
    global_count  = int(global_row["vendor_count"].iloc[0])
  else:
    # Safety net if stats were built without the global row
    global_median, global_mad, global_count = None, None, 0

  # Merge vendor stats (excluding the synthetic __GLOBAL__ row)
  vendor_only = vstats[vstats["fields.vendor"] != "__GLOBAL__"]

  df2 = df.copy()
  df2["total_num"] = pd.to_numeric(df2["fields.total"], errors="coerce")
  df2 = df2.merge(vendor_only, how="left", on="fields.vendor")

  for _, r in df2.iterrows():
    rs = []
    score = 0.0
    is_forged = 0

    total = r.get("total_num", np.nan)

    # ── 1. Missing total ────────────────────────────────────────────────────
    # BUG FIX: previously added score=2.0 but never set is_forged=1 directly,
    # and 2.0 < 2.5 so the score threshold didn't trigger either.
    if pd.isna(total):
      is_forged = 1
      score += 2.0
      rs.append("missing_total")

    # ── 2. Vendor Z-score check ─────────────────────────────────────────────
    if not pd.isna(total):
      vendor_count  = r.get("vendor_count",  np.nan)
      vendor_median = r.get("vendor_median", np.nan)
      vendor_mad    = r.get("vendor_mad",    np.nan)

      use_vendor_stats = (
        pd.notna(vendor_count) and
        pd.notna(vendor_median) and
        pd.notna(vendor_mad) and
        vendor_count >= _MIN_VENDOR_COUNT
      )

      if use_vendor_stats:
        # Reliable per-vendor stats
        z = robust_z(total, vendor_median, vendor_mad)
        rs.append(f"vendor_z={z:.2f}")
        score += min(abs(z) / z_thresh, 3.0)
        if abs(z) >= z_thresh:
          is_forged = 1
          rs.append("vendor_total_outlier")

      elif global_median is not None:
        # Unknown vendor OR too few samples — use the global distribution.
        # Apply a gentler threshold (multiplied by 1.5) since global spread
        # is naturally wider than per-vendor spread.
        z = robust_z(total, global_median, global_mad)
        rs.append(f"global_z={z:.2f}")
        global_thresh = z_thresh * 1.5
        score += min(abs(z) / global_thresh, 2.0)

        if abs(z) >= global_thresh:
          is_forged = 1
          rs.append("global_total_outlier")

        # Flag unknown vendors with a small prior suspicion
        if pd.isna(vendor_count):
          score += 0.5
          rs.append("unknown_vendor")

    # ── 3. Rare vendor + high total ─────────────────────────────────────────
    vcnt = r.get("vendor_count", np.nan)
    if not pd.isna(total) and pd.notna(vcnt):
      if vcnt <= rare_vendor_count and total >= rare_vendor_high_total:
        score += 1.0
        is_forged = 1
        rs.append("rare_vendor_high_total")

    # ── 4. Score-based catch-all ────────────────────────────────────────────
    if score >= 2.5:
      is_forged = 1

    preds.append(int(is_forged))
    scores.append(float(score))
    reasons.append(rs)

  return preds, scores, reasons