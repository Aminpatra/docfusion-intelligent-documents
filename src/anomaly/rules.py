import numpy as np
import pandas as pd


def robust_z(total, median, mad):
  return 0.6745 * (total - median) / mad


def predict_rules(df, vstats, z_thresh=3.5, rare_vendor_count=2, rare_vendor_high_total=300):
  preds = []
  scores = []
  reasons = []

  df2 = df.copy()
  df2["total_num"] = pd.to_numeric(df2["fields.total"], errors="coerce")
  df2 = df2.merge(vstats, how="left", on="fields.vendor")

  for _, r in df2.iterrows():
    rs = []
    score = 0.0
    is_forged = 0

    total = r.get("total_num", np.nan)

    # Missing total
    if pd.isna(total):
      score += 2.0
      rs.append("missing_total")

    # Vendor-based robust outlier
    if not pd.isna(total) and pd.notna(r.get("vendor_median")):
      z = robust_z(total, r["vendor_median"], r["vendor_mad"])
      rs.append(f"vendor_z={z:.2f}")
      score += min(abs(z) / z_thresh, 3.0)  # capped contribution

      if abs(z) >= z_thresh:
        is_forged = 1
        rs.append("vendor_total_outlier")

    # Rare vendor + high total
    vcnt = r.get("vendor_count", np.nan)
    if not pd.isna(total) and pd.notna(vcnt):
      if vcnt <= rare_vendor_count and total >= rare_vendor_high_total:
        score += 1.0
        is_forged = 1
        rs.append("rare_vendor_high_total")

    # Final decision: either rule triggered OR high score
    if score >= 2.5:
      is_forged = 1

    preds.append(int(is_forged))
    scores.append(float(score))
    reasons.append(rs)

  return preds, scores, reasons