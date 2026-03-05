import numpy as np
import pandas as pd


def robust_z(total, median, mad):
  return 0.6745 * (total - median) / mad


def predict_rules(df, vstats, z_thresh=3.5, rare_vendor_count=2, rare_vendor_high_total=300):
  out = []
  reasons = []

  df2 = df.copy()
  df2["total_num"] = pd.to_numeric(df2["fields.total"], errors="coerce")
  df2 = df2.merge(vstats, how="left", on="fields.vendor")

  for _, r in df2.iterrows():
    rs = []
    is_forged = 0

    total = r.get("total_num", np.nan)

    # 1) Missing total
    if pd.isna(total):
      is_forged = 1
      rs.append("missing_total")

    # 2) Vendor-based robust outlier
    if not pd.isna(total) and pd.notna(r.get("vendor_median")):
      z = robust_z(total, r["vendor_median"], r["vendor_mad"])
      if abs(z) >= z_thresh:
        is_forged = 1
        rs.append(f"vendor_total_outlier(z={z:.2f})")

    # 3) Rare vendor + high total (simple extra signal)
    vcnt = r.get("vendor_count", np.nan)
    if not pd.isna(total) and pd.notna(vcnt):
      if vcnt <= rare_vendor_count and total >= rare_vendor_high_total:
        is_forged = 1
        rs.append("rare_vendor_high_total")

    out.append(int(is_forged))
    reasons.append(rs)

  return out, reasons