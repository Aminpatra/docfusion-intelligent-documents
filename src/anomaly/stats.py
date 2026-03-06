import numpy as np
import pandas as pd


def _mad(x):
  med = np.median(x)
  return np.median(np.abs(x - med))


def vendor_stats(train_df):
  df = train_df.copy()
  df["total_num"] = pd.to_numeric(df["fields.total"], errors="coerce")

  clean = df.dropna(subset=["fields.vendor", "total_num"])
  g = clean.groupby("fields.vendor")["total_num"]

  stats = g.agg(["count", "median"]).rename(
    columns={"count": "vendor_count", "median": "vendor_median"}
  )
  stats["vendor_mad"] = g.apply(_mad)

  # Use a relative MAD floor: at least 5% of the vendor median.
  # The old absolute floor of 1e-9 caused Z-scores in the billions for any
  # vendor whose training receipts all had the same total (MAD=0).
  stats["vendor_mad"] = np.maximum(
    stats["vendor_mad"],
    stats["vendor_median"].abs() * 0.05
  )
  # Hard minimum to guard against zero-median edge cases
  stats["vendor_mad"] = stats["vendor_mad"].replace(0, 1.0).clip(lower=0.01)

  stats = stats.reset_index()

  # ── Global fallback stats ────────────────────────────────────────────────
  # Used for (a) vendors not seen in training, (b) low-count vendors whose
  # per-vendor MAD is not yet reliable.
  all_totals = clean["total_num"].values
  global_median = float(np.median(all_totals))
  global_mad = float(_mad(all_totals))
  global_mad = max(global_mad, global_median * 0.05, 0.01)

  global_row = pd.DataFrame([{
    "fields.vendor": "__GLOBAL__",
    "vendor_count": len(all_totals),
    "vendor_median": global_median,
    "vendor_mad": global_mad,
  }])
  stats = pd.concat([stats, global_row], ignore_index=True)

  return stats