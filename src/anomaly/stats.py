import numpy as np
import pandas as pd


def vendor_stats(train_df):
  df = train_df.copy()
  df["total_num"] = pd.to_numeric(df["fields.total"], errors="coerce")

  g = df.dropna(subset=["fields.vendor", "total_num"]).groupby("fields.vendor")["total_num"]

  stats = g.agg(["count", "median"]).rename(columns={"count": "vendor_count", "median": "vendor_median"})

  def mad(x):
    med = np.median(x)
    return np.median(np.abs(x - med))

  stats["vendor_mad"] = g.apply(mad)

  # Avoid division by zero
  stats["vendor_mad"] = stats["vendor_mad"].replace(0, 1e-9)

  return stats.reset_index()