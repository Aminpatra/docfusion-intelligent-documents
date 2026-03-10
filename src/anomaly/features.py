import pandas as pd
import numpy as np


def build_features(df):

  features = pd.DataFrame()

  # numeric total
  features["total"] = pd.to_numeric(df["fields.total"], errors="coerce")

  # log transform (helps ML models)
  features["log_total"] = np.log1p(features["total"])

  # vendor frequency
  vendor_counts = df["fields.vendor"].value_counts()
  features["vendor_freq"] = df["fields.vendor"].map(vendor_counts)

  # date features
  dates = pd.to_datetime(df["fields.date"], errors="coerce")

  features["year"] = dates.dt.year
  features["month"] = dates.dt.month
  features["day"] = dates.dt.day

  return features


def build_features_v2(df, vstats=None):
  out = pd.DataFrame(index=df.index)

  out["total_num"] = pd.to_numeric(df["fields.total"], errors="coerce")
  out["log_total"] = np.log1p(out["total_num"])

  dates = pd.to_datetime(df["fields.date"], errors="coerce")
  out["date_valid"] = dates.notna().astype(int)
  out["year"] = dates.dt.year.fillna(-1)
  out["month"] = dates.dt.month.fillna(-1)
  out["day"] = dates.dt.day.fillna(-1)

  out["vendor_len"] = df["fields.vendor"].fillna("").astype(str).str.len()
  out["vendor_has_digit"] = df["fields.vendor"].fillna("").astype(str).str.contains(r"\d").astype(int)

  out["missing_vendor"] = df["fields.vendor"].isna().astype(int)
  out["missing_date"] = df["fields.date"].isna().astype(int)
  out["missing_total"] = df["fields.total"].isna().astype(int)
  out["missing_count"] = out[["missing_vendor", "missing_date", "missing_total"]].sum(axis=1)

  if vstats is not None:
    tmp = df.merge(vstats, how="left", left_on="fields.vendor", right_on="fields.vendor")
    out["vendor_count"] = tmp["vendor_count"].fillna(0)
    out["vendor_median"] = tmp["vendor_median"].fillna(np.nan)
    out["vendor_mad"] = tmp["vendor_mad"].fillna(np.nan)

    z = 0.6745 * (out["total_num"] - out["vendor_median"]) / out["vendor_mad"]
    out["vendor_total_robust_z"] = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
  else:
    out["vendor_count"] = 0
    out["vendor_total_robust_z"] = 0.0

  return out