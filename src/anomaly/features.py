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