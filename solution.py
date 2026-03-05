import os
import sys
import json

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from src.anomaly.stats import vendor_stats
from src.anomaly.rules import predict_rules


def read_jsonl(path):
  rows = []
  with open(path, "r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if line:
        rows.append(json.loads(line))
  return rows


def write_jsonl(path, records):
  with open(path, "w", encoding="utf-8") as f:
    for r in records:
      f.write(json.dumps(r, ensure_ascii=False) + "\n")


class DocFusionSolution:

  def train(self, train_dir: str, work_dir: str) -> str:
    train_path = os.path.join(train_dir, "train.jsonl")
    data = read_jsonl(train_path)
    df = pd.json_normalize(data)

    # Build vendor statistics from training data
    vstats = vendor_stats(df)

    model_dir = os.path.join(work_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    # Save vendor stats as a lightweight artifact
    vstats_path = os.path.join(model_dir, "vendor_stats.csv")
    vstats.to_csv(vstats_path, index=False)

    # Save a small meta file (nice for debugging)
    meta = {
      "version": "rules_vendor_stats_v1",
      "z_thresh": 3.5,
      "rare_vendor_count": 2,
      "rare_vendor_high_total": 300
    }
    with open(os.path.join(model_dir, "meta.json"), "w", encoding="utf-8") as f:
      json.dump(meta, f, ensure_ascii=False, indent=2)

    return model_dir


  def predict(self, model_dir: str, data_dir: str, out_path: str) -> None:
    test_path = os.path.join(data_dir, "test.jsonl")
    data = read_jsonl(test_path)
    df = pd.json_normalize(data)

    # Load vendor stats
    vstats_path = os.path.join(model_dir, "vendor_stats.csv")
    vstats = pd.read_csv(vstats_path)

    # Rules-based prediction
    preds, reasons = predict_rules(df, vstats, z_thresh=3.5)

    # Write output in required JSONL format
    outputs = []
    for row, pred in zip(data, preds):
      fields = row.get("fields", {})
      outputs.append({
        "id": row.get("id"),
        "vendor": fields.get("vendor"),
        "date": fields.get("date"),
        "total": fields.get("total"),
        "is_forged": int(pred)
      })

    # Ensure output directory exists
    out_dir = os.path.dirname(out_path)
    if out_dir:
      os.makedirs(out_dir, exist_ok=True)

    write_jsonl(out_path, outputs)