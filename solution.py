import os
import sys
import json

import pandas as pd
from src.extraction.sroie_extractor import extract_vendor, extract_date, extract_total

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

    # Step 1: ensure every record has usable structured fields
    # If fields are missing, we will try to extract them later when OCR/boxes are available.
    for row in data:
      fields = row.get("fields", {})
      row["fields"] = {
        "vendor": fields.get("vendor"),
        "date": fields.get("date"),
        "total": fields.get("total")
      }

    # Step 2: normalize into dataframe
    df = pd.json_normalize(data)

    # Step 3: build lightweight training artifacts
    vstats = vendor_stats(df)

    model_dir = os.path.join(work_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    vstats.to_csv(os.path.join(model_dir, "vendor_stats.csv"), index=False)

    meta = {
      "version": "docfusion_rules_v2",
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

    # Step 1: fill missing fields before anomaly detection
    for row in data:
      fields = row.get("fields", {})

      vendor = fields.get("vendor")
      date = fields.get("date")
      total = fields.get("total")

      # Fallback extraction if fields are missing
      if vendor is None or date is None or total is None:
        items = row.get("ocr_boxes", [])

        if items:
          if vendor is None:
            vendor = extract_vendor(items)
          if date is None:
            date = extract_date(items)
          if total is None:
            total = extract_total(items)

        row["fields"] = {
          "vendor": vendor,
          "date": date,
          "total": total
        }

    # Step 2: build dataframe after extraction/fallback
    df = pd.json_normalize(data)

    # Step 3: load training artifacts
    vstats_path = os.path.join(model_dir, "vendor_stats.csv")
    vstats = pd.read_csv(vstats_path)

    # Step 4: anomaly detection
    preds, scores, reasons = predict_rules(df, vstats, z_thresh=3.5)

    # Step 5: debug output for UI
    debug_path = os.path.join(os.path.dirname(out_path), "debug_predictions.jsonl")
    debug_rows = []

    for row, pred, sc, rs in zip(data, preds, scores, reasons):
      debug_rows.append({
        "id": row.get("id"),
        "score": float(sc),
        "reasons": rs
      })

    write_jsonl(debug_path, debug_rows)

    # Step 6: official harness output
    outputs = []
    for row, pred in zip(data, preds):
      fields = row.get("fields", {})

      outputs.append({
        "id": row.get("id"),
        "vendor": fields.get("vendor"),
        "date": fields.get("date"),
        "total": fields.get("total"),
        "is_forged": int(bool(pred))
      })

    out_dir = os.path.dirname(out_path)
    if out_dir:
      os.makedirs(out_dir, exist_ok=True)

    write_jsonl(out_path, outputs)