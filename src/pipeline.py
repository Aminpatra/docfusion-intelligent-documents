from pathlib import Path
import json
import pandas as pd

from src.extraction.extractors import extract_fields
from src.anomaly.predict_detector import load_detector, predict_forgery


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def prepare_dataframe(records, images_dir):
    rows = []

    for rec in records:
        doc_id = rec["id"]

        ocr_lines = rec.get("ocr_lines", [])
        if not isinstance(ocr_lines, list):
            ocr_lines = []

        

        text = "\n".join(str(x) for x in ocr_lines if x is not None)

        image_path = Path(images_dir) / f"{doc_id}.png"

        rows.append({
            "id": doc_id,
            "image_path": str(image_path),
            "lines": ocr_lines,
            "text": text,
        })

    return pd.DataFrame(rows)


def run_pipeline(model_dir, records, images_dir):
    df = prepare_dataframe(records, images_dir)

    # anomaly model
    anomaly_model, tfidf, manual_cols = load_detector(Path(model_dir) / "anomaly")

    forged_prob = predict_forgery(df, anomaly_model, tfidf, manual_cols)
    df["forged_prob"] = forged_prob
    df["is_forged"] = (df["forged_prob"] >= 0.5).astype(int)

    predictions = []

    for _, row in df.iterrows():
        fields = extract_fields(
        doc_id=row["id"],
        lines=row["lines"],
        image_path=row["image_path"]
    )

        predictions.append({
            "id": row["id"],
            "vendor": fields.get("vendor"),
            "date": fields.get("date"),
            "total": fields.get("total"),
            "is_forged": int(row["is_forged"]),
        })

    return predictions