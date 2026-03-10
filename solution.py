import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import shutil
import pandas as pd

from src.pipeline import load_jsonl, run_pipeline
from src.anomaly.train_detector import train_detector


class DocFusionSolution:
    def train(self, train_dir: str, work_dir: str) -> str:
        train_dir = Path(train_dir)
        work_dir = Path(work_dir)
        model_dir = work_dir / "docfusion_model"
        anomaly_dir = model_dir / "anomaly"

        anomaly_dir.mkdir(parents=True, exist_ok=True)

        train_jsonl = train_dir / "train.jsonl"
        images_dir = train_dir / "images"

        records = load_jsonl(train_jsonl)

        rows = []
        for rec in records:
            doc_id = rec["id"]
            ocr_lines = rec.get("ocr_lines", [])
            if not isinstance(ocr_lines, list):
                ocr_lines = []

            text = "\n".join(str(x) for x in ocr_lines if x is not None)

            rows.append({
                "id": doc_id,
                "lines": ocr_lines,
                "text": text,
                "is_forged": int(rec.get("is_forged", 0)),
            })

        train_df = pd.DataFrame(rows)

        train_detector(train_df, anomaly_dir)

        return str(model_dir)

    def predict(self, model_dir: str, data_dir: str, out_path: str) -> None:
        model_dir = Path(model_dir)
        data_dir = Path(data_dir)
        out_path = Path(out_path)

        test_jsonl = data_dir / "test.jsonl"
        images_dir = data_dir / "images"

        records = load_jsonl(test_jsonl)

        predictions = run_pipeline(model_dir, records, images_dir)

        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            for row in predictions:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")