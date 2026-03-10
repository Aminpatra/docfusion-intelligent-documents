"""
tools/train_model.py
====================
Standalone training script for DocFusion.

Trains the anomaly detector using:
  - data/train/train.jsonl  (synthetic competition samples with forgery labels)
  - data/cord/              (CORD genuine receipt augmentation, auto-merged)

Output model is saved under:
  outputs/model/docfusion_model/anomaly/

Usage:
    cd c:\\docfusion_project
    python tools/create_training_data.py   # generates data/train/train.jsonl
    python tools/train_model.py            # trains the model
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("train_model")


def main():
    # ── Locate training data ───────────────────────────────────────────────────
    candidates = [
        ROOT / "data" / "train",
        ROOT / "ML" / "dummy_data" / "train",
        ROOT / "dummy_data" / "train",
    ]
    train_dir = None
    for c in candidates:
        if (c / "train.jsonl").exists():
            train_dir = c
            break

    if train_dir is None:
        logger.error(
            "No train.jsonl found. Run:\n"
            "  python tools/create_training_data.py\n"
            "Tried paths: %s",
            [str(c) for c in candidates],
        )
        sys.exit(1)

    work_dir = ROOT / "outputs" / "model"
    logger.info("Training dir : %s", train_dir)
    logger.info("Work dir     : %s", work_dir)

    # ── Run training ───────────────────────────────────────────────────────────
    from solution import DocFusionSolution
    sol = DocFusionSolution()

    model_dir = sol.train(str(train_dir), str(work_dir))
    logger.info("Model saved to: %s", model_dir)

    # ── Artifact summary ───────────────────────────────────────────────────────
    artifacts = sorted(Path(model_dir).rglob("*.joblib"))
    logger.info("Model artifacts (%d files):", len(artifacts))
    for a in artifacts:
        size_kb = a.stat().st_size // 1024
        logger.info("  %-45s  %d KB", str(a.relative_to(ROOT)), size_kb)

    logger.info("Training complete! Launch UI with:  streamlit run ui/app.py")


if __name__ == "__main__":
    main()
