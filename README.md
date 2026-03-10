# DocFusion — Intelligent Document Processing Pipeline

> End-to-end OCR extraction, structured field parsing, and anomaly detection for scanned receipts and invoices.

---

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [Challenge Levels](#challenge-levels)
5. [Architecture](#architecture)
6. [Running the Web UI](#running-the-web-ui)
7. [Harness Integration](#harness-integration)
8. [Docker](#docker)
9. [Design Decisions](#design-decisions)

---

## Overview

DocFusion processes scanned business documents (receipts and invoices) from three real-world datasets:
- **SROIE** – English receipts with bounding-box OCR annotations
- **Find-It-Again (L3i)** – SROIE receipts with realistic forgeries (tampered text, copy-paste attacks)
- **CORD** – Diverse receipt layouts for robustness testing

The pipeline automatically extracts `vendor`, `date`, and `total` fields, and predicts whether each document is genuine or forged (`is_forged`).

---

## Project Structure

```
docfusion/
├── solution.py                  # Harness entry point (DocFusionSolution class)
├── requirements.txt
├── Dockerfile
│
├── src/
│   ├── pipeline.py              # Main orchestration: extraction + anomaly
│   ├── extraction/
│   │   ├── extractors.py        # Unified extractor (routes to correct sub-extractor)
│   │   └── sroie_extractor.py   # Box-based SROIE extractor (vendor/date/total)
│   ├── anomaly/
│   │   ├── train_detector.py    # Trains GradientBoosting + TF-IDF model
│   │   ├── predict_detector.py  # Loads model and runs inference
│   │   ├── rules.py             # Rule-based anomaly scoring (Z-score, outliers)
│   │   ├── stats.py             # Vendor-level statistics (median, MAD)
│   │   ├── features.py          # Feature engineering utilities
│   │   └── detector.py          # RandomForest wrapper (legacy)
│   ├── ocr/
│   │   ├── ocr_engine.py        # PaddleOCR → Tesseract fallback
│   │   ├── paddle_ocr.py
│   │   └── tesseract_ocr.py
│   └── datasets/
│       ├── findit_loader.py     # Find-It-Again dataset loader
│       └── parse_findit_annotations.py
│
├── ui/
│   └── app.py                   # Streamlit web dashboard (Level 3B)
│
├── notebooks/
│   ├── 01_eda.ipynb             # Level 1: EDA on SROIE
│   ├── 02_baseline_report.ipynb # Level 2: Extraction baseline
│   ├── 03_find_it_again_eda.ipynb
│   ├── 04_find_it_again_baseline.ipynb
│   └── eda_day2.ipynb
│
└── tools/                       # Debugging and evaluation scripts
    ├── sroie_eval.py
    ├── sorie_draw_boxes.py
    └── ...
```

---

## Quick Start

### 1. Install dependencies

```bash
# Python 3.13+ required
pip install -r requirements.txt

# System: install Tesseract OCR (fallback OCR engine)
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr
# macOS:
brew install tesseract
```

### 2. Validate locally with the harness checker

```bash
# From the ML/ directory (where check_submission.py lives):
python3 check_submission.py --submission /path/to/this/folder
```

### 3. Run the Web UI

```bash
streamlit run ui/app.py
# Opens at http://localhost:8501
```

---

## Challenge Levels

### Level 1 – EDA
Notebooks in `notebooks/` explore:
- Document image samples and OCR quality
- Price and vendor distributions
- Layout patterns across SROIE, CORD, Find-It-Again

### Level 2 – Field Extraction
Two extraction strategies:

| Strategy | Input | When used |
|----------|-------|-----------|
| **Box-based** (`sroie_extractor.py`) | SROIE `.txt` bounding-box files | Training with OCR annotation files |
| **Text-line** (`extractors.py`) | `ocr_lines` from JSONL | Harness inference (most common) |

The text-line extractor uses:
- **Vendor**: Top-of-receipt heuristic scoring (business keyword boost, address/metadata penalty)
- **Date**: Multi-pattern regex across 5 date formats
- **Total**: Keyword-anchored search (20+ keywords, bottom-up, inclusive-GST aware)

### Level 3A – Anomaly Detection
Two-layer approach:

**Layer 1 – ML Model** (trained in `train()`):
- TF-IDF (uni+bigrams, 4000 features, sublinear TF)
- Manual features: text statistics, amount counts, date counts, missing-field flags, log-scaled total
- `GradientBoostingClassifier` (200 trees, depth 4, 0.05 LR) — falls back to `LogisticRegression` when training data is tiny

**Layer 2 – Rule Engine** (`rules.py`):
- Vendor-level robust Z-score (MAD-based)
- Global fallback Z-score for unseen vendors
- Rare-vendor + high-total flag
- Missing total → instant flag

### Level 3B – Web UI
See [Running the Web UI](#running-the-web-ui).

### Level 4 – Harness Integration
See [Harness Integration](#harness-integration).

---

## Architecture

```
                ┌───────────────────────────────────────┐
 JSONL record   │              solution.py               │
 (id, ocr_lines,│  DocFusionSolution.train() / .predict()│
  is_forged)    └──────────────┬───────────────┬─────────┘
                               │               │
                    ┌──────────▼──┐     ┌──────▼──────────┐
                    │ train_       │     │  pipeline.py     │
                    │ detector.py │     │  run_pipeline()  │
                    │             │     │                  │
                    │ TF-IDF +    │     │  ┌────────────┐  │
                    │ GBM model   │     │  │ extractors │  │
                    └─────────────┘     │  │ .py        │  │
                                        │  └─────┬──────┘  │
                                        │        │          │
                                        │  ┌─────▼──────┐  │
                                        │  │ predict_   │  │
                                        │  │ detector   │  │
                                        │  └────────────┘  │
                                        └──────────────────┘
                                                │
                                        predictions.jsonl
```

---

## Running the Web UI

```bash
streamlit run ui/app.py
```

Features:
- Upload any receipt/invoice image (PNG, JPG, TIFF)
- Live OCR via PaddleOCR or Tesseract
- Extracted fields displayed with ✅/❌ indicators
- Anomaly report with specific violation reasons
- Visual overlay — green boxes (normal), red boxes (suspicious)
- JSON export of results

---

## Harness Integration

The `DocFusionSolution` class in `solution.py` implements the required interface:

```python
class DocFusionSolution:
    def train(self, train_dir: str, work_dir: str) -> str: ...
    def predict(self, model_dir: str, data_dir: str, out_path: str) -> None: ...
```

**Input format** (`train.jsonl` / `test.jsonl`):
```json
{"id": "t001", "ocr_lines": ["VENDOR NAME", "Date: 01/01/2024", "Total: 15.00"], "is_forged": 0}
```

**Output format** (`predictions.jsonl`):
```json
{"id": "t001", "vendor": "VENDOR NAME", "date": "01/01/2024", "total": "15.00", "is_forged": 0}
```

---

## Docker

### Build
```bash
docker build -t docfusion .
```

### Run the Web UI
```bash
docker run -p 8501:8501 docfusion
# Visit http://localhost:8501
```

### Run harness inference inside container
```bash
docker run --rm \
  -v /path/to/data:/data \
  -v /path/to/output:/output \
  docfusion \
  python -c "
from solution import DocFusionSolution
s = DocFusionSolution()
m = s.train('/data/train', '/output/model')
s.predict(m, '/data/test', '/output/predictions.jsonl')
"
```

---

## Design Decisions

**Why GradientBoosting over RandomForest?**
GBM is more sensitive to weak forgery signals in text (subtle keyword shifts, unusual amount distributions). With `subsample=0.8` it's robust to small training sets.

**Why TF-IDF + manual features together?**
TF-IDF captures vocabulary patterns (a forged receipt often reuses unusual phrasing), while manual features (missing fields, outlier totals) are direct, interpretable signals that TF-IDF alone would miss.

**Why two extraction paths?**
The SROIE bounding-box extractor is highly accurate but requires OCR annotation files. The text-line extractor is more portable and works directly from the `ocr_lines` field in the JSONL, which is all the harness provides.

**Forgery threshold = 0.5**
Balanced precision/recall. In production, adjusting this threshold (lower for recall-first fraud catching, higher for precision-first alerting) would depend on the business cost of false positives vs. false negatives.