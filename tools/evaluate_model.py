"""
tools/evaluate_model.py – evaluates anomaly detection and field extraction.

Usage:
  python tools/evaluate_model.py --smoke_test

  python tools/evaluate_model.py \
      --model_dir  C:/ML/tmp_work/docfusion_model \
      --data_dir   C:/ML/dummy_data/test \
      --labels     C:/ML/dummy_data/test/labels.jsonl \
      --cord_jsonl C:/docfusion_project/data/cord/cord_validation.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import textwrap
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── Metrics ───────────────────────────────────────────────────────────────────

def classification_metrics(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    accuracy  = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "accuracy": round(accuracy, 4), "precision": round(precision, 4),
        "recall": round(recall, 4), "f1": round(f1, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n_total": len(y_true), "n_forged": sum(y_true),
    }


def extraction_metrics(predictions, labels):
    """
    Compare vendor/date/total predictions against ground-truth labels.
    Labels may have fields at top-level OR nested under 'fields' dict.
    If labels have no field ground truth at all, returns None.
    """
    label_map = {r["id"]: r for r in labels}

    # Check if ANY label has field ground truth
    has_field_gt = any(
        r.get("vendor") or r.get("date") or r.get("total") or r.get("fields")
        for r in labels
    )
    if not has_field_gt:
        return None  # labels.jsonl has no field ground truth — skip

    results = defaultdict(lambda: {"correct": 0, "total": 0})

    for pred in predictions:
        doc_id = pred["id"]
        if doc_id not in label_map:
            continue
        gt = label_map[doc_id]

        for field in ("vendor", "date", "total"):
            pred_val = str(pred.get(field) or "").strip().lower()
            gt_val = (
                str(gt.get(field) or "").strip().lower() or
                str((gt.get("fields") or {}).get(field) or "").strip().lower()
            )

            if not gt_val:
                continue  # no ground truth for this field — skip

            results[field]["total"] += 1
            if pred_val and pred_val == gt_val:
                results[field]["correct"] += 1

    out = {}
    for field in ("vendor", "date", "total"):
        total   = results[field]["total"]
        correct = results[field]["correct"]
        out[field] = {
            "exact_match": round(correct / max(total, 1), 4),
            "correct": correct, "total": total,
        }
    return out


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(clf_metrics, ext_metrics, cord_stats=None):
    lines = []
    lines.append("=" * 60)
    lines.append("  DocFusion Evaluation Report")
    lines.append("=" * 60)

    if clf_metrics:
        lines.append("\n── Anomaly Detection (is_forged) ──────────────────────")
        lines.append(f"  Accuracy : {clf_metrics['accuracy']:.1%}")
        lines.append(f"  Precision: {clf_metrics['precision']:.1%}  "
                     "(of predicted forged, how many actually were)")
        lines.append(f"  Recall   : {clf_metrics['recall']:.1%}  "
                     "(of actual forged, how many did we catch)")
        lines.append(f"  F1 Score : {clf_metrics['f1']:.1%}")
        lines.append(f"\n  Confusion matrix:")
        lines.append(f"                  Predicted")
        lines.append(f"                  Genuine  Forged")
        lines.append(f"  Actual Genuine  {clf_metrics['tn']:6d}  {clf_metrics['fp']:6d}")
        lines.append(f"  Actual Forged   {clf_metrics['fn']:6d}  {clf_metrics['tp']:6d}")
        lines.append(f"\n  Total: {clf_metrics['n_total']} "
                     f"({clf_metrics['n_forged']} forged, "
                     f"{clf_metrics['n_total']-clf_metrics['n_forged']} genuine)")

        lines.append("\n── Interpretation ──────────────────────────────────────")
        f1 = clf_metrics["f1"]
        if f1 >= 0.85:
            verdict = "Excellent — strong forgery detection."
        elif f1 >= 0.70:
            verdict = "Good — solid baseline, room to improve."
        elif f1 >= 0.50:
            verdict = "Fair — catching some forgeries but missing many."
        else:
            verdict = "Poor — model is struggling."
        lines.append(f"  F1={f1:.1%}: {verdict}")

        if clf_metrics["fn"] > 0 and clf_metrics["recall"] < 0.5:
            lines.append("  ⚠  Low recall: forged docs slipping through.")
            lines.append("     → The threshold was already auto-tuned; need more forged training data.")
        if clf_metrics["fp"] > clf_metrics["fn"]:
            lines.append("  ⚠  High false-positive rate: genuine docs flagged as forged.")
            lines.append("     → Raise the threshold or add more genuine training data.")
        if clf_metrics["tp"] == 0:
            lines.append("  ⚠  Zero true positives — model predicts all genuine.")
            lines.append("     → Training data has very few forged examples.")
            lines.append("       Download Find-It-Again dataset for real forgery labels.")
    else:
        lines.append("\n── Anomaly Detection: SKIPPED (no labels) ─────────────")

    if ext_metrics is None:
        lines.append("\n── Field Extraction: SKIPPED ───────────────────────────")
        lines.append("  (labels.jsonl has no vendor/date/total ground truth)")
        lines.append("  Field extraction uses pre-extracted fields from test.jsonl.")
        lines.append("  To evaluate extraction, use Find-It-Again or SROIE data")
        lines.append("  which include field-level ground truth.")
    else:
        lines.append("\n── Field Extraction (exact match) ──────────────────────")
        for field in ("vendor", "date", "total"):
            m = ext_metrics.get(field, {})
            rate = m.get("exact_match", 0)
            bar  = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
            lines.append(f"  {field:6s}: {bar} {rate:.1%}  "
                         f"({m.get('correct',0)}/{m.get('total',0)})")
        avg = sum(ext_metrics[f]["exact_match"] for f in ("vendor","date","total")) / 3
        lines.append(f"\n  Average extraction accuracy: {avg:.1%}")

    if cord_stats:
        lines.append("\n── CORD Robustness (false-positive rate) ───────────────")
        lines.append(f"  CORD samples evaluated : {cord_stats['n_total']}")
        lines.append(f"  Wrongly flagged as forged: {cord_stats['n_flagged']} "
                     f"({cord_stats['fpr']:.1%})")
        if cord_stats["fpr"] > 0.15:
            lines.append("  ⚠  FPR > 15%: over-triggering on diverse layouts.")
            lines.append("     → Add more CORD samples to training data.")
        else:
            lines.append("  ✓  FPR is healthy.")

    lines.append("\n" + "=" * 60)
    report = "\n".join(lines)
    print(report)
    return report


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(model_dir, data_dir, labels_path=None, cord_jsonl=None):
    from src.pipeline import load_jsonl, run_pipeline

    model_dir = Path(model_dir)
    data_dir  = Path(data_dir)

    test_jsonl = data_dir / "test.jsonl"
    if not test_jsonl.exists():
        candidates = list(data_dir.glob("*.jsonl"))
        if not candidates:
            print(f"ERROR: no test.jsonl in {data_dir}", file=sys.stderr)
            sys.exit(1)
        test_jsonl = candidates[0]

    print(f"[eval] Model : {model_dir}")
    print(f"[eval] Data  : {data_dir}")

    records = load_jsonl(test_jsonl)
    print(f"[eval] Loaded {len(records)} test records")

    images_dir = data_dir / "images"
    print("[eval] Running pipeline…")
    predictions = run_pipeline(model_dir, records, images_dir)
    print(f"[eval] Got {len(predictions)} predictions")

    # Load labels
    labels = []
    if labels_path and Path(labels_path).exists():
        labels = load_jsonl(labels_path)
    elif (data_dir / "labels.jsonl").exists():
        labels = load_jsonl(data_dir / "labels.jsonl")

    # Classification metrics
    clf_metrics = None
    if labels:
        label_map = {}
        for r in labels:
            if "is_forged" in r:
                label_map[r["id"]] = int(r["is_forged"])
            elif "label" in r and isinstance(r["label"], dict):
                label_map[r["id"]] = int(r["label"].get("is_forged", 0))

        y_true, y_pred = [], []
        for pred in predictions:
            if pred["id"] in label_map:
                y_true.append(label_map[pred["id"]])
                y_pred.append(pred["is_forged"])

        if y_true:
            clf_metrics = classification_metrics(y_true, y_pred)

    # Extraction metrics
    ext_metrics = extraction_metrics(predictions, labels) if labels else None

    # CORD false-positive rate
    cord_stats = None
    if cord_jsonl and Path(cord_jsonl).exists():
        print(f"[eval] Checking CORD FPR…")
        cord_records = load_jsonl(cord_jsonl)[:500]
        cord_preds = run_pipeline(
            model_dir, cord_records,
            Path(cord_jsonl).parent / "images"
        )
        n_flagged = sum(1 for p in cord_preds if p["is_forged"] == 1)
        cord_stats = {
            "n_total": len(cord_preds),
            "n_flagged": n_flagged,
            "fpr": n_flagged / max(len(cord_preds), 1),
        }

    report = print_report(clf_metrics, ext_metrics, cord_stats)

    report_path = model_dir / "evaluation_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"[eval] Report saved → {report_path}")
    return clf_metrics, ext_metrics


# ── Smoke test ────────────────────────────────────────────────────────────────

def smoke_test():
    dummy_train = ROOT / "dummy_data" / "train"
    dummy_test  = ROOT / "dummy_data" / "test"

    if not dummy_train.exists():
        print("ERROR: dummy_data/ not found. Run from C:\\ML or set ROOT correctly.",
              file=sys.stderr)
        sys.exit(1)

    from solution import DocFusionSolution

    with tempfile.TemporaryDirectory() as tmp:
        sol = DocFusionSolution()
        print("[smoke] Training…")
        model_dir = sol.train(str(dummy_train), tmp)
        print("[smoke] Predicting…")
        out_path = Path(tmp) / "predictions.jsonl"
        sol.predict(model_dir, str(dummy_test), str(out_path))
        evaluate(
            model_dir   = model_dir,
            data_dir    = dummy_test,
            labels_path = dummy_test / "labels.jsonl",
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DocFusion model accuracy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_dir")
    parser.add_argument("--data_dir")
    parser.add_argument("--labels",     help="Path to labels.jsonl")
    parser.add_argument("--cord_jsonl", help="CORD JSONL for FP-rate check")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test()
        return

    if not args.model_dir or not args.data_dir:
        parser.print_help()
        sys.exit(1)

    evaluate(
        model_dir   = args.model_dir,
        data_dir    = args.data_dir,
        labels_path = args.labels,
        cord_jsonl  = args.cord_jsonl,
    )


if __name__ == "__main__":
    main()