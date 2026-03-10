"""
tools/prepare_cord.py
=====================
Downloads CORD-v2 from HuggingFace and converts it to DocFusion JSONL format.

Fix: ground_truth is stored as a JSON *string* in CORD-v2 — must json.loads() it.

Usage:
    python tools/prepare_cord.py
    python tools/prepare_cord.py --out_dir data/cord --splits train validation test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_gt(sample: dict) -> dict:
    """
    Return the first gt_parse dict from a CORD sample.
    Handles both cases:
      - ground_truth is already a dict  (older versions)
      - ground_truth is a JSON string   (cord-v2 Parquet release)
    """
    gt_raw = sample.get("ground_truth", {})

    # Deserialise if it came back as a string
    if isinstance(gt_raw, str):
        try:
            gt_raw = json.loads(gt_raw)
        except json.JSONDecodeError:
            return {}

    parses = gt_raw.get("gt_parses", [])
    if not parses:
        return {}

    parse = parses[0]
    # Each element of gt_parses can itself be a string in some versions
    if isinstance(parse, str):
        try:
            parse = json.loads(parse)
        except json.JSONDecodeError:
            return {}

    return parse


def _extract_ocr_lines(sample: dict) -> list[str]:
    parse = _parse_gt(sample)
    if not parse:
        return []

    lines: list[str] = []

    for item in parse.get("menu", []):
        for field in ("nm", "cnt", "price", "unitprice", "discountprice"):
            val = item.get(field)
            if val:
                lines.append(str(val).strip())
        for sub in item.get("sub_nm", []):
            if sub:
                lines.append(str(sub).strip())
        for sub in item.get("sub_price", []):
            if sub:
                lines.append(str(sub).strip())

    sub_total = parse.get("sub_total", {})
    for field in ("subtotal_price", "discount_price", "service_price",
                  "othersvc_price", "tax_price", "etc"):
        val = sub_total.get(field)
        if val:
            lines.append(str(val).strip())

    total_block = parse.get("total", {})
    for field in ("total_price", "cashprice", "changeprice",
                  "creditcardprice", "emoneyprice"):
        val = total_block.get(field)
        if val:
            lines.append(str(val).strip())

    return [l for l in lines if l]


def _extract_fields(sample: dict) -> dict:
    parse = _parse_gt(sample)
    total = None
    if parse:
        total = parse.get("total", {}).get("total_price")
    return {
        "vendor": None,
        "date":   None,
        "total":  str(total).strip() if total else None,
    }


def convert_split(split: str, out_path: Path, verbose: bool = True) -> int:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: install the 'datasets' package:\n  pip install datasets",
              file=sys.stderr)
        sys.exit(1)

    print(f"[CORD] Loading split '{split}' from HuggingFace…")
    # trust_remote_code removed — not supported in newer huggingface_hub
    ds = load_dataset("naver-clova-ix/cord-v2", split=split)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for i, sample in enumerate(ds):
            try:
                ocr_lines = _extract_ocr_lines(sample)
                fields    = _extract_fields(sample)
            except Exception as e:
                print(f"  [CORD] WARNING: skipping sample {i} ({e})")
                continue

            record = {
                "id":        f"cord_{split}_{i:05d}",
                "ocr_lines": ocr_lines,
                "fields":    fields,
                "is_forged": 0,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    if verbose:
        print(f"[CORD] Wrote {count} records → {out_path}")

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Convert CORD-v2 dataset to DocFusion JSONL."
    )
    parser.add_argument("--out_dir", default="data/cord",
                        help="Output directory (default: data/cord)")
    parser.add_argument("--splits", nargs="+",
                        default=["train", "validation", "test"],
                        help="Which splits to download")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    total = 0
    for split in args.splits:
        out_path = out_dir / f"cord_{split}.jsonl"
        total += convert_split(split, out_path)

    print(f"\n[CORD] Done. Total records: {total}")
    print(f"[CORD] Files saved to: {out_dir.resolve()}")
    print("\nNext: retrain to pick up CORD data automatically.")
    print("  python check_submission.py --submission ./")


if __name__ == "__main__":
    main()