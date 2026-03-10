"""
tools/prepare_cord.py
=====================
Downloads CORD-v2 from HuggingFace and converts it to DocFusion JSONL format.

Schema (confirmed from inspection):
  - ground_truth: JSON string with key "gt_parse" (singular, NOT "gt_parses")
    - gt_parse.menu[].{nm, cnt, price, ...}
    - gt_parse.sub_total.{subtotal_price, tax_price, ...}
    - gt_parse.total.total_price
  - valid_line: list of text-line objects
    - valid_line[].words[].text  ← actual OCR word tokens

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
    Return the gt_parse dict from a CORD sample.
    Key is 'gt_parse' (singular) — not 'gt_parses'.
    """
    gt_raw = sample.get("ground_truth", {})
    if isinstance(gt_raw, str):
        try:
            gt_raw = json.loads(gt_raw)
        except json.JSONDecodeError:
            return {}
    # CORD uses 'gt_parse' (singular)
    return gt_raw.get("gt_parse", {})


def _extract_ocr_from_valid_lines(sample: dict) -> list[str]:
    """
    Extract OCR text from valid_line[].words[].text.
    Groups each valid_line's words into a single text line.
    """
    lines: list[str] = []
    for line_obj in sample.get("valid_line", []) or []:
        words = []
        for word in line_obj.get("words", []) or []:
            text = str(word.get("text", "")).strip()
            if text:
                words.append(text)
        if words:
            lines.append(" ".join(words))
    return lines


def _extract_ocr_lines(sample: dict) -> list[str]:
    """
    Extract OCR text lines from a CORD sample.
    Uses valid_line word tokens (primary), then gt_parse structured values.
    """
    lines: list[str] = []

    # ── Source 1: valid_line word tokens (actual OCR) ─────────────────────────
    lines.extend(_extract_ocr_from_valid_lines(sample))

    # ── Source 2: gt_parse structured values ──────────────────────────────────
    parse = _parse_gt(sample)
    if parse:
        for item in parse.get("menu", []) or []:
            for field in ("nm", "cnt", "price", "unitprice", "discountprice"):
                val = item.get(field)
                if val:
                    lines.append(str(val).strip())

        sub_total = parse.get("sub_total", {}) or {}
        for field in ("subtotal_price", "discount_price", "service_price",
                      "othersvc_price", "tax_price", "etc"):
            val = sub_total.get(field)
            if val:
                lines.append(str(val).strip())

        total_block = parse.get("total", {}) or {}
        for field in ("total_price", "cashprice", "changeprice",
                      "creditcardprice", "emoneyprice"):
            val = total_block.get(field)
            if val:
                lines.append(str(val).strip())

    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for line in lines:
        if line and line not in seen:
            seen.add(line)
            result.append(line)

    return result


def _extract_fields(sample: dict) -> dict:
    parse = _parse_gt(sample)
    total = None
    if parse:
        total = (parse.get("total") or {}).get("total_price")
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

    print(f"[CORD] Loading split '{split}' from HuggingFace...")
    ds = load_dataset("naver-clova-ix/cord-v2", split=split)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    empty_count = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for i, sample in enumerate(ds):
            try:
                ocr_lines = _extract_ocr_lines(sample)
                fields    = _extract_fields(sample)
            except Exception as e:
                print(f"  [CORD] WARNING: skipping sample {i} ({e})")
                continue

            if not ocr_lines:
                empty_count += 1

            record = {
                "id":        f"cord_{split}_{i:05d}",
                "ocr_lines": ocr_lines,
                "fields":    fields,
                "is_forged": 0,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    if verbose:
        filled = count - empty_count
        pct = 100 * filled // max(count, 1)
        print(f"[CORD] Wrote {count} records -> {out_path}")
        print(f"[CORD]   With OCR text : {filled}/{count} ({pct}%)")
        if empty_count:
            print(f"[CORD]   Empty records : {empty_count}")

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
    print("\nNext: retrain with enriched CORD data.")
    print("  python tools/train_model.py")


if __name__ == "__main__":
    main()