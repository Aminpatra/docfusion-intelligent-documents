"""
tools/inspect_cord_schema.py
=============================
Prints the actual field names and sample values from one CORD record.
Run this to figure out where the OCR text really lives.

Usage:
    python tools/inspect_cord_schema.py
"""
import json
from datasets import load_dataset

print("Loading 1 CORD record...")
ds = load_dataset("naver-clova-ix/cord-v2", split="train[:1]")
sample = ds[0]

print("\n=== Top-level keys ===")
for k, v in sample.items():
    if k == "image":
        print(f"  {k}: <PIL Image {v.size}>")
    elif isinstance(v, str) and len(v) > 200:
        print(f"  {k}: (string, {len(v)} chars) → first 300 chars:")
        print(f"    {v[:300]}")
    else:
        print(f"  {k}: {repr(v)[:200]}")

print("\n=== ground_truth parsed ===")
gt_raw = sample.get("ground_truth", "")
if isinstance(gt_raw, str):
    gt = json.loads(gt_raw)
else:
    gt = gt_raw

def show(obj, prefix="  ", depth=0):
    if depth > 4:
        print(prefix + "...")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                print(f"{prefix}{k}:")
                show(v, prefix + "  ", depth + 1)
            else:
                print(f"{prefix}{k}: {repr(v)[:120]}")
    elif isinstance(obj, list):
        print(f"{prefix}[list of {len(obj)} items]")
        for i, item in enumerate(obj[:3]):
            print(f"{prefix}  [{i}]:")
            show(item, prefix + "    ", depth + 1)
        if len(obj) > 3:
            print(f"{prefix}  ... ({len(obj) - 3} more)")
    else:
        print(f"{prefix}{repr(obj)[:120]}")

show(gt)
