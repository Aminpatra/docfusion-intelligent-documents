"""
tools/create_training_data.py
==============================
Generates a synthetic training dataset for DocFusion in the correct JSONL format.

Creates data/train/train.jsonl with:
  - 80 genuine receipt records (from CORD data + synthetic SROIE-style)
  - 20 forged/anomalous records (tampered totals, missing fields, outlier values)

This is used as the seed competition data for training. CORD genuine records
are automatically appended by DocFusionSolution.train().

Usage:
    cd c:\\docfusion_project
    python tools/create_training_data.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

random.seed(42)

# ── Realistic receipt data pools ──────────────────────────────────────────────
VENDORS = [
    "GIANT HYPERMARKET", "TESCO STORES", "PARKSON GRAND", "AEON BIG",
    "THE STORE", "MYDIN HYPERMARKET", "ECONSAVE CASH & CARRY", "HERO SUPERMARKET",
    "JAYA GROCER", "COLD STORAGE", "VILLAGE GROCER", "WATSONS STORE",
    "GUARDIAN HEALTH AND BEAUTY", "CARING PHARMACY", "99 SPEEDMART",
    "SUNWAY PYRAMID FOOD COURT", "PAVILION KL RESTAURANT", "KFC MALAYSIA",
    "McDONALD'S MALAYSIA", "STARBUCKS COFFEE", "OldTown WHITE COFFEE",
    "BREAD STORY BAKERY", "SUBWAY RESTAURANT", "PIZZA HUT DELIVERY",
]

DATES = [
    "01/03/2024", "15/03/2024", "22/03/2024", "05/02/2024", "14/02/2024",
    "30/01/2024", "10/04/2024", "25/04/2024", "07/05/2024", "19/05/2024",
    "02/06/2024", "18/06/2024", "03/07/2024", "21/07/2024", "09/08/2024",
    "26/08/2024", "12/09/2024", "30/09/2024", "16/10/2024", "28/10/2024",
]

ITEMS = [
    ("NESCAFE 3IN1 ORIGINAL 20S", 8.90),
    ("DUTCH LADY FULL CREAM MILK 1L", 6.50),
    ("GARDENIA WHITE BREAD 400G", 3.50),
    ("MILO CHOCOLATE MALT 400G", 12.90),
    ("MAGGI TOMYAM KARI 5'S", 5.20),
    ("MAMEE MONSTER NOODLE 5'S", 4.80),
    ("VONO VEGETABLE SOUP 45G", 3.20),
    ("AYAM BRAND SARDINE 155G", 4.10),
    ("CAMPBELL CHICKEN SOUP 300ML", 5.60),
    ("TIGER BISCUIT ASSORTED 700G", 11.50),
    ("PRINGLES ORIGINAL 107G", 7.90),
    ("TWISTIES CHEESE 65G", 2.80),
    ("COCA COLA 1.5L", 4.20),
    ("100PLUS ISOTONIC 1.5L", 3.90),
    ("YAKULT CULTURED DRINK 5's", 7.50),
    ("DETTOL HAND WASH 200ML REFILL", 5.40),
    ("SOFTLAN FABRIC COND 800ML", 8.30),
    ("DYNAMO LIQUID DETERGENT 2KG", 23.90),
    ("DARLIE TOOTHPASTE 225G", 7.20),
    ("COLGATE TOTAL 150G", 9.80),
]


def make_ocr_lines(vendor: str, date: str, items: list[tuple], subtotal: float,
                   tax: float, total: float) -> list[str]:
    lines = [
        vendor,
        f"Receipt No: INV{random.randint(10000, 99999)}",
        f"Date: {date}",
        f"Time: {random.randint(8, 22):02d}:{random.randint(0, 59):02d}",
        "--------------------------------",
        "ITEM                   QTY  AMT",
        "--------------------------------",
    ]
    for name, price in items:
        lines.append(f"{name[:25]:<25} 1  {price:.2f}")
    lines += [
        "--------------------------------",
        f"SUBTOTAL              MYR {subtotal:.2f}",
        f"TAX (6% SST)          MYR {tax:.2f}",
        f"TOTAL                 MYR {total:.2f}",
        f"CASH                  MYR {total + random.choice([0, 0.50, 1, 2, 5]):.2f}",
        "--------------------------------",
        "THANK YOU FOR SHOPPING WITH US!",
        "PLEASE COME AGAIN",
    ]
    return lines


def generate_genuine_record(doc_id: str) -> dict:
    vendor = random.choice(VENDORS)
    date = random.choice(DATES)
    n_items = random.randint(2, 6)
    chosen = random.sample(ITEMS, min(n_items, len(ITEMS)))

    subtotal = round(sum(p for _, p in chosen), 2)
    tax = round(subtotal * 0.06, 2)
    total = round(subtotal + tax, 2)

    ocr_lines = make_ocr_lines(vendor, date, chosen, subtotal, tax, total)

    return {
        "id": doc_id,
        "ocr_lines": ocr_lines,
        "fields": {
            "vendor": vendor,
            "date": date,
            "total": f"{total:.2f}",
        },
        "is_forged": 0,
    }


def generate_forged_record(doc_id: str) -> dict:
    """Generate a forged record with one or more anomalies."""
    forgery_type = random.choice([
        "inflated_total",
        "negative_total",
        "missing_fields",
        "inconsistent_total",
        "suspicious_vendor",
        "extreme_total",
    ])

    vendor = random.choice(VENDORS)
    date = random.choice(DATES)
    n_items = random.randint(2, 4)
    chosen = random.sample(ITEMS, min(n_items, len(ITEMS)))
    subtotal = round(sum(p for _, p in chosen), 2)
    tax = round(subtotal * 0.06, 2)
    real_total = round(subtotal + tax, 2)

    if forgery_type == "inflated_total":
        # Total is much higher than sum of items
        fake_total = round(real_total * random.uniform(3.0, 10.0), 2)
        ocr_lines = make_ocr_lines(vendor, date, chosen, subtotal, tax, fake_total)
        return {
            "id": doc_id,
            "ocr_lines": ocr_lines,
            "fields": {"vendor": vendor, "date": date, "total": f"{fake_total:.2f}"},
            "is_forged": 1,
        }

    elif forgery_type == "negative_total":
        fake_total = round(-abs(real_total), 2)
        ocr_lines = make_ocr_lines(vendor, date, chosen, subtotal, tax, fake_total)
        return {
            "id": doc_id,
            "ocr_lines": ocr_lines,
            "fields": {"vendor": vendor, "date": date, "total": f"{fake_total:.2f}"},
            "is_forged": 1,
        }

    elif forgery_type == "missing_fields":
        # All key fields are missing
        return {
            "id": doc_id,
            "ocr_lines": ["RECEIPT", f"AMOUNT: {real_total:.2f}", "PAID"],
            "fields": {"vendor": None, "date": None, "total": None},
            "is_forged": 1,
        }

    elif forgery_type == "inconsistent_total":
        # Stated total doesn't match items (off by a factor)
        fake_total = round(real_total + random.uniform(50, 200), 2)
        ocr_lines = make_ocr_lines(vendor, date, chosen, subtotal, tax, fake_total)
        # But ocr also shows real total elsewhere
        ocr_lines.insert(4, f"Previous balance: {real_total:.2f}")
        return {
            "id": doc_id,
            "ocr_lines": ocr_lines,
            "fields": {"vendor": vendor, "date": date, "total": f"{fake_total:.2f}"},
            "is_forged": 1,
        }

    elif forgery_type == "suspicious_vendor":
        return {
            "id": doc_id,
            "ocr_lines": [
                "UNKNOWN VENDOR XXX",
                f"Date: {date}",
                f"Total: {real_total:.2f}",
                "TEST TRANSACTION",
            ],
            "fields": {"vendor": "UNKNOWN TEST XXX", "date": date, "total": f"{real_total:.2f}"},
            "is_forged": 1,
        }

    else:  # extreme_total
        fake_total = round(random.uniform(8000, 50000), 2)
        ocr_lines = make_ocr_lines(vendor, date, chosen, subtotal, tax, fake_total)
        return {
            "id": doc_id,
            "ocr_lines": ocr_lines,
            "fields": {"vendor": vendor, "date": date, "total": f"{fake_total:.2f}"},
            "is_forged": 1,
        }


def main():
    out_dir = ROOT / "data" / "train"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []

    # 80 genuine records
    for i in range(80):
        records.append(generate_genuine_record(f"train_genuine_{i:04d}"))

    # 20 forged records
    for i in range(20):
        records.append(generate_forged_record(f"train_forged_{i:04d}"))

    # Shuffle
    random.shuffle(records)

    out_path = out_dir / "train.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_forged = sum(1 for r in records if r["is_forged"] == 1)
    n_genuine = len(records) - n_forged
    print(f"[TrainData] Created {len(records)} records → {out_path}")
    print(f"[TrainData]   Genuine : {n_genuine}")
    print(f"[TrainData]   Forged  : {n_forged}")

    # Also create a minimal test.jsonl for validation
    test_dir = ROOT / "data" / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_records = []
    for i in range(10):
        test_records.append(generate_genuine_record(f"test_{i:04d}"))
    for i in range(5):
        test_records.append(generate_forged_record(f"test_forged_{i:04d}"))
    random.shuffle(test_records)

    # Write test JSONL (no labels — simulates harness format)
    test_out = test_dir / "test.jsonl"
    with open(test_out, "w", encoding="utf-8") as f:
        for rec in test_records:
            out_rec = {k: v for k, v in rec.items() if k != "is_forged"}
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    print(f"[TrainData] Created {len(test_records)} test records → {test_out}")
    print("[TrainData] Done. Run: python tools/train_model.py")


if __name__ == "__main__":
    main()
