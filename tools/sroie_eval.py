import sys
from pathlib import Path
import json

# import inspect
# from src.extraction import sroie_extractor

# print("Extractor file:", sroie_extractor.__file__)
# print("Has _money_in_next_lines:", hasattr(sroie_extractor, "_money_in_next_lines"))
# print("extract_total starts with:")
# print(inspect.getsource(sroie_extractor.extract_total).splitlines()[0:5])

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.extraction.sroie_extractor import load_boxes_with_pos, extract_vendor, extract_date, extract_total


def read_entities(path):
  raw = Path(path).read_text(encoding="utf-8", errors="ignore").strip()

  # Some SROIE dumps store JSON in .txt
  try:
    obj = json.loads(raw)
    # Common keys
    company = obj.get("company") or obj.get("vendor")
    date = obj.get("date")
    total = obj.get("total")
    return company, date, total
  except Exception:
    pass

  # Fallback: old 4-line format
  lines = raw.splitlines()
  company = lines[0].strip() if len(lines) > 0 else None
  date = lines[1].strip() if len(lines) > 1 else None
  total = lines[3].strip() if len(lines) > 3 else None
  return company, date, total


def norm(s):
  if s is None:
    return None
  s = str(s).strip().lower()
  s = " ".join(s.split())
  return s


def norm_total(s):
  if s is None:
    return None
  t = str(s).strip().lower()
  t = t.replace("rm", "").replace(",", "").strip()
  try:
    return f"{float(t):.2f}"
  except Exception:
    return t


def total_close(a, b, tol=0.05):
  try:
    return abs(float(norm_total(a)) - float(norm_total(b))) <= tol
  except Exception:
    return False


def main():
  base = Path(r"C:\docfusion_data\SROIE2019\train")
  ent_dir = base / "entities"
  box_dir = base / "box"

  ent_files = sorted(ent_dir.glob("*.txt"))
  n = min(200, len(ent_files))

  ok_vendor = 0
  ok_date = 0
  ok_total = 0
  total_count = 0

  examples = []

  for ent_path in ent_files[:n]:
    stem = ent_path.stem
    box_path = box_dir / f"{stem}.txt"

    gt_vendor, gt_date, gt_total = read_entities(ent_path)

    items = load_boxes_with_pos(box_path)
    pred_vendor = extract_vendor(items)
    pred_date = extract_date(items)
    pred_total = extract_total(items)

    total_count += 1

    if norm(pred_vendor) == norm(gt_vendor):
      ok_vendor += 1
    else:
      examples.append(("vendor", stem, gt_vendor, pred_vendor))

    if norm(pred_date) == norm(gt_date):
      ok_date += 1
    else:
      examples.append(("date", stem, gt_date, pred_date))

    if total_close(pred_total, gt_total):
      ok_total += 1
    else:
      examples.append(("total", stem, gt_total, pred_total))

  print("Evaluated:", total_count)
  print("Vendor accuracy:", ok_vendor / total_count)
  print("Date accuracy:  ", ok_date / total_count)
  print("Total accuracy: ", ok_total / total_count)

  print("\nSample errors (first 10):")
  for e in examples[:10]:
    field, stem, gt, pred = e
    print(field, stem, "GT:", gt, "| PRED:", pred)


if __name__ == "__main__":
  main()