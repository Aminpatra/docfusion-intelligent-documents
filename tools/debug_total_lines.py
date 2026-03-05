import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.extraction.sroie_extractor import load_boxes_with_pos, _line_items


def main():
  stem = "X51005268472"  # change this id to debug another one
  box_path = Path(r"C:\docfusion_data\SROIE2019\train\box") / f"{stem}.txt"

  items = load_boxes_with_pos(box_path)

  # build "lines" by grouping items with similar y_mid
  line_tol = 8
  lines = []
  for it in items:
    placed = False
    for ln in lines:
      if abs(ln["y_mid"] - it["y_mid"]) <= line_tol:
        ln["items"].append(it)
        # update average y
        ln["y_mid"] = sum(x["y_mid"] for x in ln["items"]) / len(ln["items"])
        placed = True
        break
    if not placed:
      lines.append({"y_mid": it["y_mid"], "items": [it]})

  # sort lines top->bottom
  lines.sort(key=lambda d: d["y_mid"])
  for ln in lines:
    ln["items"].sort(key=lambda d: d["x_mid"])

  keys = ["rounded total", "grand total", "amount due", "total"]

  for idx, ln in enumerate(lines):
    text_line = " | ".join(x["text"] for x in ln["items"])
    low = text_line.lower()

    if any(k in low for k in keys):
      print("\n=== MATCH LINE ===")
      print(text_line)

      # print next 3 lines too
      for j in range(1, 4):
        if idx + j < len(lines):
          nxt = " | ".join(x["text"] for x in lines[idx + j]["items"])
          print(f"NEXT {j}:", nxt)

      # break

  # if printed == 0:
  #   print("No TOTAL-like lines found.")


if __name__ == "__main__":
  main()