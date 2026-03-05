import json
from pathlib import Path
from src.extraction.sroie_extractor import (
  load_boxes,
  extract_total,
  _money_on_same_line_right,
  _money_in_next_lines,
  _line_items
)

stem = "X51005268400"  # change to X51005301661

root = Path("C:/docfusion_data/SROIE2019/test")  # adjust if needed
img_dir = root / "img"
box_dir = root / "box"

items = load_boxes(box_dir / f"{stem}.txt")
print("Extract_total:", extract_total(items))

# show lines that contain 'total'
for it in items:
  if "total" in it["text"].lower():
    lt = " ".join(x["text"].lower() for x in _line_items(items, it["y_mid"]))
    print("\nANCHOR:", it["text"])
    print("LINE:", lt)
    print("same_line_right:", _money_on_same_line_right(items, it))
    print("next_lines:", _money_in_next_lines(items, it, max_lines=4))