import ast
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image


def safe_literal_eval(x):
  if pd.isna(x):
    return None

  x = str(x).strip()

  if x in ["", "0", "0.0", "None", "nan"]:
    return None

  try:
    return ast.literal_eval(x)
  except Exception:
    return None


def ensure_dir(path):
  Path(path).mkdir(parents=True, exist_ok=True)


def xywh_to_yolo(x, y, w, h, img_w, img_h):
  x_center = (x + w / 2) / img_w
  y_center = (y + h / 2) / img_h
  w_norm = w / img_w
  h_norm = h / img_h
  return x_center, y_center, w_norm, h_norm


def parse_row_regions(region_obj):
  if region_obj is None:
    return []

  if not isinstance(region_obj, dict):
    return []

  regions = region_obj.get("regions", [])
  parsed = []

  for r in regions:
    shape = r.get("shape_attributes", {})
    attrs = r.get("region_attributes", {})

    if shape.get("name") != "rect":
      continue

    x = shape.get("x")
    y = shape.get("y")
    w = shape.get("width")
    h = shape.get("height")

    if None in [x, y, w, h]:
      continue

    entity_type = attrs.get("Entity type", "Unknown")

    modified_area = attrs.get("Modified area", {})
    if isinstance(modified_area, dict):
      mod_types = [k for k, v in modified_area.items() if v is True]
    else:
      mod_types = []

    parsed.append({
      "x": x,
      "y": y,
      "w": w,
      "h": h,
      "entity_type": entity_type,
      "mod_types": mod_types
    })

  return parsed


def build_yolo_dataset(base_dir, split, out_dir):
  base_dir = Path(base_dir)
  out_dir = Path(out_dir)

  csv_path = base_dir / f"{split}.txt"
  img_dir = base_dir / split

  df = pd.read_csv(csv_path)

  out_img_dir = out_dir / "images" / split
  out_lbl_dir = out_dir / "labels" / split

  ensure_dir(out_img_dir)
  ensure_dir(out_lbl_dir)

  summary_rows = []

  for _, row in df.iterrows():
    image_name = row["image"]
    forged = int(row["forged"])

    src_img = img_dir / image_name
    if not src_img.exists():
      continue

    # Copy image
    dst_img = out_img_dir / image_name
    shutil.copy2(src_img, dst_img)

    # Open image for width/height
    with Image.open(src_img) as im:
      img_w, img_h = im.size

    region_obj = safe_literal_eval(row["forgery annotations"])
    boxes = parse_row_regions(region_obj)

    label_lines = []

    # one-class detection: class 0 = forged region
    for b in boxes:
      x_c, y_c, w_n, h_n = xywh_to_yolo(
        b["x"], b["y"], b["w"], b["h"], img_w, img_h
      )

      label_lines.append(
        f"0 {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}"
      )

    label_path = out_lbl_dir / f"{Path(image_name).stem}.txt"
    with open(label_path, "w", encoding="utf-8") as f:
      for line in label_lines:
        f.write(line + "\n")

    summary_rows.append({
      "image": image_name,
      "image_path": str(dst_img),
      "split": split,
      "forged": forged,
      "num_boxes": len(boxes),
      "digital_annotation": int(row["digital annotation"]),
      "handwritten_annotation": int(row["handwritten annotation"])
    })

  summary_df = pd.DataFrame(summary_rows)
  return summary_df


def main():
  base_dir = r"C:\docfusion_data\findit2"
  out_dir = r"C:\docfusion_data\findit2_yolo"

  all_summaries = []

  for split in ["train", "val", "test"]:
    summary_df = build_yolo_dataset(base_dir, split, out_dir)
    all_summaries.append(summary_df)
    print(f"{split}: {len(summary_df)} images processed")

  full_df = pd.concat(all_summaries, ignore_index=True)
  full_df.to_csv(Path(out_dir) / "dataset_summary.csv", index=False)

  print("\nDone.")
  print(full_df.groupby("split")[["forged", "num_boxes"]].agg(["count", "sum", "mean"]))


if __name__ == "__main__":
  main()