from pathlib import Path
import cv2
import numpy as np


def parse_box_line(line):
  # Format: x1,y1,x2,y2,x3,y3,x4,y4,text(with commas sometimes)
  parts = line.split(",")
  coords = list(map(int, parts[:8]))
  text = ",".join(parts[8:]).strip()
  pts = [(coords[i], coords[i + 1]) for i in range(0, 8, 2)]
  return pts, text


def main():
  base = Path(r"C:\docfusion_data\SROIE2019\train")

  ent_files = sorted((base / "entities").glob("*.txt"))
  if not ent_files:
    raise ValueError("No entity files found")

  one = ent_files[0]

  img_path = base / "img" / (one.stem + ".jpg")
  if not img_path.exists():
    img_path = base / "img" / (one.stem + ".png")

  box_path = base / "box" / (one.stem + ".txt")

  img = cv2.imread(str(img_path))
  if img is None:
    raise ValueError(f"Cannot read image: {img_path}")

  lines = Path(box_path).read_text(encoding="utf-8", errors="ignore").splitlines()

  for line in lines:
    pts, _ = parse_box_line(line)
    pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts_np], True, (0, 255, 0), 1)

  out_dir = Path("outputs")
  out_dir.mkdir(exist_ok=True)
  out_path = out_dir / f"{one.stem}_boxes.jpg"
  cv2.imwrite(str(out_path), img)
  print("Wrote:", out_path)


if __name__ == "__main__":
  main()