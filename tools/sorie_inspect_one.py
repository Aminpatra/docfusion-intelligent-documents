from pathlib import Path


def read_text(path):
  return Path(path).read_text(encoding="utf-8", errors="ignore")


def main():
  base = Path(r"C:\docfusion_data\SROIE2019\train")

  # pick one entity file
  ent_files = sorted((base / "entities").glob("*.txt"))
  print("Entity files:", len(ent_files))
  one = ent_files[0]
  print("\nEntity file:", one.name)
  print(read_text(one))

  # matching image usually has same stem
  img_jpg = base / "img" / (one.stem + ".jpg")
  img_png = base / "img" / (one.stem + ".png")
  img = img_jpg if img_jpg.exists() else img_png
  print("Image exists:", img.exists(), "->", img)

  # matching box file
  box = base / "box" / (one.stem + ".txt")
  print("Box exists:", box.exists(), "->", box)
  if box.exists():
    print("\nFirst 5 box lines:")
    lines = read_text(box).splitlines()
    for l in lines[:5]:
      print(l)


if __name__ == "__main__":
  main()