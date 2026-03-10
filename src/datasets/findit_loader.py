from pathlib import Path
import pandas as pd


def load_findit_split(base_dir, split="train"):
    """
    Load one Find-It-Again split.

    Supports the CSV-style annotation files where:
    - base_dir/train.txt, val.txt, test.txt are actually comma-separated tables
    - split folder contains OCR text files and images

    Returns a dataframe with:
    id, image_path, lines, is_forged
    """

    base = Path(base_dir)
    split_dir = base / split
    ann_file = base / f"{split}.txt"

    # Read annotation table
    ann = pd.read_csv(ann_file)

    # Normalize column names just in case
    ann.columns = [c.strip().lower() for c in ann.columns]

    # Try to identify important columns
    image_col = None
    forged_col = None

    for c in ann.columns:
        if c in ["image", "image_name", "filename", "file_name"]:
            image_col = c
        if c in ["forged", "is_forged", "label"]:
            forged_col = c

    if image_col is None:
        raise ValueError(f"Could not find image column in {ann_file}. Columns: {list(ann.columns)}")

    if forged_col is None:
        raise ValueError(f"Could not find forged label column in {ann_file}. Columns: {list(ann.columns)}")

    rows = []

    for _, row in ann.iterrows():
        image_name = str(row[image_col]).strip()
        doc_id = Path(image_name).stem
        is_forged = int(row[forged_col])

        txt_file = split_dir / f"{doc_id}.txt"

        # Try common image extensions
        image_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = split_dir / f"{doc_id}{ext}"
            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            image_path = split_dir / image_name  # fallback

        text_lines = []
        if txt_file.exists():
            with open(txt_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        text_lines.append(line)

        rows.append({
            "id": doc_id,
            "image_path": str(image_path),
            "lines": text_lines,
            "is_forged": is_forged,
        })

    df = pd.DataFrame(rows)
    return df