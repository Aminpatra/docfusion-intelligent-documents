import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.extraction.sroie_extractor import load_boxes_with_pos
from src.extraction.sroie_extractor import extract_vendor, extract_date, extract_total

path = r"C:\docfusion_data\SROIE2019\train\box\X00016469612.txt"

items = load_boxes_with_pos(path)

print("Vendor:", extract_vendor(items))
print("Date:", extract_date(items))
print("Total:", extract_total(items))