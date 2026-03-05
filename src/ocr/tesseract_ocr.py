import pytesseract
import cv2

def run_ocr(image_path):
  img = cv2.imread(image_path)
  if img is None:
    raise ValueError(f"Image not found: {image_path}")

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  text = pytesseract.image_to_string(gray, lang="eng")
  return text