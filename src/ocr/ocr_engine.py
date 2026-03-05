def run_ocr(image_path):
  # Try PaddleOCR first
  try:
    from src.ocr.paddle_ocr import run_ocr as run_paddle
    return run_paddle(image_path)
  except Exception:
    pass

  # Fallback: Tesseract
  from src.ocr.tesseract_ocr import run_ocr as run_tess
  return run_tess(image_path)