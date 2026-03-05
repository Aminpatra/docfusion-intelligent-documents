from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang="en")

def run_ocr(image_path):

    result = ocr.ocr(image_path)

    text_lines = []

    for line in result[0]:
        text = line[1][0]
        text_lines.append(text)

    return "\n".join(text_lines)