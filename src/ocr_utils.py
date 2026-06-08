import easyocr
from functools import lru_cache
from typing import List

@lru_cache(maxsize=1)
def get_reader():
    # Languages: adjust list based on dataset
    # 'bg','hr','sr' might not be directly supported; you can still try 'en' + others as fallback
    langs = ["en"]  # start minimal; extend if you know supported codes
    reader = easyocr.Reader(langs, gpu=False)  # CPU OCR is OK; small images
    return reader

def ocr_image_to_text(pil_image) -> str:
    reader = get_reader()
    result = reader.readtext(pil_image, detail=0)
    text = " ".join(result)
    return text.strip()