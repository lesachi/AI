import easyocr
import cv2
import numpy as np
import re
from typing import Optional, Dict

class VietnameseOCR:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=True)
        self.patterns = [
            r'^[0-9]{2}[A-Z]{1,2}-[0-9]{3,5}$',   # VD: 29A-12345
            r'^[0-9]{2}[A-Z]{1,2}[0-9]{3,5}$',    # VD: 29A12345
            r'^[0-9]{2}[A-Z]{1,2}-[0-9]{2}\.[0-9]{2}$'  # VD: 43A-12.34
        ]

    def preprocess_plate_image(self, plate_img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Giảm clipLimit
        enhanced = clahe.apply(gray)
        blur = cv2.GaussianBlur(enhanced, (3, 3), 0)  # Giảm kernel
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2  # Sử dụng THRESH_BINARY thay THRESH_BINARY_INV
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return morph

    def extract_text(self, plate_img: np.ndarray) -> Optional[str]:
        try:
            processed = self.preprocess_plate_image(plate_img)
            results = self.reader.readtext(processed)

            texts = [r[1] for r in sorted(results, key=lambda x: x[0][0][1]) if r[2] > 0.3]  # Giảm xuống 0.3
            raw_text = ''.join(texts)
            cleaned = self.clean_text(raw_text)
            print("OCR kết quả:", cleaned)
            return cleaned if cleaned else None

        except Exception as e:
            print(f"Lỗi OCR: {e}")
            return None

    def clean_text(self, text: str) -> str:
        text = text.upper()
        corrections = {
            'O': '0', 'Q': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6'
        }
        for wrong, right in corrections.items():
            text = text.replace(wrong, right)
        text = re.sub(r'[^A-Z0-9\-\.]', '', text)
        return text

    def validate_plate_format(self, text: str) -> bool:
        return any(re.match(pattern, text) for pattern in self.patterns)

    def get_plate_info(self, text: str) -> Dict:
        return {
            "valid": self.validate_plate_format(text),
            "length": len(text),
            "type": self.classify_plate_type(text)
        }

    def classify_plate_type(self, text: str) -> str:
        if re.match(r'^[0-9]{2}[A-Z]{1,2}-[0-9]{3,5}$', text):
            return "classic"
        elif re.match(r'^[0-9]{2}[A-Z]{1,2}-[0-9]{2}\.[0-9]{2}$', text):
            return "new"
        return "unknown"