import easyocr
import cv2
import numpy as np
import re
from typing import Optional

class VietnameseOCR:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=True)
        self.patterns = [
            r'^[0-9]{2}[A-Z]{1,2}-[0-9]{3,5}$',
            r'^[0-9]{2}[A-Z]{1,2}[0-9]{3,5}$'
        ]

    def preprocess_plate_image(self, plate_img: np.ndarray) -> np.ndarray:
        if len(plate_img.shape) == 3 and plate_img.shape[2] == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        resized = cv2.resize(enhanced, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_LINEAR)
        blur = cv2.GaussianBlur(resized, (3, 3), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def extract_text(self, plate_img: np.ndarray) -> Optional[str]:
        try:
            processed = self.preprocess_plate_image(plate_img)
           

            # Hiển thị để debug nếu cần:
            # cv2.imshow("Processed OCR", processed)
            # cv2.waitKey(0)
            results = self.reader.readtext(processed, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-')

            if not results:
                return None
            sorted_lines = sorted(results, key=lambda x: x[0][0][1])
            texts = [line[1] for line in sorted_lines if line[2] > 0.1] 

            full_text = ' '.join(texts)
            cleaned = self.clean_text(full_text)

            for pattern in self.patterns:
                if re.match(pattern, cleaned):
                    print("Text sau làm sạch:", cleaned)
                    return cleaned

            return cleaned if cleaned else None

        except Exception as e:
            print(f"❌ Lỗi OCR: {e}")
            return None

    def clean_text(self, text: str) -> str:
        cleaned = re.sub(r'[^A-Z0-9-]', '', text.upper())
        corrections = {
            'O': '0',
            'I': '1',
            'Q': '0'
        }
        for wrong, correct in corrections.items():
            cleaned = cleaned.replace(wrong, correct)
        return cleaned

    def validate_plate_format(self, text: str) -> bool:
        return any(re.match(pattern, text) for pattern in self.patterns)
