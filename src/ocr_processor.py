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
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img.copy()

        # Nâng tương phản bằng CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Resize 2x để dễ nhận diện
        resized = cv2.resize(enhanced, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)

        # Làm mịn nhẹ
        blur = cv2.GaussianBlur(resized, (3, 3), 0)

        # Nhị phân hóa bằng Otsu
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def extract_text(self, plate_img: np.ndarray) -> Optional[str]:
        try:
            processed = self.preprocess_plate_image(plate_img)

            # OCR với confidence + position
            results = self.reader.readtext(
                processed,
                allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-',
                detail=1
            )

            if not results:
                return None

            # Lọc bỏ các kết quả có confidence thấp (<0.3)
            filtered = [r for r in results if r[2] > 0.3]
            if not filtered:
                return None

            # Sắp xếp theo trung bình tung độ (để phân biệt dòng trên và dưới)
            sorted_lines = sorted(filtered, key=lambda x: (x[0][0][1] + x[0][2][1]) / 2)

            # Ghép nối text các dòng
            text_parts = [r[1] for r in sorted_lines]
            full_text = ''.join(text_parts)

            cleaned = self.clean_text(full_text)
            return cleaned

        except Exception as e:
            print(f"❌ Lỗi OCR: {e}")
            return None
    

    def clean_text(self, text: str) -> str:
        # Loại bỏ ký tự không hợp lệ
        cleaned = re.sub(r'[^A-Z0-9-]', '', text.upper())

        # Sửa lỗi nhầm ký tự thường gặp
        corrections = {
            'O': '0', 'Q': '0', 'D': '0', 'U': '0',
            'I': '1', 'Z': '2', 'L': '4',
            'S': '5', 'G': '6', 'H': '-', 'T': '7'
        }
        for wrong, correct in corrections.items():
            cleaned = cleaned.replace(wrong, correct)

        return cleaned

    def validate_plate_format(self, text: str) -> bool:
        return any(re.match(pattern, text) for pattern in self.patterns)
