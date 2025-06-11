import easyocr
import cv2
import numpy as np
import re
from typing import Optional

class VietnameseOCR:
    def __init__(self):
        # Chỉ dùng tiếng Anh vì biển số không có ký tự tiếng Việt
        self.reader = easyocr.Reader(['en'], gpu=True)
        # Các pattern để xác thực nếu cần sau này
        self.patterns = [
            r'^[0-9]{2}[A-Z]{1,2}-[0-9]{3,5}$',
            r'^[0-9]{2}[A-Z]{1,2}[0-9]{3,5}$'
        ]

    def preprocess_plate_image(self, plate_img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_LINEAR)
        denoised = cv2.GaussianBlur(resized, (3, 3), 0)
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def extract_text(self, plate_img: np.ndarray) -> Optional[str]:
        try:
           processed = self.preprocess_plate_image(plate_img)
           results = self.reader.readtext(processed)

           if not results:
            return None

            # Sắp xếp theo vị trí y tăng dần (dòng trên trước dòng dưới)
           sorted_lines = sorted(results, key=lambda x: x[0][0][1])

            # ✅ Lấy 2 dòng có độ tin cậy cao nhất
           texts = []
           for line in sorted_lines:
            if line[2] > 0.4:
                texts.append(line[1])
            if len(texts) == 2:
                break
            
           full_text = ' '.join(texts)
           cleaned = self.clean_text(full_text)
           print("🔤 Text sau làm sạch:", cleaned)
           return cleaned if cleaned else None

        except Exception as e:
            print(f"❌ Lỗi OCR: {e}")
            return None


    def clean_text(self, text: str) -> str:
        # Làm sạch ký tự không hợp lệ và chuẩn hóa
        cleaned = re.sub(r'[^A-Z0-9-]', '', text.upper())
        corrections = {
    'O': '0',  # Biển VN không có chữ 'O'
    'I': '1',  # Không có 'I'
    'Q': '0'   # Không có 'Q'
}
        for wrong, correct in corrections.items():
            cleaned = cleaned.replace(wrong, correct)
        return cleaned

    def validate_plate_format(self, text: str) -> bool:
        return any(re.match(pattern, text) for pattern in self.patterns)
