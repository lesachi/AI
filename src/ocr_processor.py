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
        resized = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        return resized

    def extract_text(self, plate_img: np.ndarray) -> Optional[str]:
        try:
            processed_img = self.preprocess_plate_image(plate_img)
            
            # Debug: Lưu ảnh đã xử lý để kiểm tra nếu cần
            # cv2.imwrite("processed_plate.png", processed_img)

            results = self.reader.readtext(processed_img)
            print("📸 OCR kết quả:", results)

            if not results:
                return None

            # Lấy kết quả có độ tin cậy cao nhất
            best_text = max(results, key=lambda x: x[2])[1]
            cleaned_text = self.clean_text(best_text)

            print("🔤 Text sau làm sạch:", cleaned_text)

            # Tạm thời bỏ kiểm tra định dạng để kiểm tra OCR có hoạt động hay không
            return cleaned_text
            # Nếu muốn bật kiểm tra định dạng sau:
            # return cleaned_text if self.validate_plate_format(cleaned_text) else None

        except Exception as e:
            print(f"❌ Lỗi OCR: {e}")
            return None

    def clean_text(self, text: str) -> str:
        # Làm sạch ký tự không hợp lệ và chuẩn hóa
        cleaned = re.sub(r'[^A-Z0-9-]', '', text.upper())
        corrections = {'O': '0', 'I': '1', 'S': '5'}
        for wrong, correct in corrections.items():
            cleaned = cleaned.replace(wrong, correct)
        return cleaned

    def validate_plate_format(self, text: str) -> bool:
        return any(re.match(pattern, text) for pattern in self.patterns)
