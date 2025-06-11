import cv2
import numpy as np
from typing import Optional

class ImagePreprocessor:
    def __init__(self, target_size=(640, 640)):
        self.target_size = target_size

    # Tăng cường độ sáng và tương phản
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l)
        enhanced_lab = cv2.merge([enhanced_l, a, b])
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # Khử nhiễu ảnh màu
    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # Resize ảnh về kích thước model yêu cầu
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, self.target_size)

    # Tiền xử lý ảnh cho mô hình YOLO (resize bắt buộc)
    def preprocess_for_model(self, image: np.ndarray) -> np.ndarray:
        enhanced = self.enhance_image(image)
        denoised = self.denoise_image(enhanced)
        resized = self.resize_image(denoised)
        return resized

    # Tiền xử lý ảnh biển số nhỏ (để OCR): không resize, tránh méo
    def preprocess_for_ocr(self, plate_img: np.ndarray) -> np.ndarray:
        enhanced = self.enhance_image(plate_img)
        denoised = self.denoise_image(enhanced)
        return denoised
