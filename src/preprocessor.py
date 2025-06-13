import cv2
import numpy as np
from typing import Optional

class ImagePreprocessor:
    def __init__(self, target_size=(640, 640), clip_limit=2.0):
        self.target_size = target_size
        self.clip_limit = clip_limit

    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l)
        enhanced_lab = cv2.merge([enhanced_l, a, b])
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, self.target_size)

    def preprocess_for_model(self, image: np.ndarray) -> np.ndarray:
        enhanced = self.enhance_image(image)
        denoised = self.denoise_image(enhanced)
        resized = self.resize_image(denoised)
        return resized

    def preprocess_for_ocr(self, plate_img: np.ndarray) -> np.ndarray:
        enhanced = self.enhance_image(plate_img)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return morph