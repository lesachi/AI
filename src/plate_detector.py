import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict
from src.ocr_processor import VietnameseOCR
from src.preprocessor import ImagePreprocessor

class MotorbikePlateDetector:
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.ocr = VietnameseOCR()
        self.preprocessor = ImagePreprocessor()


    def detect_plates(self, image: np.ndarray) -> List[Dict]:
        # Tiền xử lý ảnh cho YOLO (chỉ resize + tăng cường)
        processed_image = self.preprocessor.preprocess_for_model(image)

        results = self.model.predict(processed_image, conf=self.confidence_threshold)
        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])

                # Cắt vùng biển số từ ảnh GỐC để giữ độ sắc nét
                plate_img = image[y1:y2, x1:x2]

                # OCR nhận diện văn bản
                plate_text = self.ocr.extract_text(plate_img)
                if plate_text is None:
                    print(f"⚠️ Không nhận diện được biển số tại vị trí ({x1}, {y1}, {x2}, {y2})")

                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'text': plate_text
                })

        return detections

    def annotate_image(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        annotated = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label_text = det['text'] if det['text'] else "Không nhận diện"
            label = f"{label_text} ({det['confidence']:.2f})"

            # Vẽ khung và label
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 2)
        return annotated
