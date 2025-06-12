import os
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple
from src.ocr_processor import VietnameseOCR
from src.preprocessor import ImagePreprocessor

class MotorbikePlateDetector:
    def __init__(self, model_path: str, confidence_threshold: float = 0.4, ocr_confidence_threshold: float = 0.4):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.ocr_confidence_threshold = ocr_confidence_threshold
        self.ocr = VietnameseOCR()
        self.preprocessor = ImagePreprocessor()

    def detect_plates(self, image: np.ndarray, use_preprocessing: bool = True) -> List[Dict]:
        input_image = self.preprocessor.preprocess_for_model(image) if use_preprocessing else image
        results = self.model.predict(input_image, conf=self.confidence_threshold, verbose=False, imgsz=640)
        detections = []

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])

                # Filter: confidence
                if confidence < self.confidence_threshold:
                    continue

                # Filter: aspect ratio (thích nghi hơn với biển số VN)
                aspect_ratio = (x2 - x1) / max((y2 - y1), 1)
                if aspect_ratio < 1.2 or aspect_ratio > 6.5:  # Thu hẹp range
                    continue

                # Filter: area (thu hẹp range)
                detection_area = (x2 - x1) * (y2 - y1)
                if detection_area < 300 or detection_area > 50000:
                    continue

                plate_crop = self._extract_plate_region(image, x1, y1, x2, y2)
                if plate_crop is None or plate_crop.size == 0:
                    continue

                # Debug: Lưu ảnh cắt để kiểm tra
                debug_path = f"debug_plates/plate_{x1}_{y1}_{confidence:.2f}.jpg"
                os.makedirs("debug_plates", exist_ok=True)
                cv2.imwrite(debug_path, plate_crop)

                plate_text, ocr_confidence = self._perform_ocr(plate_crop)

                detection = {
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'text': plate_text,
                    'ocr_confidence': ocr_confidence,
                    'plate_info': self.ocr.get_plate_info(plate_text) if plate_text else None,
                    'plate_crop': plate_crop
                }

                detections.append(detection)

        return self._filter_detections(detections)

    def _extract_plate_region(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        h, w = image.shape[:2]
        pad_x = max(int((x2 - x1) * 0.1), 5)
        pad_y = max(int((y2 - y1) * 0.15), 5)
        x1_pad = max(x1 - pad_x, 0)
        y1_pad = max(y1 - pad_y, 0)
        x2_pad = min(x2 + pad_x, w)
        y2_pad = min(y2 + pad_y, h)
        return image[y1_pad:y2_pad, x1_pad:x2_pad]

    def _perform_ocr(self, plate_crop: np.ndarray) -> Tuple[str, float]:
        try:
            plate_text = self.ocr.extract_text(plate_crop)
            if not plate_text:
                raw_result = self.ocr.reader.readtext(plate_crop)
                texts = [t[1] for t in raw_result if t[2] > 0.3]
                plate_text = self.ocr.clean_text(''.join(texts)) if texts else None

            if not plate_text:
                return None, 0.0

            if self.ocr.validate_plate_format(plate_text):
                return plate_text, 1.0
            return plate_text, 0.5
        except Exception as e:
            print(f"Lỗi OCR: {e}")
            return None, 0.0

    def _filter_detections(self, detections: List[Dict]) -> List[Dict]:
        if not detections:
            return []

        for det in detections:
            total_conf = det['confidence']
            if det['text']:
                total_conf += det['ocr_confidence'] * 0.5
                if det['plate_info'] and det['plate_info']['valid']:
                    total_conf += 0.3
            det['total_confidence'] = total_conf

        detections.sort(key=lambda x: x['total_confidence'], reverse=True)

        filtered = []
        for det in detections:
            is_duplicate = False
            for existing in filtered:
                if self._calculate_iou(det['bbox'], existing['bbox']) > 0.3:
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered.append(det)

        return filtered

    def _calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0

    def detect_and_recognize(self, image_path: str) -> List[Dict]:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Không thể đọc ảnh: {image_path}")
        return self.detect_plates(image)

    def annotate_image(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        annotated = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            text = det['text']
            plate_info = det['plate_info']

            color = (0, 255, 0) if plate_info and plate_info['valid'] else (0, 165, 255)
            status = "✓" if plate_info and plate_info['valid'] else "?"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"{status} {text}" if text else "Không nhận diện được"
            if plate_info and text:
                label += f" ({plate_info['type']})"

            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.putText(annotated, f"Conf: {confidence:.2f}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return annotated

    def save_detection_results(self, image_path: str, output_dir: str):
        import os
        from pathlib import Path

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        detections = self.detect_and_recognize(image_path)
        image = cv2.imread(image_path)
        annotated = self.annotate_image(image, detections)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_detected.jpg"), annotated)

        for i, det in enumerate(detections):
            if det['plate_crop'] is not None:
                crop_name = f"{base_name}_plate_{i}_{det['text'] or 'unknown'}.jpg"
                crop_name = "".join(c for c in crop_name if c.isalnum() or c in "._-")
                cv2.imwrite(os.path.join(output_dir, crop_name), det['plate_crop'])

        return detections