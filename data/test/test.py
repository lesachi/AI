import os
import yaml
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import cv2
from src.plate_detector import MotorbikePlateDetector
from src.preprocessor import ImagePreprocessor

# Script kiểm tra mô hình trên tập test
def main():
    # Đọc cấu hình từ file config.yaml
    with open(os.path.join(os.path.dirname(__file__), '../../config.yaml'), 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    # Lấy thông tin từ config
    paths_config = config["paths"]
    model_config = config["model"]
    
    test_path = paths_config["test_path"]
    model_path = paths_config["model_path"]
    confidence_threshold = model_config["confidence_threshold"]
    
    # Khởi tạo detector
    detector = MotorbikePlateDetector(model_path, confidence_threshold)
    preprocessor = ImagePreprocessor()
    
    # Xử lý từng ảnh trong thư mục test
    test_images_path = os.path.join(test_path, "images")
    output_dir = paths_config["output_path"]
    os.makedirs(output_dir, exist_ok=True)
    
    for image_file in os.listdir(test_images_path):
        if image_file.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(test_images_path, image_file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Không thể tải ảnh: {image_path}")
                continue
            
            # Phát hiện và ghi chú
            detections = detector.detect_plates(image)
            annotated = detector.annotate_image(image, detections)
            output_path = os.path.join(output_dir, f"annotated_{image_file}")
            cv2.imwrite(output_path, annotated)
            print(f"Kết quả đã lưu tại: {output_path}")

if __name__ == "__main__":
    main()