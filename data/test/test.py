import os
import yaml
import sys
import cv2
import concurrent.futures
import time


# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.plate_detector import MotorbikePlateDetector
from src.preprocessor import ImagePreprocessor

# Hàm hỗ trợ chạy hàm khác có timeout
def run_with_timeout(func, args=(), timeout=10):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print("[⏱️ TIMEOUT] Hàm xử lý mất quá nhiều thời gian và đã bị huỷ.")
            return None

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
    
    # Khởi tạo detector và preprocessor
    detector = MotorbikePlateDetector(model_path, confidence_threshold)
    preprocessor = ImagePreprocessor()
    
    # Xử lý từng ảnh trong thư mục test
    test_images_path = os.path.join(test_path, "images")
    output_dir = paths_config["output_path"]
    os.makedirs(output_dir, exist_ok=True)
    
    def process_image(image_path, detector, output_dir):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Không thể tải ảnh: {image_path}")
            return

        detections = detector.detect_plates(image)
        annotated = detector.annotate_image(image, detections)
        output_path = os.path.join(output_dir, f"annotated_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, annotated)
        print(f"Kết quả đã lưu tại: {output_path}")
 
    # Duyệt ảnh
    for image_file in os.listdir(test_images_path):
       if image_file.endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(test_images_path, image_file)
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(process_image, image_path, detector, output_dir)
                future.result(timeout=10)  # ⏰ Giới hạn xử lý mỗi ảnh là 10 giây
        except concurrent.futures.TimeoutError:
            print(f"[TIMEOUT] Ảnh {image_file} mất quá nhiều thời gian và đã bị bỏ qua.")
        except Exception as e:
            print(f"[ERROR] Lỗi khi xử lý {image_file}: {e}")



if __name__ == "__main__":
    main()
