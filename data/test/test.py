import os
import yaml
import sys
import cv2
import concurrent.futures
import time
import re

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

# Hàm đo độ nét ảnh bằng variance of Laplacian
def sharpness_score(image_path):
    try:
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    except:
        return 0

# Hàm lấy key định danh ảnh (ví dụ: "0042_01875" hoặc "IMG202...")
def extract_plate_key(filename):
    match = re.search(r"annotated_(.*?)(_b)?_jpg", filename)
    return match.group(1) if match else None

# Hàm lọc ảnh trùng, giữ ảnh rõ nhất
def filter_duplicate_images(output_dir):
    grouped = {}
    for fname in os.listdir(output_dir):
        if not fname.startswith("annotated_") or not fname.endswith(".jpg"):
            continue
        key = extract_plate_key(fname)
        if key:
            grouped.setdefault(key, []).append(fname)

    print(f"\n🔍 Đang lọc ảnh trùng...")

    for key, files in grouped.items():
        if len(files) <= 1:
            continue
        best_file = None
        max_score = -1
        for f in files:
            path = os.path.join(output_dir, f)
            score = sharpness_score(path)
            if score > max_score:
                max_score = score
                best_file = f
        for f in files:
            if f != best_file:
                try:
                    os.remove(os.path.join(output_dir, f))
                    print(f"❌ Đã xoá ảnh trùng: {f}")
                except Exception as e:
                    print(f"[Lỗi] Không thể xoá {f}: {e}")
        print(f"✔ Giữ lại ảnh rõ nhất: {best_file}")

    print(f"\n✅ Đã lọc xong ảnh trùng lặp.\n")

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

    # ✅ Sau khi xử lý xong ảnh, lọc ảnh trùng:
    filter_duplicate_images(output_dir)


if __name__ == "__main__":
    main()
